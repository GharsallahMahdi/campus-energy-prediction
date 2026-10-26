"""
ashrae_experiment.py
--------------------
Runs the same kind of model comparison as evaluate_models.py, but on REAL
building electricity data (ASHRAE Great Energy Predictor III, from Kaggle).

Buildings used: up to 30 "Education" buildings (the closest match to a campus)
that have clean, complete electricity readings for the year 2016.

Time-based split: train on Jan-Oct 2016, test on Nov-Dec 2016.

Needs, inside data/ashrae/:
  train.csv               (required)
  building_metadata.csv   (required)
  weather_train.csv       (optional - if present, air temperature is used too)

Run from the project root:   python3 src/ashrae_experiment.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT = Path(__file__).resolve().parent.parent
ASHRAE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / "data" / "ashrae"
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT / "data"

N_BUILDINGS = 30
TEST_START = pd.Timestamp("2016-11-01")

# ---------- 0. Check the files ----------
for name in ["train.csv", "building_metadata.csv"]:
    if not (ASHRAE_DIR / name).exists():
        sys.exit(f"Missing file: {ASHRAE_DIR / name}\nPut the Kaggle files in that folder.")

# ---------- 1. Choose buildings ----------
meta = pd.read_csv(ASHRAE_DIR / "building_metadata.csv")
edu = meta[meta["primary_use"] == "Education"]
print(f"Education buildings in the dataset: {len(edu)}")

# ---------- 2. Read electricity readings (in chunks, to save memory) ----------
print("Reading train.csv (this can take a minute)...")
edu_ids = set(edu["building_id"])
chunks = []
for chunk in pd.read_csv(
    ASHRAE_DIR / "train.csv",
    usecols=["building_id", "meter", "timestamp", "meter_reading"],
    dtype={"building_id": "int32", "meter": "int8", "meter_reading": "float32"},
    chunksize=2_000_000,
):
    chunk = chunk[(chunk["meter"] == 0) & (chunk["building_id"].isin(edu_ids))]
    chunks.append(chunk)
raw = pd.concat(chunks, ignore_index=True)
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
print(f"Electricity rows for Education buildings: {len(raw):,}")

# ---------- 3. Keep only clean buildings ----------
hours_in_year = 8784
stats = raw.groupby("building_id")["meter_reading"].agg(
    n="size", zero_frac=lambda s: (s <= 0).mean(),
    median="median", max="max",
)
stats["missing_frac"] = 1 - stats["n"] / hours_in_year
# Outlier check: if a building's single highest reading is many times its
# typical (median) reading, treat it as a sensor glitch rather than real
# demand - a few such spikes can distort RMSE for the whole dataset.
stats["spike_ratio"] = stats["max"] / stats["median"].replace(0, np.nan)
clean = stats[(stats["zero_frac"] < 0.05) & (stats["missing_frac"] < 0.05)
             & (stats["spike_ratio"] < 15)]
n_spiky = ((stats["zero_frac"] < 0.05) & (stats["missing_frac"] < 0.05)
          & (stats["spike_ratio"] >= 15)).sum()
print(f"Buildings with clean, complete data: {len(clean)} "
     f"(excluded {n_spiky} more for extreme meter spikes)")
if len(clean) == 0:
    sys.exit("No clean buildings found - please tell your assistant.")

rng = np.random.RandomState(42)
chosen = rng.choice(clean.index.values, size=min(N_BUILDINGS, len(clean)), replace=False)
raw = raw[raw["building_id"].isin(chosen)]
raw = raw[raw["meter_reading"] > 0]

# ---------- 4. Build a complete hourly grid so lags are correct ----------
full_index = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
grid = pd.MultiIndex.from_product([sorted(chosen), full_index],
                                  names=["building_id", "timestamp"]).to_frame(index=False)
df = grid.merge(raw[["building_id", "timestamp", "meter_reading"]],
                on=["building_id", "timestamp"], how="left")
df = df.merge(meta[["building_id", "site_id", "square_feet"]], on="building_id", how="left")
df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

# ---------- 5. Features ----------
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["month"] = df["timestamp"].dt.month
g = df.groupby("building_id")["meter_reading"]
df["lag_24"] = g.shift(24)     # same hour yesterday
df["lag_168"] = g.shift(168)   # same hour last week

use_weather = (ASHRAE_DIR / "weather_train.csv").exists()
if use_weather:
    w = pd.read_csv(ASHRAE_DIR / "weather_train.csv", usecols=["site_id", "timestamp", "air_temperature"])
    w["timestamp"] = pd.to_datetime(w["timestamp"])
    df = df.merge(w, on=["site_id", "timestamp"], how="left")
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    df["air_temperature"] = df.groupby("building_id")["air_temperature"].transform(
        lambda s: s.interpolate(limit_direction="both"))
    print("Weather file found: air temperature will be used.")
else:
    print("No weather_train.csv found: models run without temperature.")

# ---------- 6. Time-based split ----------
train = df[(df["timestamp"] < TEST_START) & df["meter_reading"].notna()].copy()
test = df[(df["timestamp"] >= TEST_START) & df["meter_reading"].notna()].copy()

# The building's typical size level, learned from the TRAINING period only
b_mean = train.groupby("building_id")["meter_reading"].mean().rename("building_mean")
df = df.join(b_mean, on="building_id")
train = train.join(b_mean, on="building_id")
test = test.join(b_mean, on="building_id")

test_lag = test.dropna(subset=["lag_24", "lag_168"])
train_lag = train.dropna(subset=["lag_24", "lag_168"])
test = test.loc[test_lag.index]          # compare all models on the same rows
print(f"Train rows: {len(train):,}  |  Test rows: {len(test):,}  "
      f"({TEST_START.date()} to 2016-12-31)\n")

y_test = test["meter_reading"]
# Range used to normalize errors, exactly as in the reference paper's
# NMAE/NRMSE formulas: (max - min) of the actual values being predicted.
y_range = y_test.max() - y_test.min()
results, predictions = {}, {}


def score(name, y_pred):
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {
        "MAE (kWh)": mae,
        "NMAE": mae / y_range,
        "RMSE (kWh)": rmse,
        "NRMSE": rmse / y_range,
        "MAPE (%)": np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-6))) * 100,
        "R2": r2_score(y_test, y_pred),
    }
    predictions[name] = y_pred


# ---------- 7. Baselines ----------
score("Mean baseline", test["building_mean"])

profile = (train.groupby(["building_id", "is_weekend", "hour"])["meter_reading"]
           .mean().rename("profile"))
score("Hourly-profile baseline",
      test.join(profile, on=["building_id", "is_weekend", "hour"])["profile"])

score("Last-week baseline (same hour)", test["lag_168"])

# ---------- 8. Machine learning models ----------
base = ["hour", "day_of_week", "is_weekend", "month", "square_feet", "building_mean"]
lags = ["lag_24", "lag_168"]
temp = ["air_temperature"] if use_weather else []


def fit_rf(features, name, training):
    print(f"Training: {name} ...")
    m = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
    m.fit(training[features], training["meter_reading"])
    score(name, m.predict(test[features]))
    return m


fit_rf(base, "Random Forest (time features)", train)
fit_rf(base + lags, "Random Forest + lags", train_lag)
if use_weather:
    fit_rf(base + lags + temp, "Random Forest + lags + temperature", train_lag)

print("Training: Gradient Boosting + lags ...")
hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42)
hgb_features = base + lags + temp
hgb.fit(train_lag[hgb_features], train_lag["meter_reading"])
score("Gradient Boosting + lags" + (" + temperature" if use_weather else ""),
      hgb.predict(test[hgb_features]))

# ---------- 8b. LSTM - same methodology as the simulated-data experiments,
# so this is directly comparable to forecast_buildings_j9.py's results ----------
print("Training: LSTM ...")
import tensorflow as tf
from tensorflow import keras

LSTM_WINDOW = 24
df["scaled"] = df["meter_reading"] / df["building_mean"]
df["hour_sin"] = np.sin(df["hour"] / 24 * 2 * np.pi)
df["hour_cos"] = np.cos(df["hour"] / 24 * 2 * np.pi)
lstm_channels = ["scaled", "hour_sin", "hour_cos", "is_weekend"]
if use_weather:
    df["temp_scaled"] = (df["air_temperature"] - df["air_temperature"].mean()) / df["air_temperature"].std()
    lstm_channels.append("temp_scaled")

X, y, meta_rows = [], [], []
for bid in sorted(chosen):
    bdf = df[df.building_id == bid].sort_values("timestamp")
    vals = bdf[lstm_channels].fillna(0).values
    target = bdf["scaled"].values
    times = bdf["timestamp"].values
    for i in range(LSTM_WINDOW, len(vals)):
        if pd.isna(bdf["meter_reading"].iloc[i]):
            continue
        X.append(vals[i - LSTM_WINDOW:i])
        y.append(target[i])
        meta_rows.append((bid, times[i]))
X, y = np.array(X), np.array(y)
meta_rows = pd.DataFrame(meta_rows, columns=["building_id", "timestamp"])
meta_rows["timestamp"] = pd.to_datetime(meta_rows["timestamp"])
is_test_row = meta_rows["timestamp"] >= TEST_START
X_train_lstm, y_train_lstm = X[~is_test_row.values], y[~is_test_row.values]
X_test_lstm = X[is_test_row.values]
meta_test_lstm = meta_rows[is_test_row.values].reset_index(drop=True)

lstm = keras.Sequential([
    keras.layers.Input(shape=(LSTM_WINDOW, len(lstm_channels))),
    keras.layers.LSTM(32),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=512, verbose=2)

lstm_pred_scaled = lstm.predict(X_test_lstm, batch_size=1024, verbose=0).flatten()
meta_test_lstm = meta_test_lstm.merge(
    df.drop_duplicates("building_id")[["building_id", "building_mean"]], on="building_id")
lstm_pred_kwh = lstm_pred_scaled * meta_test_lstm["building_mean"].values

# Align to the SAME test rows the other models were scored on (test_lag's
# index order), so every model in the table is compared on identical data.
lstm_lookup = meta_test_lstm.copy()
lstm_lookup["pred"] = lstm_pred_kwh
test_with_lstm = test.merge(lstm_lookup[["building_id", "timestamp", "pred"]],
                            on=["building_id", "timestamp"], how="left")
# Rows with no LSTM prediction (first 24h of each building's test data,
# before a full window exists) are dropped only for the LSTM score - the
# other models already have a full lag-based test set.
lstm_valid = test_with_lstm["pred"].notna()
y_test_lstm_subset = test_with_lstm.loc[lstm_valid, "meter_reading"]
pred_lstm_subset = test_with_lstm.loc[lstm_valid, "pred"].values
mae_l = mean_absolute_error(y_test_lstm_subset, pred_lstm_subset)
rmse_l = np.sqrt(mean_squared_error(y_test_lstm_subset, pred_lstm_subset))
results["LSTM" + (" + temperature" if use_weather else "")] = {
    "MAE (kWh)": mae_l, "NMAE": mae_l / y_range,
    "RMSE (kWh)": rmse_l, "NRMSE": rmse_l / y_range,
    "MAPE (%)": np.mean(np.abs((y_test_lstm_subset - pred_lstm_subset)
                               / np.maximum(y_test_lstm_subset, 1e-6))) * 100,
    "R2": r2_score(y_test_lstm_subset, pred_lstm_subset),
}
predictions["LSTM" + (" + temperature" if use_weather else "")] = \
    test_with_lstm["pred"].fillna(test_with_lstm["building_mean"]).values

# ---------- 9. Results ----------
table = pd.DataFrame(results).T.round(3)
print("\n=== ASHRAE real-data comparison (test period: Nov-Dec 2016) ===")
print(table.to_string())
OUT_DIR.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT_DIR / "ashrae_results.csv")
best = table["MAE (kWh)"].idxmin()
print(f"\nBest model by MAE: {best}")

# ---------- 9b. Per-building errors for the best model (to spot outliers) ----------
per_b = test[["building_id", "meter_reading"]].copy()
per_b["abs_error"] = np.abs(y_test.values - predictions[best])
per_building = (per_b.groupby("building_id")
                 .agg(mean_kwh=("meter_reading", "mean"),
                      mae=("abs_error", "mean"),
                      max_error=("abs_error", "max"))
                 .sort_values("mae", ascending=False))
per_building.to_csv(OUT_DIR / "ashrae_per_building_errors.csv")
print(f"\n=== 5 hardest-to-predict buildings ({best}) ===")
print(per_building.head(5).round(1).to_string())
print(f"\n=== 5 easiest-to-predict buildings ({best}) ===")
print(per_building.tail(5).round(1).to_string())

# ---------- 10. Plot one building (last 14 days) ----------
b = test["building_id"].iloc[0]
mask = ((test["building_id"] == b) & (test["timestamp"] >= "2016-12-18")).to_numpy()
best_name = table["MAE (kWh)"].idxmin()
plt.figure(figsize=(11, 4.5))
plt.plot(test.loc[mask, "timestamp"], y_test[mask], label="Actual", color="black", linewidth=1.6)
plt.plot(test.loc[mask, "timestamp"], predictions["Hourly-profile baseline"][mask],
         label="Hourly-profile baseline", linestyle="--", color="#e3b23c")
plt.plot(test.loc[mask, "timestamp"], predictions[best_name][mask], label=best_name, color="#2f7d5f")
plt.title(f"Real building {b} (ASHRAE): actual vs predicted, last 14 days of 2016")
plt.ylabel("Electricity (kWh)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "ashrae_plot.png", dpi=150)
print(f"Saved {OUT_DIR / 'ashrae_results.csv'} and {OUT_DIR / 'ashrae_plot.png'}")
