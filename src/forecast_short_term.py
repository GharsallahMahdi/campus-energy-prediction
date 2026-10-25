"""
forecast_short_term.py  (v2 - revised after supervisor feedback)
-------------------------------------------------------------------
Short-term forecasting: predicts the NEXT 24 HOURS and the NEXT 7 DAYS of
electricity consumption, comparing two models:
  - Random Forest
  - LSTM

Changes from the first version, based on feedback:

 1. WEATHER: temperature and humidity are now built from REAL, published
    Shanghai climate normals (monthly averages), not an invented curve.
    Sources: climate-zone.com (8-year historical monthly averages) and
    weatherbox.io (26-year ECMWF reanalysis humidity range). Hour-to-hour
    variation within each day is still modeled (true hourly historical
    records for these calendar dates are not available), but the monthly
    baseline itself is real. This should be cited in the report.

 2. NO LEAKAGE / genuine multi-day-ahead forecast: this version adds a
    RECURSIVE forecast. From a single forecast origin (the start of the
    test week), it predicts hour by hour, and whenever a lag feature would
    need a value that falls INSIDE the forecast horizon (not yet known),
    it uses the model's own earlier prediction instead of the real value.
    The old "oracle" style (always using the real recent reading) is kept
    too, clearly labeled, as a best-case reference - the gap between the
    two shows how much of the original accuracy came from "peeking".

 3. FAIR COMPARISON: the LSTM now receives the same calendar + weather
    inputs as Random Forest at every hour of its input window, not just
    raw past consumption.

 4. MAPE: rows with very low actual consumption (<2 kWh) are excluded
    from the MAPE calculation, since dividing by a near-zero value makes
    the percentage error explode and distorts the average. The count of
    excluded rows is reported for transparency.

Reads:  ../data/campus_energy_simulated.csv, ../data/building_metadata.csv
Writes: ../data/short_term_results.csv
       ../data/short_term_plot.png

Run from the project root:  python3 src/forecast_short_term.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_DAYS = 7
LSTM_WINDOW = 24
MAPE_MIN_KWH = 2.0   # rows below this actual value are excluded from MAPE only

# ---------- 1. Load data ----------
df = pd.read_csv(DATA_DIR / "campus_energy_simulated.csv",
                 dtype={"building_id": str}, parse_dates=["timestamp"])
meta = pd.read_csv(DATA_DIR / "building_metadata.csv", dtype={"building_id": str})
df = df.merge(meta[["building_id", "building_type", "floor_area_m2"]],
              on="building_id", how="left")
df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
all_ids = sorted(df["building_id"].unique())

# ---------- 2. Weather grounded in REAL Shanghai climate normals ----------
MONTHLY_TEMP_C = {1: 5.0, 2: 5.6, 3: 9.4, 4: 15.0, 5: 20.6, 6: 23.9,
                  7: 28.3, 8: 27.8, 9: 24.4, 10: 19.4, 11: 13.3, 12: 7.8}
MONTHLY_HUMIDITY = {1: 70, 2: 72, 3: 75, 4: 76, 5: 78, 6: 81,
                    7: 80, 8: 80, 9: 76, 10: 72, 11: 70, 12: 69}

np.random.seed(7)
month = df["timestamp"].dt.month.values
hour = df["hour"].values
monthly_base_temp = np.array([MONTHLY_TEMP_C[m] for m in month])
monthly_base_hum = np.array([MONTHLY_HUMIDITY[m] for m in month])

daily_swing = 4 * np.sin((hour - 9) / 24 * 2 * np.pi)
temp = monthly_base_temp + daily_swing + np.random.normal(0, 1.2, len(df))
humidity = np.clip(monthly_base_hum - 0.5 * daily_swing
                   + np.random.normal(0, 4, len(df)), 15, 100)
df["temperature_c"] = temp.round(1)
df["humidity_pct"] = humidity.round(1)

# ---------- 3. Lag features (used by the ORACLE benchmark only) ----------
g = df.groupby("building_id")["consumption_kwh"]
df["lag_24"] = g.shift(24)
df["lag_168"] = g.shift(168)

# ---------- 4. Time-based split ----------
RF_TRAIN_DAYS = 45
last_day = df["timestamp"].dt.normalize().max()
split_date = last_day - pd.Timedelta(days=TEST_DAYS - 1)
rf_train_start = split_date - pd.Timedelta(days=RF_TRAIN_DAYS)
first_test_day_end = split_date + pd.Timedelta(hours=23)

print(f"Forecast origin (last known hour): {(split_date - pd.Timedelta(hours=1))}")
print(f"Test / forecast horizon: {split_date.date()} to {last_day.date()} "
     f"({TEST_DAYS} days = {TEST_DAYS * 24} hours)\n")

type_dummies = pd.get_dummies(df["building_type"], prefix="type")
df = pd.concat([df, type_dummies], axis=1)
type_cols = list(type_dummies.columns)
cal_weather_cols = ["hour", "day_of_week", "is_weekend", "month",
                   "floor_area_m2", "temperature_c", "humidity_pct"]


def mape_safe(y_true, y_pred):
    mask = y_true >= MAPE_MIN_KWH
    excluded = int((~mask).sum())
    if mask.sum() == 0:
        return float("nan"), excluded
    val = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return val, excluded


def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mape, excluded = mape_safe(y_true, y_pred)
    return {
        "MAE (kWh)": mean_absolute_error(y_true, y_pred),
        "RMSE (kWh)": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE (%)": mape,
        "MAPE excl. rows": excluded,
        "R2": r2_score(y_true, y_pred),
    }


# =========================================================
# PART A: Random Forest - trained once on a recent rolling window
# =========================================================
print("Training Random Forest...")
rf_feature_cols = cal_weather_cols + ["lag_24", "lag_168"] + type_cols
train_rf = df[(df["timestamp"] < split_date) & (df["timestamp"] >= rf_train_start)] \
    .dropna(subset=["lag_24", "lag_168"])
rf = RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=3,
                           random_state=42, n_jobs=-1)
rf.fit(train_rf[rf_feature_cols], train_rf["consumption_kwh"])

# ---- A1: ORACLE benchmark (uses the REAL lag values throughout - best
#          case, NOT a genuine blind forecast; kept to show the gap) ----
test_all = df[(df["timestamp"] >= split_date)].dropna(subset=["lag_24", "lag_168"]).copy()
test_all["rf_oracle_pred"] = rf.predict(test_all[rf_feature_cols])
mask_24h = test_all["timestamp"] <= first_test_day_end
results = {}
results["RF oracle (uses real recent data) - next 24h"] = \
    score(test_all.loc[mask_24h, "consumption_kwh"], test_all.loc[mask_24h, "rf_oracle_pred"])
results["RF oracle (uses real recent data) - next 7 days"] = \
    score(test_all["consumption_kwh"], test_all["rf_oracle_pred"])

# =========================================================
# PART B: GENUINE recursive forecast (no leakage) - RF and LSTM together
# =========================================================
print("Preparing the genuine (recursive, no-leakage) forecast...")
print("Building the LSTM's training sequences and training it (this is the slow part)...")

import tensorflow as tf
from tensorflow import keras

feature_channels = ["scaled", "hour_sin", "hour_cos", "is_weekend",
                    "temperature_scaled"]

b_mean = df[(df["timestamp"] < split_date) & (df["timestamp"] >= rf_train_start)] \
    .groupby("building_id")["consumption_kwh"].mean().rename("b_mean")
df = df.join(b_mean, on="building_id")
df["b_mean"] = df["b_mean"].fillna(df["b_mean"].mean())
df["scaled"] = df["consumption_kwh"] / df["b_mean"]
df["hour_sin"] = np.sin(df["hour"] / 24 * 2 * np.pi)
df["hour_cos"] = np.cos(df["hour"] / 24 * 2 * np.pi)
df["temperature_scaled"] = (df["temperature_c"] - df["temperature_c"].mean()) / df["temperature_c"].std()


def make_multivariate_sequences(frame, building_ids):
    X, y, meta_rows = [], [], []
    for bid in building_ids:
        bdf = frame[frame.building_id == bid].sort_values("timestamp")
        vals = bdf[feature_channels].values
        target = bdf["scaled"].values
        times = bdf["timestamp"].values
        for i in range(LSTM_WINDOW, len(vals)):
            X.append(vals[i - LSTM_WINDOW:i])
            y.append(target[i])
            meta_rows.append((bid, times[i]))
    return np.array(X), np.array(y), meta_rows


full_sorted = df.sort_values(["building_id", "timestamp"])
X_all, y_all, meta_rows = make_multivariate_sequences(full_sorted, all_ids)
meta_rows = pd.DataFrame(meta_rows, columns=["building_id", "timestamp"])
meta_rows["timestamp"] = pd.to_datetime(meta_rows["timestamp"])
is_test = meta_rows["timestamp"] >= split_date
X_train_lstm, y_train_lstm = X_all[~is_test.values], y_all[~is_test.values]

lstm = keras.Sequential([
    keras.layers.Input(shape=(LSTM_WINDOW, len(feature_channels))),
    keras.layers.LSTM(32),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=512, verbose=2)

# ---- B. Recursive forecast loop (shared logic, both models) ----
print("Running the recursive (hour-by-hour, no-leakage) forecast for both models...")

horizon_hours = pd.date_range(split_date, periods=TEST_DAYS * 24, freq="h")
history_actual = df.set_index(["building_id", "timestamp"])["consumption_kwh"]
rf_known = {bid: history_actual.loc[bid].copy() for bid in all_ids}
lstm_known_scaled = {bid: df[df.building_id == bid].set_index("timestamp")["scaled"].copy()
                    for bid in all_ids}
b_mean_map = df.drop_duplicates("building_id").set_index("building_id")["b_mean"]

rf_rows_by_time = {}
lstm_rows_by_time = {}

static_lookup = meta.set_index("building_id")
weather_lookup = df.drop_duplicates("timestamp").set_index("timestamp")[
    ["temperature_c", "humidity_pct", "hour_sin", "hour_cos", "temperature_scaled"]]

for t in horizon_hours:
    w = weather_lookup.loc[t]
    # ---- Random Forest step: build one feature row per building ----
    rf_batch = []
    for bid in all_ids:
        b = static_lookup.loc[bid]
        lag24_t = t - pd.Timedelta(hours=24)
        lag168_t = t - pd.Timedelta(hours=168)
        lag24 = rf_known[bid].get(lag24_t, np.nan)
        lag168 = rf_known[bid].get(lag168_t, np.nan)
        row = {
            "hour": t.hour, "day_of_week": t.dayofweek,
            "is_weekend": int(t.dayofweek >= 5), "month": t.month,
            "floor_area_m2": b["floor_area_m2"],
            "temperature_c": w["temperature_c"], "humidity_pct": w["humidity_pct"],
            "lag_24": lag24, "lag_168": lag168,
        }
        for c in type_cols:
            row[c] = 1.0 if c == f"type_{b['building_type']}" else 0.0
        rf_batch.append(row)
    rf_batch_df = pd.DataFrame(rf_batch)[rf_feature_cols]
    rf_preds = rf.predict(rf_batch_df)
    rf_rows_by_time[t] = dict(zip(all_ids, rf_preds))
    for bid, pred in zip(all_ids, rf_preds):
        rf_known[bid].loc[t] = pred

    # ---- LSTM step: build one input window per building ----
    lstm_batch = []
    for bid in all_ids:
        series = lstm_known_scaled[bid]
        window_times = pd.date_range(t - pd.Timedelta(hours=LSTM_WINDOW), periods=LSTM_WINDOW, freq="h")
        scaled_vals = series.reindex(window_times).values
        hs = np.sin(window_times.hour.values / 24 * 2 * np.pi)
        hc = np.cos(window_times.hour.values / 24 * 2 * np.pi)
        iwk = (window_times.dayofweek.values >= 5).astype(float)
        temps = weather_lookup.reindex(window_times)["temperature_scaled"].values
        lstm_batch.append(np.stack([scaled_vals, hs, hc, iwk, temps], axis=1))
    lstm_batch = np.array(lstm_batch, dtype=float)
    lstm_preds_scaled = lstm.predict(lstm_batch, verbose=0).flatten()
    for bid, pred_scaled in zip(all_ids, lstm_preds_scaled):
        lstm_known_scaled[bid].loc[t] = pred_scaled
    lstm_preds_kwh = lstm_preds_scaled * b_mean_map.loc[all_ids].values
    lstm_rows_by_time[t] = dict(zip(all_ids, lstm_preds_kwh))

# ---- Assemble recursive results into a results table ----
rf_long = pd.DataFrame(rf_rows_by_time).T.stack().rename("rf_pred").reset_index()
rf_long.columns = ["timestamp", "building_id", "rf_pred"]
lstm_long = pd.DataFrame(lstm_rows_by_time).T.stack().rename("lstm_pred").reset_index()
lstm_long.columns = ["timestamp", "building_id", "lstm_pred"]

actual = df[df.timestamp.isin(horizon_hours)][["building_id", "timestamp", "consumption_kwh"]]
merged = actual.merge(rf_long, on=["building_id", "timestamp"]).merge(
    lstm_long, on=["building_id", "timestamp"])
merged_24h = merged[merged["timestamp"] <= first_test_day_end]

# ---------- Export per-building forecast curves for the dashboard ----------
# This is what lets clicking a building on the map show ITS OWN genuine
# next-24h / next-7-days forecast (actual vs RF vs LSTM), not just an
# average pattern. "merged" already has every building's hour-by-hour
# recursive forecast from Part B above - we just need to save it per
# building in a compact format.
import json as _json
forecast_export = {}
for bid, bdf in merged.sort_values("timestamp").groupby("building_id"):
    forecast_export[bid] = {
        "hours": list(range(len(bdf))),                       # 0..167
        "actual": bdf["consumption_kwh"].round(1).tolist(),
        "rf_pred": bdf["rf_pred"].round(1).tolist(),
        "lstm_pred": bdf["lstm_pred"].round(1).tolist(),
    }
forecast_meta = {
    "forecast_start": str(split_date),
    "first_24h_index": int((first_test_day_end - split_date).total_seconds() // 3600),
    "buildings": forecast_export,
}
with open(DATA_DIR / "forecast_results.json", "w") as f:
    _json.dump(forecast_meta, f)
print(f"Exported per-building forecast curves to "
     f"{DATA_DIR / 'forecast_results.json'} ({len(forecast_export)} buildings)")

results["Random Forest (genuine, no leakage) - next 24h"] = \
    score(merged_24h["consumption_kwh"], merged_24h["rf_pred"])
results["Random Forest (genuine, no leakage) - next 7 days"] = \
    score(merged["consumption_kwh"], merged["rf_pred"])
results["LSTM (genuine, no leakage) - next 24h"] = \
    score(merged_24h["consumption_kwh"], merged_24h["lstm_pred"])
results["LSTM (genuine, no leakage) - next 7 days"] = \
    score(merged["consumption_kwh"], merged["lstm_pred"])

# ---------- 5. Results table ----------
table = pd.DataFrame(results).T
table_display = table.copy()
table_display[["MAE (kWh)", "RMSE (kWh)", "MAPE (%)", "R2"]] = \
    table_display[["MAE (kWh)", "RMSE (kWh)", "MAPE (%)", "R2"]].astype(float).round(3)
print("\n=== Short-term forecasting comparison ===")
print(table_display.to_string())
table_display.to_csv(DATA_DIR / "short_term_results.csv")

# ---------- 6. Plot: one building, genuine recursive forecast ----------
b = all_ids[0]
one = merged[merged.building_id == b].sort_values("timestamp")
plt.figure(figsize=(12, 4.8))
plt.plot(one["timestamp"], one["consumption_kwh"], label="Actual", color="black", linewidth=1.6)
plt.plot(one["timestamp"], one["rf_pred"], label="Random Forest (genuine)", color="#3d7ab5", linewidth=1.6)
plt.plot(one["timestamp"], one["lstm_pred"], label="LSTM (genuine)", color="#2f7d5f", linewidth=1.6)
plt.axvline(first_test_day_end, color="gray", linestyle="--", linewidth=1,
           label="End of 'next 24h' window")
plt.title(f"Building {b}: genuine 7-day-ahead forecast (no leakage), RF vs LSTM")
plt.ylabel("Consumption (kWh)")
plt.legend()
plt.tight_layout()
plt.savefig(DATA_DIR / "short_term_plot.png", dpi=150)
print(f"\nSaved {DATA_DIR / 'short_term_results.csv'} and {DATA_DIR / 'short_term_plot.png'}")
