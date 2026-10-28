"""
forecast_buildings_j9.py
--------------------------
A fast, SCOPED version of the recursive no-leakage forecast (RF + LSTM),
restricted to Computer Building (J) and Dormitory 9 - the two buildings
used on the focused campus map.

WEATHER: the training period (the ~45 days before the forecast week)
uses simulated weather grounded in real Shanghai climate normals, since
true hourly records for that period don't exist either. The FORECAST
WEEK itself uses REAL recorded Shanghai weather when available (run
fetch_real_weather.py first) via weather_source.py - this is the actual
input the model uses to make its predictions, which is what matters most.
Both temperature AND humidity are used as model features.

Updates (merges into, not overwrites) forecast_results.json, so the other
35 buildings' forecasts - if you ever use the full campus-wide dashboard -
are left untouched.

Reads:  ../data/campus_energy_simulated.csv, ../data/building_metadata.csv,
       ../data/real_weather_shanghai.csv (if present, via weather_source.py)
       ../data/forecast_results.json (merged into)
Writes: ../data/forecast_results.json   (J and 9 entries updated)
       ../data/forecast_buildings_j9_metrics.csv

Run from the project root:  python3 src/forecast_buildings_j9.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from weather_source import get_horizon_weather, MONTHLY_TEMP_C, MONTHLY_HUMIDITY

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BUILDING_IDS = ["J", "9"]
TEST_DAYS = 7
LSTM_WINDOW = 24
RF_TRAIN_DAYS = 45

# ---------- 1. Load data ----------
df = pd.read_csv(DATA_DIR / "campus_energy_simulated.csv",
                 dtype={"building_id": str}, parse_dates=["timestamp"])
meta = pd.read_csv(DATA_DIR / "building_metadata.csv", dtype={"building_id": str})
df = (df[df.building_id.isin(BUILDING_IDS)]
     .merge(meta[["building_id", "building_type", "floor_area_m2"]], on="building_id")
     .sort_values(["building_id", "timestamp"]).reset_index(drop=True))

# ---------- 2. Weather for the TRAINING period: simulated, grounded in
# real monthly climate normals (shared constants from weather_source.py) ----------
np.random.seed(7)
month = df["timestamp"].dt.month.values
hour = df["hour"].values
base_temp = np.array([MONTHLY_TEMP_C[m] for m in month])
base_hum = np.array([MONTHLY_HUMIDITY[m] for m in month])
daily_swing = 4 * np.sin((hour - 9) / 24 * 2 * np.pi)
df["temperature_c"] = (base_temp + daily_swing + np.random.normal(0, 1.2, len(df))).round(1)
df["humidity_pct"] = np.clip(base_hum - 0.5 * daily_swing
                             + np.random.normal(0, 4, len(df)), 15, 100).round(1)

g = df.groupby("building_id")["consumption_kwh"]
df["lag_24"] = g.shift(24)
df["lag_168"] = g.shift(168)

last_day = df["timestamp"].dt.normalize().max()
split_date = last_day - pd.Timedelta(days=TEST_DAYS - 1)
rf_train_start = split_date - pd.Timedelta(days=RF_TRAIN_DAYS)
first_test_day_end = split_date + pd.Timedelta(hours=23)
horizon_hours = pd.date_range(split_date, periods=TEST_DAYS * 24, freq="h")

# ---------- 3. Weather for the FORECAST WEEK: real if available ----------
horizon_weather = get_horizon_weather(horizon_hours).set_index("timestamp")
# Overwrite df's weather for the forecast-week rows with this real/best data,
# so anything reading df for this period (and the lag features) sees it too.
for ts, row in horizon_weather.iterrows():
    mask = df["timestamp"] == ts
    df.loc[mask, "temperature_c"] = row["temperature_c"]
    df.loc[mask, "humidity_pct"] = row["humidity_pct"]

type_dummies = pd.get_dummies(df["building_type"], prefix="type")
df = pd.concat([df, type_dummies], axis=1)
type_cols = list(type_dummies.columns)
feature_cols = (["hour", "day_of_week", "is_weekend", "month", "floor_area_m2",
                "temperature_c", "humidity_pct", "lag_24", "lag_168"] + type_cols)

# ---------- 4. Train Random Forest ----------
train_rf = df[(df.timestamp < split_date) & (df.timestamp >= rf_train_start)] \
    .dropna(subset=["lag_24", "lag_168"])
rf = RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=2,
                           random_state=42, n_jobs=-1)
rf.fit(train_rf[feature_cols], train_rf["consumption_kwh"])

# ---------- 5. Train LSTM (now with humidity too, not just temperature) ----------
import tensorflow as tf
from tensorflow import keras

b_mean = train_rf.groupby("building_id")["consumption_kwh"].mean().rename("b_mean")
df = df.join(b_mean, on="building_id")
df["b_mean"] = df["b_mean"].fillna(df["b_mean"].mean())
df["scaled"] = df["consumption_kwh"] / df["b_mean"]
df["hour_sin"] = np.sin(df["hour"] / 24 * 2 * np.pi)
df["hour_cos"] = np.cos(df["hour"] / 24 * 2 * np.pi)
df["temperature_scaled"] = (df["temperature_c"] - df["temperature_c"].mean()) / df["temperature_c"].std()
df["humidity_scaled"] = (df["humidity_pct"] - df["humidity_pct"].mean()) / df["humidity_pct"].std()
feature_channels = ["scaled", "hour_sin", "hour_cos", "is_weekend",
                    "temperature_scaled", "humidity_scaled"]

X, y = [], []
for bid in BUILDING_IDS:
    bdf = df[df.building_id == bid].sort_values("timestamp")
    bdf_train = bdf[bdf.timestamp < split_date]
    vals = bdf_train[feature_channels].values
    target = bdf_train["scaled"].values
    for i in range(LSTM_WINDOW, len(vals)):
        X.append(vals[i - LSTM_WINDOW:i])
        y.append(target[i])
X, y = np.array(X), np.array(y)

lstm = keras.Sequential([
    keras.layers.Input(shape=(LSTM_WINDOW, len(feature_channels))),
    keras.layers.LSTM(32),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(X, y, epochs=5, batch_size=256, verbose=2)

# ---------- 6. Recursive, no-leakage forecast loop ----------
history_actual = df.set_index(["building_id", "timestamp"])["consumption_kwh"]
rf_known = {bid: history_actual.loc[bid].copy() for bid in BUILDING_IDS}
lstm_known_scaled = {bid: df[df.building_id == bid].set_index("timestamp")["scaled"].copy()
                    for bid in BUILDING_IDS}
b_mean_map = df.drop_duplicates("building_id").set_index("building_id")["b_mean"]
static_lookup = meta.set_index("building_id")
weather_lookup = df.drop_duplicates("timestamp").set_index("timestamp")[
    ["temperature_c", "humidity_pct", "hour_sin", "hour_cos", "temperature_scaled", "humidity_scaled"]]

results = {bid: {"actual": [], "rf_pred": [], "lstm_pred": [],
                 "temperature_c": [], "humidity_pct": []} for bid in BUILDING_IDS}

for t in horizon_hours:
    w = weather_lookup.loc[t]
    for bid in BUILDING_IDS:
        b = static_lookup.loc[bid]
        lag24 = rf_known[bid].get(t - pd.Timedelta(hours=24), np.nan)
        lag168 = rf_known[bid].get(t - pd.Timedelta(hours=168), np.nan)
        row = {"hour": t.hour, "day_of_week": t.dayofweek, "is_weekend": int(t.dayofweek >= 5),
              "month": t.month, "floor_area_m2": b["floor_area_m2"],
              "temperature_c": w["temperature_c"], "humidity_pct": w["humidity_pct"],
              "lag_24": lag24, "lag_168": lag168}
        for c in type_cols:
            row[c] = 1.0 if c == f"type_{b['building_type']}" else 0.0
        rf_pred = rf.predict(pd.DataFrame([row])[feature_cols])[0]
        rf_known[bid].loc[t] = rf_pred

        series = lstm_known_scaled[bid]
        window_times = pd.date_range(t - pd.Timedelta(hours=LSTM_WINDOW), periods=LSTM_WINDOW, freq="h")
        scaled_vals = series.reindex(window_times).values
        hs = np.sin(window_times.hour.values / 24 * 2 * np.pi)
        hc = np.cos(window_times.hour.values / 24 * 2 * np.pi)
        iwk = (window_times.dayofweek.values >= 5).astype(float)
        temps = weather_lookup.reindex(window_times)["temperature_scaled"].values
        hums = weather_lookup.reindex(window_times)["humidity_scaled"].values
        lstm_input = np.stack([scaled_vals, hs, hc, iwk, temps, hums], axis=1)[None, ...]
        lstm_pred_scaled = lstm.predict(lstm_input, verbose=0).flatten()[0]
        lstm_known_scaled[bid].loc[t] = lstm_pred_scaled
        lstm_pred = lstm_pred_scaled * b_mean_map.loc[bid]

        actual_val = history_actual.get((bid, t), np.nan)
        results[bid]["actual"].append(round(float(actual_val), 2) if pd.notna(actual_val) else None)
        results[bid]["rf_pred"].append(round(float(rf_pred), 2))
        results[bid]["lstm_pred"].append(round(float(lstm_pred), 2))
        results[bid]["temperature_c"].append(round(float(w["temperature_c"]), 1))
        results[bid]["humidity_pct"].append(round(float(w["humidity_pct"]), 1))

# ---------- 7. Metrics (MAE, RMSE, MAPE, R2) - SEPARATELY for next-24h
# and next-7-days, as her point 2 literally asked for ("next 24 hours
# AND next 7 days"). This split was present in the original version and
# was accidentally lost when real weather was added - restored here. ----------
N_24H = 24


def metric_row(bid, model_name, horizon_name, actual_vals, pred_vals):
    a = np.array(actual_vals, dtype=float)
    p = np.array(pred_vals, dtype=float)
    mape = np.mean(np.abs((a - p) / np.maximum(a, 1e-6))) * 100
    return {
        "building_id": bid, "model": model_name, "horizon": horizon_name,
        "MAE_kWh": round(mean_absolute_error(a, p), 3),
        "RMSE_kWh": round(float(np.sqrt(mean_squared_error(a, p))), 3),
        "MAPE_pct": round(float(mape), 2),
        "R2": round(r2_score(a, p), 3),
    }


metric_rows = []
for bid in BUILDING_IDS:
    a_full = [v for v in results[bid]["actual"] if v is not None]
    n = len(a_full)
    rf_full = results[bid]["rf_pred"][:n]
    lstm_full = results[bid]["lstm_pred"][:n]

    for model_name, pred_full in [("Random Forest", rf_full), ("LSTM", lstm_full)]:
        # next-24h: the first 24 hours of the forecast horizon only
        metric_rows.append(metric_row(bid, model_name, "next_24h",
                                      a_full[:N_24H], pred_full[:N_24H]))
        # next-7-days: the full 168-hour horizon
        metric_rows.append(metric_row(bid, model_name, "next_7_days",
                                      a_full, pred_full))

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(DATA_DIR / "forecast_buildings_j9_metrics.csv", index=False)
print("\n=== Forecast metrics (J and 9) - next 24h and next 7 days, separately ===")
print(metrics_df.to_string(index=False))

# ---------- 8. Save into forecast_results.json ----------
results_path = DATA_DIR / "forecast_results.json"
if results_path.exists():
    with open(results_path) as f:
        existing = json.load(f)
else:
    existing = {"buildings": {}}
for bid in BUILDING_IDS:
    existing["buildings"][bid] = results[bid]
existing["forecast_start"] = str(split_date)
existing["first_24h_index"] = int((first_test_day_end - split_date).total_seconds() // 3600)
with open(results_path, "w") as f:
    json.dump(existing, f)

print(f"\nUpdated {results_path} for buildings: {BUILDING_IDS}")
print(f"Saved {DATA_DIR / 'forecast_buildings_j9_metrics.csv'}")
