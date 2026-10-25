"""
forecast_rooms.py
-------------------
Same genuine, no-leakage recursive forecasting approach as
forecast_short_term.py (RF vs LSTM, next-24h / next-7-days), but applied
to just the TWO selected rooms in Dormitory 9 - the normal baseline room
(9-101) and the room with the injected "AC left on" anomaly (9-105).
Kept as a separate, small script since only 2 rooms need this (not all 20),
so it runs in seconds rather than minutes.

Reads:  ../data/rooms_simulated.csv
Writes: ../data/forecast_rooms_results.json

Run from the project root:  python3 src/forecast_rooms.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ROOM_IDS = ["9-101", "9-105"]
TEST_DAYS = 7
LSTM_WINDOW = 24
RF_TRAIN_DAYS = 45

df = pd.read_csv(DATA_DIR / "rooms_simulated.csv",
                 dtype={"room_id": str}, parse_dates=["timestamp"])
df = df[df["room_id"].isin(ROOM_IDS)].sort_values(["room_id", "timestamp"]).reset_index(drop=True)

# ---------- Same real-Shanghai-climate weather as forecast_short_term.py ----------
MONTHLY_TEMP_C = {1: 5.0, 2: 5.6, 3: 9.4, 4: 15.0, 5: 20.6, 6: 23.9,
                  7: 28.3, 8: 27.8, 9: 24.4, 10: 19.4, 11: 13.3, 12: 7.8}
np.random.seed(11)
month = df["timestamp"].dt.month.values
hour = df["hour"].values
monthly_base_temp = np.array([MONTHLY_TEMP_C[m] for m in month])
daily_swing = 4 * np.sin((hour - 9) / 24 * 2 * np.pi)
df["temperature_c"] = (monthly_base_temp + daily_swing + np.random.normal(0, 1.2, len(df))).round(1)

g = df.groupby("room_id")["consumption_kwh"]
df["lag_24"] = g.shift(24)
df["lag_168"] = g.shift(168)

last_day = df["timestamp"].dt.normalize().max()
split_date = last_day - pd.Timedelta(days=TEST_DAYS - 1)
rf_train_start = split_date - pd.Timedelta(days=RF_TRAIN_DAYS)
first_test_day_end = split_date + pd.Timedelta(hours=23)
print(f"Forecast origin: {split_date - pd.Timedelta(hours=1)}")
print(f"Test horizon: {split_date.date()} to {last_day.date()}\n")

feature_cols = ["hour", "day_of_week", "is_weekend", "month", "temperature_c",
                "lag_24", "lag_168"]

print("Training Random Forest on rooms 9-101 and 9-105...")
train_rf = df[(df["timestamp"] < split_date) & (df["timestamp"] >= rf_train_start)] \
    .dropna(subset=["lag_24", "lag_168"])
rf = RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=2,
                           random_state=42, n_jobs=-1)
rf.fit(train_rf[feature_cols], train_rf["consumption_kwh"])

print("Training LSTM on rooms 9-101 and 9-105...")
import tensorflow as tf
from tensorflow import keras

b_mean = train_rf.groupby("room_id")["consumption_kwh"].mean().rename("b_mean")
df = df.join(b_mean, on="room_id")
df["b_mean"] = df["b_mean"].fillna(df["b_mean"].mean())
df["scaled"] = df["consumption_kwh"] / df["b_mean"]
df["hour_sin"] = np.sin(df["hour"] / 24 * 2 * np.pi)
df["hour_cos"] = np.cos(df["hour"] / 24 * 2 * np.pi)
df["temperature_scaled"] = (df["temperature_c"] - df["temperature_c"].mean()) / df["temperature_c"].std()
feature_channels = ["scaled", "hour_sin", "hour_cos", "is_weekend", "temperature_scaled"]

X, y = [], []
for rid in ROOM_IDS:
    rdf = df[df.room_id == rid].sort_values("timestamp")
    rdf_train = rdf[rdf["timestamp"] < split_date]
    vals = rdf_train[feature_channels].values
    target = rdf_train["scaled"].values
    for i in range(LSTM_WINDOW, len(vals)):
        X.append(vals[i - LSTM_WINDOW:i])
        y.append(target[i])
X, y = np.array(X), np.array(y)

lstm = keras.Sequential([
    keras.layers.Input(shape=(LSTM_WINDOW, len(feature_channels))),
    keras.layers.LSTM(24),
    keras.layers.Dense(12, activation="relu"),
    keras.layers.Dense(1),
])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(X, y, epochs=8, batch_size=128, verbose=2)

# ---------- Recursive, no-leakage forecast loop (just 2 rooms, so it's fast) ----------
print("\nRunning the recursive forecast for both rooms...")
horizon_hours = pd.date_range(split_date, periods=TEST_DAYS * 24, freq="h")
history_actual = df.set_index(["room_id", "timestamp"])["consumption_kwh"]
rf_known = {rid: history_actual.loc[rid].copy() for rid in ROOM_IDS}
lstm_known_scaled = {rid: df[df.room_id == rid].set_index("timestamp")["scaled"].copy()
                    for rid in ROOM_IDS}
b_mean_map = df.drop_duplicates("room_id").set_index("room_id")["b_mean"]
weather_lookup = df.drop_duplicates("timestamp").set_index("timestamp")[
    ["temperature_c", "hour_sin", "hour_cos", "temperature_scaled"]]

results = {rid: {"actual": [], "rf_pred": [], "lstm_pred": []} for rid in ROOM_IDS}

for t in horizon_hours:
    w = weather_lookup.loc[t]
    for rid in ROOM_IDS:
        lag24 = rf_known[rid].get(t - pd.Timedelta(hours=24), np.nan)
        lag168 = rf_known[rid].get(t - pd.Timedelta(hours=168), np.nan)
        row = pd.DataFrame([{
            "hour": t.hour, "day_of_week": t.dayofweek,
            "is_weekend": int(t.dayofweek >= 5), "month": t.month,
            "temperature_c": w["temperature_c"], "lag_24": lag24, "lag_168": lag168,
        }])[feature_cols]
        rf_pred = rf.predict(row)[0]
        rf_known[rid].loc[t] = rf_pred

        series = lstm_known_scaled[rid]
        window_times = pd.date_range(t - pd.Timedelta(hours=LSTM_WINDOW), periods=LSTM_WINDOW, freq="h")
        scaled_vals = series.reindex(window_times).values
        hs = np.sin(window_times.hour.values / 24 * 2 * np.pi)
        hc = np.cos(window_times.hour.values / 24 * 2 * np.pi)
        iwk = (window_times.dayofweek.values >= 5).astype(float)
        temps = weather_lookup.reindex(window_times)["temperature_scaled"].values
        lstm_input = np.stack([scaled_vals, hs, hc, iwk, temps], axis=1)[None, ...]
        lstm_pred_scaled = lstm.predict(lstm_input, verbose=0).flatten()[0]
        lstm_known_scaled[rid].loc[t] = lstm_pred_scaled
        lstm_pred = lstm_pred_scaled * b_mean_map.loc[rid]

        actual_val = history_actual.get((rid, t), np.nan)
        results[rid]["actual"].append(round(float(actual_val), 2) if pd.notna(actual_val) else None)
        results[rid]["rf_pred"].append(round(float(rf_pred), 2))
        results[rid]["lstm_pred"].append(round(float(lstm_pred), 2))

import json
output = {
    "forecast_start": str(split_date),
    "first_24h_index": int((first_test_day_end - split_date).total_seconds() // 3600),
    "rooms": results,
}
with open(DATA_DIR / "forecast_rooms_results.json", "w") as f:
    json.dump(output, f)
print(f"\nSaved {DATA_DIR / 'forecast_rooms_results.json'}")
for rid in ROOM_IDS:
    a = np.array([v for v in results[rid]["actual"] if v is not None])
    p = np.array(results[rid]["rf_pred"][:len(a)])
    mae = np.mean(np.abs(a - p))
    print(f"  {rid}: RF MAE = {mae:.3f} kWh")
