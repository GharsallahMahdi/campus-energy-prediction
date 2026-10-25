"""
forecast_buildings_j9.py
--------------------------
A fast, SCOPED version of forecast_short_term.py's recursive no-leakage
forecast (RF + LSTM), restricted to just Computer Building (J) and
Dormitory 9 - the two buildings used on the focused campus map. Running
the full 37-building version takes several minutes; this takes under a
minute, since only 2 buildings need updating after a data change (e.g.
after the weekend-dorm fix).

Updates (merges into, not overwrites) forecast_results.json, so the other
35 buildings' forecasts - if you ever use the full campus-wide dashboard -
are left untouched.

Reads:  ../data/campus_energy_simulated.csv, ../data/building_metadata.csv,
       ../data/forecast_results.json (merged into)
Writes: ../data/forecast_results.json  (J and 9 entries updated)

Run from the project root:  python3 src/forecast_buildings_j9.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import json

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BUILDING_IDS = ["J", "9"]
TEST_DAYS = 7
LSTM_WINDOW = 24
RF_TRAIN_DAYS = 45

df = pd.read_csv(DATA_DIR / "campus_energy_simulated.csv", dtype={"building_id": str}, parse_dates=["timestamp"])
meta = pd.read_csv(DATA_DIR / "building_metadata.csv", dtype={"building_id": str})
df = df[df.building_id.isin(BUILDING_IDS)].merge(meta[["building_id","building_type","floor_area_m2"]], on="building_id").sort_values(["building_id","timestamp"]).reset_index(drop=True)

MONTHLY_TEMP_C = {1:5.0,2:5.6,3:9.4,4:15.0,5:20.6,6:23.9,7:28.3,8:27.8,9:24.4,10:19.4,11:13.3,12:7.8}
np.random.seed(7)
month = df["timestamp"].dt.month.values; hour = df["hour"].values
base = np.array([MONTHLY_TEMP_C[m] for m in month])
daily_swing = 4*np.sin((hour-9)/24*2*np.pi)
df["temperature_c"] = (base + daily_swing + np.random.normal(0,1.2,len(df))).round(1)

g = df.groupby("building_id")["consumption_kwh"]
df["lag_24"] = g.shift(24); df["lag_168"] = g.shift(168)

last_day = df["timestamp"].dt.normalize().max()
split_date = last_day - pd.Timedelta(days=TEST_DAYS-1)
rf_train_start = split_date - pd.Timedelta(days=RF_TRAIN_DAYS)
first_test_day_end = split_date + pd.Timedelta(hours=23)

type_dummies = pd.get_dummies(df["building_type"], prefix="type")
df = pd.concat([df, type_dummies], axis=1)
type_cols = list(type_dummies.columns)
feature_cols = ["hour","day_of_week","is_weekend","month","floor_area_m2","temperature_c","lag_24","lag_168"] + type_cols

train_rf = df[(df.timestamp < split_date) & (df.timestamp >= rf_train_start)].dropna(subset=["lag_24","lag_168"])
rf = RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(train_rf[feature_cols], train_rf["consumption_kwh"])

import tensorflow as tf
from tensorflow import keras
b_mean = train_rf.groupby("building_id")["consumption_kwh"].mean().rename("b_mean")
df = df.join(b_mean, on="building_id"); df["b_mean"] = df["b_mean"].fillna(df["b_mean"].mean())
df["scaled"] = df["consumption_kwh"]/df["b_mean"]
df["hour_sin"]=np.sin(df["hour"]/24*2*np.pi); df["hour_cos"]=np.cos(df["hour"]/24*2*np.pi)
df["temperature_scaled"]=(df["temperature_c"]-df["temperature_c"].mean())/df["temperature_c"].std()
feature_channels = ["scaled","hour_sin","hour_cos","is_weekend","temperature_scaled"]

X,y=[],[]
for bid in BUILDING_IDS:
    bdf = df[df.building_id==bid].sort_values("timestamp")
    bdf_train = bdf[bdf.timestamp < split_date]
    vals = bdf_train[feature_channels].values; target = bdf_train["scaled"].values
    for i in range(LSTM_WINDOW, len(vals)):
        X.append(vals[i-LSTM_WINDOW:i]); y.append(target[i])
X,y = np.array(X), np.array(y)
lstm = keras.Sequential([keras.layers.Input(shape=(LSTM_WINDOW,len(feature_channels))), keras.layers.LSTM(32), keras.layers.Dense(16,activation="relu"), keras.layers.Dense(1)])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(X,y, epochs=5, batch_size=256, verbose=2)

horizon_hours = pd.date_range(split_date, periods=TEST_DAYS*24, freq="h")
history_actual = df.set_index(["building_id","timestamp"])["consumption_kwh"]
rf_known = {bid: history_actual.loc[bid].copy() for bid in BUILDING_IDS}
lstm_known_scaled = {bid: df[df.building_id==bid].set_index("timestamp")["scaled"].copy() for bid in BUILDING_IDS}
b_mean_map = df.drop_duplicates("building_id").set_index("building_id")["b_mean"]
static_lookup = meta.set_index("building_id")
weather_lookup = df.drop_duplicates("timestamp").set_index("timestamp")[["temperature_c","hour_sin","hour_cos","temperature_scaled"]]

results = {bid: {"actual":[], "rf_pred":[], "lstm_pred":[]} for bid in BUILDING_IDS}
for t in horizon_hours:
    w = weather_lookup.loc[t]
    for bid in BUILDING_IDS:
        b = static_lookup.loc[bid]
        lag24 = rf_known[bid].get(t-pd.Timedelta(hours=24), np.nan)
        lag168 = rf_known[bid].get(t-pd.Timedelta(hours=168), np.nan)
        row = {"hour":t.hour,"day_of_week":t.dayofweek,"is_weekend":int(t.dayofweek>=5),"month":t.month,
               "floor_area_m2":b["floor_area_m2"],"temperature_c":w["temperature_c"],"lag_24":lag24,"lag_168":lag168}
        for c in type_cols: row[c] = 1.0 if c==f"type_{b['building_type']}" else 0.0
        rf_pred = rf.predict(pd.DataFrame([row])[feature_cols])[0]
        rf_known[bid].loc[t] = rf_pred

        series = lstm_known_scaled[bid]
        window_times = pd.date_range(t-pd.Timedelta(hours=LSTM_WINDOW), periods=LSTM_WINDOW, freq="h")
        scaled_vals = series.reindex(window_times).values
        hs=np.sin(window_times.hour.values/24*2*np.pi); hc=np.cos(window_times.hour.values/24*2*np.pi)
        iwk=(window_times.dayofweek.values>=5).astype(float)
        temps = weather_lookup.reindex(window_times)["temperature_scaled"].values
        lstm_input = np.stack([scaled_vals,hs,hc,iwk,temps],axis=1)[None,...]
        lstm_pred_scaled = lstm.predict(lstm_input, verbose=0).flatten()[0]
        lstm_known_scaled[bid].loc[t] = lstm_pred_scaled
        lstm_pred = lstm_pred_scaled * b_mean_map.loc[bid]

        actual_val = history_actual.get((bid,t), np.nan)
        results[bid]["actual"].append(round(float(actual_val),2) if pd.notna(actual_val) else None)
        results[bid]["rf_pred"].append(round(float(rf_pred),2))
        results[bid]["lstm_pred"].append(round(float(lstm_pred),2))

results_path = DATA_DIR / "forecast_results.json"
if results_path.exists():
    with open(results_path) as f:
        existing = json.load(f)
else:
    # No full 37-building run yet - that's fine, start fresh with just
    # these two buildings (all the focused dashboard needs).
    existing = {"buildings": {}}
for bid in BUILDING_IDS:
    existing["buildings"][bid] = results[bid]
existing["forecast_start"] = str(split_date)
existing["first_24h_index"] = int((first_test_day_end - split_date).total_seconds()//3600)
with open(DATA_DIR / "forecast_results.json", "w") as f:
    json.dump(existing, f)
print("Updated forecast_results.json for buildings:", BUILDING_IDS)
for bid in BUILDING_IDS:
    a = np.array([v for v in results[bid]["actual"] if v is not None])
    p = np.array(results[bid]["rf_pred"][:len(a)])
    print(f"  {bid}: RF MAE = {np.mean(np.abs(a-p)):.3f} kWh")
