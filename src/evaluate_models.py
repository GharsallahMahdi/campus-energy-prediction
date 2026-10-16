"""
evaluate_models.py
------------------
Compares simple baselines with Random Forest models, using a TIME-BASED split:
the model is trained on the first days and tested on the LAST days, which is
how real forecasting works (you never see the future when you train).

Models compared:
  1. Mean baseline          - predicts each building's average consumption
  2. Hourly-profile baseline - predicts each building's average for that hour
                               and day type (weekday/weekend)
  3. Yesterday baseline     - predicts "same hour yesterday"
  4. Random Forest          - time + building features
  5. Random Forest + lag    - same, plus yesterday's consumption as a feature

Reads:  ../data/campus_energy_simulated.csv, ../data/building_metadata.csv
Writes: ../data/model_comparison.csv, ../data/prediction_plot.png

Run from the project root:  python3 src/evaluate_models.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save the plot to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------- 1. Load and prepare ----------
data = pd.read_csv(DATA_DIR / "campus_energy_simulated.csv",
                   dtype={"building_id": str}, parse_dates=["timestamp"])
meta = pd.read_csv(DATA_DIR / "building_metadata.csv", dtype={"building_id": str})

df = data.merge(meta[["building_id", "building_type", "floor_area_m2"]],
                on="building_id", how="left")
df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

# Lag feature: consumption of the same building exactly 24 hours earlier
df["lag_24"] = df.groupby("building_id")["consumption_kwh"].shift(24)

# ---------- 2. Time-based split ----------
last_day = df["timestamp"].dt.normalize().max()
TEST_DAYS = 4
split_date = last_day - pd.Timedelta(days=TEST_DAYS - 1)

train = df[df["timestamp"] < split_date].copy()
test = df[df["timestamp"] >= split_date].copy()
# The first day has no "yesterday", so lag rows there are empty
train_lag = train.dropna(subset=["lag_24"])

print(f"Training: {train['timestamp'].min().date()} to "
      f"{(split_date - pd.Timedelta(days=1)).date()}  ({len(train)} rows)")
print(f"Testing:  {split_date.date()} to {last_day.date()}  ({len(test)} rows)\n")

y_test = test["consumption_kwh"]
results = {}
predictions = {}


def score(name, y_pred):
    y_pred = np.asarray(y_pred)
    results[name] = {
        "MAE (kWh)": mean_absolute_error(y_test, y_pred),
        "RMSE (kWh)": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAPE (%)": np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-6))) * 100,
        "R2": r2_score(y_test, y_pred),
    }
    predictions[name] = y_pred


# ---------- 3. Baselines ----------
# 3a. Mean baseline
mean_map = train.groupby("building_id")["consumption_kwh"].mean()
score("Mean baseline", test["building_id"].map(mean_map))

# 3b. Hourly-profile baseline
profile = (train.groupby(["building_id", "is_weekend", "hour"])["consumption_kwh"]
           .mean().rename("profile"))
prof_pred = test.join(profile, on=["building_id", "is_weekend", "hour"])["profile"]
score("Hourly-profile baseline", prof_pred)

# 3c. Yesterday baseline (same hour yesterday)
score("Yesterday baseline", test["lag_24"])

# ---------- 4. Random Forest models ----------
type_dummies = pd.get_dummies(df["building_type"], prefix="type")
base_features = ["hour", "day_of_week", "is_weekend", "floor_area_m2"]
X_all = pd.concat([df[base_features], type_dummies], axis=1).astype(float)
X_all["lag_24"] = df["lag_24"]

train_idx, test_idx = train.index, test.index
lag_train_idx = train_lag.index
feat_no_lag = list(X_all.columns.drop("lag_24"))
feat_lag = list(X_all.columns)

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_all.loc[train_idx, feat_no_lag], df.loc[train_idx, "consumption_kwh"])
score("Random Forest", rf.predict(X_all.loc[test_idx, feat_no_lag]))

rf_lag = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_lag.fit(X_all.loc[lag_train_idx, feat_lag], df.loc[lag_train_idx, "consumption_kwh"])
score("Random Forest + lag_24", rf_lag.predict(X_all.loc[test_idx, feat_lag]))

# ---------- 5. Results table ----------
table = pd.DataFrame(results).T.round(3)
print("=== Model comparison on the test period ===")
print(table.to_string())
table.to_csv(DATA_DIR / "model_comparison.csv")

best = table["MAE (kWh)"].idxmin()
print(f"\nLowest MAE: {best}")

# ---------- 6. Feature importance ----------
imp = pd.Series(rf_lag.feature_importances_, index=feat_lag).sort_values(ascending=False)
print("\n=== Feature importance (Random Forest + lag_24) ===")
print(imp.head(6).round(3).to_string())

# ---------- 7. Plot one building ----------
building = "A"
mask = (test["building_id"] == building).to_numpy()
t = test.loc[mask, "timestamp"]
plt.figure(figsize=(11, 4.5))
plt.plot(t, y_test[mask], label="Actual (simulated)", color="black", linewidth=1.8)
plt.plot(t, predictions["Hourly-profile baseline"][mask], label="Hourly-profile baseline",
         linestyle="--", color="#e3b23c")
plt.plot(t, predictions["Random Forest + lag_24"][mask], label="Random Forest + lag_24",
         color="#2f7d5f")
plt.title(f"Building {building}: actual vs predicted (test period)")
plt.ylabel("Consumption (kWh)")
plt.legend()
plt.tight_layout()
plt.savefig(DATA_DIR / "prediction_plot.png", dpi=150)
print(f"\nSaved {DATA_DIR / 'model_comparison.csv'} and {DATA_DIR / 'prediction_plot.png'}")
