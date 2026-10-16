"""
train_model.py
----------------
Trains a simple Random Forest model to predict a building's hourly
electricity consumption, using:
  - hour of day
  - day of week
  - is_weekend
  - building type
  - building floor area

Folder structure assumed:
  campus-energy-project/
    data/   <- reads and writes CSVs here
    src/    <- this script lives here

This mirrors what the reference GitHub project and the research paper
both do, just scaled down for a class project.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

# 1. Load our simulated data + building info
data = pd.read_csv(DATA_DIR / "campus_energy_simulated.csv")
buildings = pd.read_csv(DATA_DIR / "building_metadata.csv")

df = data.merge(buildings[["building_id", "building_type", "floor_area_m2"]],
                 on="building_id", how="left")

# 2. Turn building_type (text) into numbers the model can use
df = pd.get_dummies(df, columns=["building_type"], prefix="type")

# 3. Choose our input features (X) and the thing we want to predict (y)
feature_cols = ["hour", "day_of_week", "is_weekend", "floor_area_m2"] + \
               [c for c in df.columns if c.startswith("type_")]
X = df[feature_cols]
y = df["consumption_kwh"]

# 4. Split into training data (80%) and testing data (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train the model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate: how good are the predictions on data it hasn't seen?
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model performance on held-out test data:")
print(f"  MAE  = {mae:.2f} kWh")
print(f"  RMSE = {rmse:.2f} kWh")
print(f"  R^2  = {r2:.3f}")

# 7. Save predictions for every row (useful later for the dashboard)
df["predicted_kwh"] = model.predict(X)
output_path = DATA_DIR / "campus_energy_with_predictions.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved {output_path}")
