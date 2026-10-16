"""
simulate_data.py
-----------------
Creates fake (but realistic-looking) hourly electricity consumption
data for every building on campus, since real meter data isn't
available yet. Each building type has its own daily pattern.

Folder structure assumed:
  campus-energy-project/
    data/   <- reads building_metadata.csv from here, writes results here
    src/    <- this script lives here

Output: ../data/campus_energy_simulated.csv
Columns: building_id, timestamp, hour, day_of_week, is_weekend, consumption_kwh
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)  # so results are the same every time we run this

# This makes the paths work no matter where you run the script FROM,
# as long as it stays inside src/ and data/ is its sibling folder.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

# 1. Load the building list we made in Step 1
buildings = pd.read_csv(DATA_DIR / "building_metadata.csv")

# 2. Decide how many days of fake data to generate
DAYS = 14
timestamps = pd.date_range(start="2026-09-01", periods=DAYS * 24, freq="h")

# 3. Define a typical 24-hour usage "shape" for each building type.
#    These are just relative weights (0 = no use, 1 = peak use).
#    Index 0 = midnight, index 23 = 11pm.
patterns = {
    "academic": [0.1,0.1,0.1,0.1,0.1,0.15,0.3,0.6,0.9,1.0,1.0,0.95,
                 0.7,0.9,1.0,0.95,0.8,0.5,0.3,0.2,0.15,0.1,0.1,0.1],
    "dorm":     [0.5,0.4,0.3,0.3,0.3,0.35,0.5,0.6,0.4,0.3,0.3,0.35,
                 0.4,0.35,0.3,0.35,0.5,0.7,0.9,1.0,1.0,0.9,0.7,0.6],
    "sports":   [0.05,0.05,0.05,0.05,0.05,0.1,0.2,0.3,0.4,0.4,0.4,0.4,
                 0.5,0.4,0.4,0.5,0.7,1.0,1.0,0.9,0.6,0.3,0.1,0.05],
    "cafeteria":[0.1,0.1,0.1,0.1,0.1,0.2,0.6,1.0,0.6,0.3,0.3,0.9,
                 1.0,0.6,0.3,0.3,0.5,1.0,1.0,0.6,0.3,0.2,0.1,0.1],
    "library":  [0.2,0.2,0.2,0.2,0.2,0.2,0.3,0.6,0.9,1.0,1.0,1.0,
                 0.8,0.9,1.0,1.0,0.9,0.8,0.9,1.0,0.9,0.6,0.4,0.3],
    "admin":    [0.1,0.1,0.1,0.1,0.1,0.15,0.3,0.6,0.9,1.0,1.0,0.9,
                 0.6,0.9,1.0,0.9,0.6,0.3,0.15,0.1,0.1,0.1,0.1,0.1],
}

rows = []
for _, b in buildings.iterrows():
    base_load = b["floor_area_m2"] / 100  # bigger buildings use more power
    shape = patterns.get(b["building_type"], patterns["academic"])

    for ts in timestamps:
        hour = ts.hour
        is_weekend = ts.dayofweek >= 5

        weekday_factor = 0.6 if (is_weekend and b["building_type"] != "dorm") else 1.0
        noise = np.random.normal(1.0, 0.08)  # +/- 8% random variation

        consumption = base_load * shape[hour] * weekday_factor * noise
        consumption = max(consumption, 0)  # never negative

        rows.append({
            "building_id": b["building_id"],
            "timestamp": ts,
            "hour": hour,
            "day_of_week": ts.dayofweek,
            "is_weekend": int(is_weekend),
            "consumption_kwh": round(consumption, 2),
        })

df = pd.DataFrame(rows)
output_path = DATA_DIR / "campus_energy_simulated.csv"
df.to_csv(output_path, index=False)
print(f"Created {output_path} with {len(df)} rows "
      f"for {buildings.shape[0]} buildings over {DAYS} days.")
print(df.head(10))
