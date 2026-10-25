"""
simulate_data.py
-----------------
Creates fake (but realistic-looking) hourly electricity consumption data
for every building on campus, for a FULL YEAR, since real meter data
isn't available yet.

Each building type has its own daily usage shape (from before), and now
ALSO a seasonal factor: higher consumption in winter (heating) and summer
(cooling), lower in spring/fall - the same pattern your reference paper
found for real campus data (their Fig. 6 and Fig. 7).

Output: ../data/campus_energy_simulated.csv
Columns: building_id, timestamp, hour, day_of_week, is_weekend, month,
         season, consumption_kwh
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

buildings = pd.read_csv(DATA_DIR / "building_metadata.csv")

# A full year of hourly timestamps
YEAR = 2026
timestamps = pd.date_range(start=f"{YEAR}-01-01", end=f"{YEAR}-12-31 23:00", freq="h")

# ---------- Daily usage "shape" per building type (same as before) ----------
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


def season_of(month):
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


# Seasonal multiplier: higher in winter (heating) and summer (cooling),
# lower in spring/fall - a "U-shape" over the year, like the paper found.
# day_of_year runs 1-365; this gives two bumps (winter + summer) and two dips.
day_of_year = timestamps.dayofyear.values
seasonal_factor = 1.0 + 0.35 * np.cos((day_of_year - 15) / 365 * 4 * np.pi)
# ^ cos with period 365/2 creates two peaks (around day 15 = mid-Jan, and
#   around day 197 = mid-July) and two troughs (around early April, early Oct)

rows = []
for _, b in buildings.iterrows():
    base_load = b["floor_area_m2"] / 100
    shape = patterns.get(b["building_type"], patterns["academic"])
    shape_arr = np.array(shape)[timestamps.hour]

    # Office/teaching buildings nearly empty on weekends (no classes).
    # Dorms ALSO drop on weekends - a meaningful share of students go home
    # Friday evening and return Sunday, so occupancy (and electricity use)
    # is lower, not identical to weekdays. 0.72 is a stated assumption
    # (roughly a quarter of occupancy away), not a measured real figure.
    is_weekend_ts = timestamps.dayofweek >= 5
    weekday_factor = np.where(is_weekend_ts & (b["building_type"] != "dorm"), 0.6, 1.0)
    weekday_factor = np.where(is_weekend_ts & (b["building_type"] == "dorm"), 0.72, weekday_factor)
    # Dorms are much emptier during the winter and summer academic breaks
    is_break = ((timestamps.month == 1) | (timestamps.month == 7) | (timestamps.month == 8))
    break_factor = np.where(is_break & (b["building_type"] == "dorm"), 0.4, 1.0)
    break_factor = np.where(is_break & (b["building_type"] != "dorm"), 0.3, break_factor)

    noise = np.random.normal(1.0, 0.08, len(timestamps))

    consumption = (base_load * shape_arr * weekday_factor
                  * seasonal_factor * break_factor * noise)
    consumption = np.maximum(consumption, 0)

    rows.append(pd.DataFrame({
        "building_id": b["building_id"],
        "timestamp": timestamps,
        "hour": timestamps.hour,
        "day_of_week": timestamps.dayofweek,
        "is_weekend": (timestamps.dayofweek >= 5).astype(int),
        "month": timestamps.month,
        "season": [season_of(m) for m in timestamps.month],
        "is_break": is_break.astype(int),
        "consumption_kwh": np.round(consumption, 2),
    }))

df = pd.concat(rows, ignore_index=True)

# ---------- Inject a few realistic anomalies, for testing the anomaly
# detector later. These are clearly labeled (is_injected_anomaly column)
# so they can be identified and excluded from any accuracy evaluation -
# they exist ONLY to prove the anomaly detector actually catches problems,
# not as part of the "normal" data the forecasting models learn from. ----
df["is_injected_anomaly"] = 0
rng2 = np.random.RandomState(99)


def inject(building_id, start, end, multiplier, note):
    mask = (df["building_id"] == building_id) & \
          (df["timestamp"] >= start) & (df["timestamp"] <= end)
    df.loc[mask, "consumption_kwh"] = (df.loc[mask, "consumption_kwh"] * multiplier).round(2)
    df.loc[mask, "is_injected_anomaly"] = 1
    print(f"  Injected: {note}")


print("Injecting test anomalies (for the anomaly-detector demo):")
# "AC left on during lunch break" - a teaching building using 2x normal
# during a time it should be quiet
inject("H", f"{YEAR}-11-16 12:00", f"{YEAR}-11-16 14:00", 2.4,
      "Building H, Nov 16, 12-14h: AC left on during lunch (2.4x normal)")
# A dormitory with an unusually high few days (e.g. a faulty appliance)
inject("3", f"{YEAR}-11-20 00:00", f"{YEAR}-11-23 23:00", 1.8,
      "Dormitory 3, Nov 20-23: sustained high usage, +80% (e.g. faulty appliance)")
# A classroom/office running overnight when it should be near-zero
inject("J", f"{YEAR}-12-05 01:00", f"{YEAR}-12-05 05:00", 6.0,
      "Building J, Dec 5, 01-05h: equipment left running overnight")
# A short, very sharp spike (possible meter glitch or equipment fault)
inject("17", f"{YEAR}-12-10 18:00", f"{YEAR}-12-10 18:00", 3.5,
      "Dormitory 17, Dec 10, 18h: brief sharp spike")
# Sustained high usage during the LAST week of data, so the monthly-bill
# projection alert also has a demonstrable example
inject("4", f"{YEAR}-12-26 00:00", f"{YEAR}-12-31 23:00", 1.7,
      "Dormitory 4, Dec 26-31: sustained high usage (+70%), for the monthly-bill alert demo")

df.to_csv(DATA_DIR / "campus_energy_simulated.csv", index=False)
print(f"Created campus_energy_simulated.csv with {len(df):,} rows "
     f"for {buildings.shape[0]} buildings over the full year {YEAR}.")
print(df.head(3))
print("\nSeasonal average consumption (sanity check - winter/summer should be higher):")
print(df.groupby("season")["consumption_kwh"].mean().round(2))
