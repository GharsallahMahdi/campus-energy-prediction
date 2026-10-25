"""
simulate_rooms.py
-------------------
Generates realistic hourly electricity data for individual ROOMS inside
one chosen dormitory building, since the supervisor asked to be able to
select individual rooms within the dormitory, not just the whole building.

Each room gets the dormitory's overall daily/seasonal pattern (same shape
as simulate_data.py uses for the building), scaled to a room's share of
the building, with its own independent random variation - so rooms look
similar but not identical, like real students would behave differently.

Reads:  ../data/building_metadata.csv (for the chosen dorm's floor area)
Writes: ../data/rooms_simulated.csv

Run from the project root:  python3 src/simulate_rooms.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(123)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

DORM_BUILDING_ID = "9"      # Dormitory 9 - chosen for its strong forecast fit
ROOMS_PER_BUILDING = 20      # e.g. 4 floors x 5 rooms, a believable dorm size

buildings = pd.read_csv(DATA_DIR / "building_metadata.csv", dtype={"building_id": str})
dorm = buildings[buildings.building_id == DORM_BUILDING_ID].iloc[0]

YEAR = 2026
timestamps = pd.date_range(start=f"{YEAR}-01-01", end=f"{YEAR}-12-31 23:00", freq="h")

# Same dorm daily shape as simulate_data.py uses
dorm_pattern = np.array([0.5,0.4,0.3,0.3,0.3,0.35,0.5,0.6,0.4,0.3,0.3,0.35,
                         0.4,0.35,0.3,0.35,0.5,0.7,0.9,1.0,1.0,0.9,0.7,0.6])


def season_of(month):
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


day_of_year = timestamps.dayofyear.values
seasonal_factor = 1.0 + 0.35 * np.cos((day_of_year - 15) / 365 * 4 * np.pi)
is_break = ((timestamps.month == 1) | (timestamps.month == 7) | (timestamps.month == 8))
break_factor = np.where(is_break, 0.4, 1.0)
# Same weekend effect as simulate_data.py: some students go home on
# weekends, so room consumption is lower, not identical to weekdays.
weekend_factor = np.where(timestamps.dayofweek >= 5, 0.72, 1.0)

room_base_load = (dorm["floor_area_m2"] / ROOMS_PER_BUILDING) / 100  # each room's share

rows = []
for room_num in range(1, ROOMS_PER_BUILDING + 1):
    room_id = f"{DORM_BUILDING_ID}-{100 + room_num}"   # e.g. "9-101", "9-102", ...
    shape = dorm_pattern[timestamps.hour]

    # Each room has its own baseline "occupancy intensity" (some students use
    # more electricity than others) and its own random noise per hour.
    room_intensity = np.random.normal(1.0, 0.18)
    noise = np.random.normal(1.0, 0.10, len(timestamps))

    consumption = room_base_load * shape * seasonal_factor * break_factor * weekend_factor * room_intensity * noise
    consumption = np.maximum(consumption, 0)

    rows.append(pd.DataFrame({
        "room_id": room_id,
        "building_id": DORM_BUILDING_ID,
        "room_number": 100 + room_num,
        "timestamp": timestamps,
        "hour": timestamps.hour,
        "day_of_week": timestamps.dayofweek,
        "is_weekend": (timestamps.dayofweek >= 5).astype(int),
        "month": timestamps.month,
        "season": [season_of(m) for m in timestamps.month],
        "is_break": is_break.astype(int),
        "consumption_kwh": np.round(consumption, 3),
    }))

df = pd.concat(rows, ignore_index=True)

# ---------- Inject one realistic anomaly: a room where the AC was left on
# for a few days, near the end of the data (so it shows up in the last-30-
# days view AND inside the 7-day forecast test window). Room 9-101 is kept
# clean as a normal comparison baseline, so the two rooms show a genuine,
# demonstrable usage difference - exactly the scenario the supervisor
# described: "students forgot to close the air conditioner." ----
df["is_injected_anomaly"] = 0
ANOMALY_ROOM = f"{DORM_BUILDING_ID}-105"
mask = (df["room_id"] == ANOMALY_ROOM) & \
      (df["timestamp"] >= f"{YEAR}-12-27 00:00") & \
      (df["timestamp"] <= f"{YEAR}-12-29 23:00")
df.loc[mask, "consumption_kwh"] = (df.loc[mask, "consumption_kwh"] * 2.3).round(3)
df.loc[mask, "is_injected_anomaly"] = 1
print(f"Injected anomaly: Room {ANOMALY_ROOM}, Dec 27-29, +130% "
     f"(AC left on) - {mask.sum()} hours affected")
print(f"Room 9-101 kept clean as the normal comparison baseline")

df.to_csv(DATA_DIR / "rooms_simulated.csv", index=False)
print(f"Created rooms_simulated.csv with {len(df):,} rows "
     f"for {ROOMS_PER_BUILDING} rooms in Dormitory {DORM_BUILDING_ID}.")
print(df.head(3))
print(f"\nSanity check - average room consumption vs building/{ROOMS_PER_BUILDING}:")
print(f"  Avg per room: {df['consumption_kwh'].mean():.3f} kWh/hour")
