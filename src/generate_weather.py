"""
generate_weather.py
---------------------
Generates ONE shared hourly temperature series for the 7-day forecast
test week, grounded in real published Shanghai climate normals (same
source as the other scripts: climate-zone.com monthly averages).

This is deliberately a SEPARATE, single source of weather - used by the
focused dashboard's forecast charts (temperature overlay) - rather than
letting each script resample its own random weather noise, so every
building and room shows the exact same real-world weather for the same
hour (one campus has one weather station, not one per building).

Writes: ../data/weather_test_week.json

Run from the project root:  python3 src/generate_weather.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Real published Shanghai monthly average temperatures, Celsius
# (climate-zone.com, 8-year historical average)
MONTHLY_TEMP_C = {1: 5.0, 2: 5.6, 3: 9.4, 4: 15.0, 5: 20.6, 6: 23.9,
                  7: 28.3, 8: 27.8, 9: 24.4, 10: 19.4, 11: 13.3, 12: 7.8}

# Must match the test week used by forecast_short_term.py / forecast_rooms.py
# / forecast_buildings_j9.py: the last 7 days of the simulated year.
horizon = pd.date_range("2026-12-25", periods=7 * 24, freq="h")

np.random.seed(42)
month = horizon.month.values
hour = horizon.hour.values
base = np.array([MONTHLY_TEMP_C[m] for m in month])
daily_swing = 4 * np.sin((hour - 9) / 24 * 2 * np.pi)   # peak mid-afternoon
temperature_c = (base + daily_swing + np.random.normal(0, 1.0, len(horizon))).round(1)

import json
output = {
    "timestamps": [str(t) for t in horizon],
    "temperature_c": temperature_c.tolist(),
}
with open(DATA_DIR / "weather_test_week.json", "w") as f:
    json.dump(output, f)

print(f"Saved {DATA_DIR / 'weather_test_week.json'}")
print(f"Temperature range this week: {temperature_c.min()} to {temperature_c.max()} \u00b0C")
