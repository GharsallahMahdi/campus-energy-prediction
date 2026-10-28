"""
generate_weather.py
---------------------
Generates the weather series used by the dashboard's temperature overlay
chart, for the 7-day forecast test week - via weather_source.py, so this
is GUARANTEED to match exactly what the forecast models actually used
(real recorded Shanghai weather if available, simulated-but-grounded
otherwise). Previously this script had its own separate copy of the
weather logic, which could silently drift out of sync with what the
models saw - now there is one single source of truth.

Writes: ../data/weather_test_week.json

Run from the project root:  python3 src/generate_weather.py
"""

import json
from pathlib import Path

import pandas as pd

from weather_source import get_horizon_weather

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Must match the test week used by forecast_buildings_j9.py / forecast_rooms.py:
# the last 7 days of the simulated year.
horizon = pd.date_range("2026-12-25", periods=7 * 24, freq="h")

weather = get_horizon_weather(horizon)

output = {
    "timestamps": [str(t) for t in weather["timestamp"]],
    "temperature_c": weather["temperature_c"].tolist(),
    "humidity_pct": weather["humidity_pct"].tolist(),
    "source_type": weather.attrs.get("source_type", "simulated"),
    "source_description": weather.attrs.get("source_description", ""),
}
with open(DATA_DIR / "weather_test_week.json", "w") as f:
    json.dump(output, f)

print(f"Saved {DATA_DIR / 'weather_test_week.json'}")
print(f"Temperature range: {weather['temperature_c'].min()} to {weather['temperature_c'].max()} \u00b0C")
print(f"Humidity range: {weather['humidity_pct'].min()} to {weather['humidity_pct'].max()} %")
