"""
fetch_real_weather.py
-----------------------
Fetches REAL historical hourly weather for Shanghai from Open-Meteo's
free historical archive (no API key needed): https://open-meteo.com/

Our forecast test week is set in the FUTURE (Dec 25-31, 2026), and real
weather can't exist for dates that haven't happened yet. So this pulls
the REAL recorded weather from the same calendar week one year earlier
(Dec 25-31, 2025, which has already happened) - a standard, defensible
stand-in, clearly labeled as such.

Run this from YOUR Mac, not inside Claude's sandbox - Open-Meteo isn't
reachable from there. You need the 'requests' library (already installed
if you've used pip before; if not: pip install requests).

Writes: ../data/real_weather_shanghai.csv

Run from the project root:  python3 src/fetch_real_weather.py
"""

from pathlib import Path

import requests
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Shanghai coordinates
LATITUDE = 31.23
LONGITUDE = 121.47

# Real historical week - the same calendar dates as our forecast test
# week, one year earlier (2025 has actually happened; 2026 hasn't yet)
START_DATE = "2025-12-25"
END_DATE = "2025-12-31"

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": "temperature_2m,relative_humidity_2m",
    "timezone": "Asia/Shanghai",
}

print(f"Fetching real Shanghai weather for {START_DATE} to {END_DATE}...")
response = requests.get(url, params=params, timeout=30)
response.raise_for_status()
data = response.json()

df = pd.DataFrame({
    "timestamp": data["hourly"]["time"],
    "temperature_c": data["hourly"]["temperature_2m"],
    "humidity_pct": data["hourly"]["relative_humidity_2m"],
})

df.to_csv(DATA_DIR / "real_weather_shanghai.csv", index=False)
print(f"Saved {DATA_DIR / 'real_weather_shanghai.csv'} ({len(df)} hours)")
print(f"Temperature range: {df.temperature_c.min()} to {df.temperature_c.max()} \u00b0C")
print(f"\nSource: Open-Meteo Historical Weather API (open-meteo.com)")
print(f"Note: this is REAL recorded weather from Dec 2025, used as the best")
print(f"available stand-in since Dec 2026 (our forecast period) hasn't happened yet.")
