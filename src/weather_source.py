"""
weather_source.py
-------------------
Shared weather loader used by generate_weather.py, forecast_buildings_j9.py
and forecast_rooms.py, so all three use the exact same weather for the
forecast test week - never independently resampled noise.

Priority:
  1. If ../data/real_weather_shanghai.csv exists (made by
     fetch_real_weather.py), use that REAL recorded Shanghai weather,
     aligned POSITIONALLY to the forecast horizon (hour 0 of the real
     file -> hour 0 of the forecast week, and so on). The real file
     covers the same calendar week one year earlier, since true future
     weather doesn't exist yet.
  2. Otherwise, fall back to simulated weather grounded in real published
     Shanghai monthly climate normals (climate-zone.com / weatherbox.io),
     the same approach used before real data was wired in.

Either way, this returns BOTH temperature and humidity for every hour of
the horizon - humidity is a genuine model feature now, not just
temperature.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MONTHLY_TEMP_C = {1: 5.0, 2: 5.6, 3: 9.4, 4: 15.0, 5: 20.6, 6: 23.9,
                  7: 28.3, 8: 27.8, 9: 24.4, 10: 19.4, 11: 13.3, 12: 7.8}
MONTHLY_HUMIDITY = {1: 70, 2: 72, 3: 75, 4: 76, 5: 78, 6: 81,
                    7: 80, 8: 80, 9: 76, 10: 72, 11: 70, 12: 69}


def get_horizon_weather(horizon_timestamps, seed=42):
    """
    horizon_timestamps: a pandas DatetimeIndex or list of Timestamps for
    the forecast week (e.g. 168 hourly timestamps).

    Returns a DataFrame indexed the same length as horizon_timestamps,
    with columns temperature_c and humidity_pct, plus prints which
    source was used.
    """
    horizon_timestamps = pd.DatetimeIndex(horizon_timestamps)
    real_path = DATA_DIR / "real_weather_shanghai.csv"

    if real_path.exists():
        real = pd.read_csv(real_path, parse_dates=["timestamp"]).sort_values("timestamp")
        if len(real) >= len(horizon_timestamps):
            real = real.iloc[:len(horizon_timestamps)].reset_index(drop=True)
            real_start = real["timestamp"].iloc[0].strftime("%b %d, %Y")
            real_end = real["timestamp"].iloc[-1].strftime("%b %d, %Y")
            print(f"[weather_source] Using REAL recorded Shanghai weather from "
                 f"{real_start} to {real_end} "
                 f"({real_path.name}), aligned positionally to the forecast week.")
            result = pd.DataFrame({
                "timestamp": horizon_timestamps,
                "temperature_c": real["temperature_c"].values,
                "humidity_pct": real["humidity_pct"].values,
            })
            result.attrs["source_type"] = "real"
            result.attrs["source_description"] = (
                f"Real recorded Shanghai weather from {real_start}\u2013{real_end}, "
                f"used as a stand-in for this forecast week since true future "
                f"weather does not exist yet."
            )
            return result
        else:
            print(f"[weather_source] {real_path.name} found but has fewer rows "
                 f"than needed ({len(real)} < {len(horizon_timestamps)}) - "
                 f"falling back to simulated weather.")
    else:
        print(f"[weather_source] No {real_path.name} found - using simulated "
             f"weather grounded in real Shanghai climate normals. Run "
             f"fetch_real_weather.py to use genuinely real data instead.")

    # ---- Fallback: simulated, grounded in real monthly climate normals ----
    rng = np.random.RandomState(seed)
    month = horizon_timestamps.month.values
    hour = horizon_timestamps.hour.values
    base_temp = np.array([MONTHLY_TEMP_C[m] for m in month])
    base_hum = np.array([MONTHLY_HUMIDITY[m] for m in month])
    daily_swing = 4 * np.sin((hour - 9) / 24 * 2 * np.pi)
    temperature_c = (base_temp + daily_swing + rng.normal(0, 1.0, len(horizon_timestamps))).round(1)
    humidity_pct = np.clip(base_hum - 0.5 * daily_swing + rng.normal(0, 4, len(horizon_timestamps)), 15, 100).round(1)

    result = pd.DataFrame({
        "timestamp": horizon_timestamps,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
    })
    result.attrs["source_type"] = "simulated"
    result.attrs["source_description"] = (
        "Simulated weather grounded in real Shanghai monthly climate normals "
        "(temperature and humidity averages), not actual hourly records. "
        "Run fetch_real_weather.py to use genuinely real recorded weather instead."
    )
    return result
