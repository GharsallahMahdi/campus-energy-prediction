"""
build_dashboard_data.py
------------------------
Builds the data file the campus map dashboard reads from. For each
building, computes:
  - its own TYPICAL weekday/weekend 24-hour pattern (same baseline the
    anomaly detector uses - excludes injected test anomalies and
    academic-break days, so "typical" means a normal operating day)
  - its RECENT ACTUAL weekday/weekend 24-hour pattern, from the last 14
    days of data (so you can see at a glance whether recent behavior
    matches or deviates from what's typical)
  - how many anomaly alerts that building had in anomaly_alerts.csv

Reads:  ../data/campus_energy_simulated.csv, ../data/building_metadata.csv,
       ../data/anomaly_alerts.csv (if present)
Writes: ../dashboard/data.json (if the folder exists) and
       ../data/dashboard_data.json

Run from the project root:  python3 src/build_dashboard_data.py
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"

df = pd.read_csv(DATA_DIR / "campus_energy_simulated.csv",
                 dtype={"building_id": str}, parse_dates=["timestamp"])
meta = pd.read_csv(DATA_DIR / "building_metadata.csv", dtype={"building_id": str})

# "Typical": excludes injected test anomalies and academic-break days,
# same definition anomaly_detection.py uses.
typical_source = df[(df["is_injected_anomaly"] == 0) & (df["is_break"] == 0)]
typical = (typical_source.groupby(["building_id", "is_weekend", "hour"])
          ["consumption_kwh"].mean().round(1))

# "Recent actual": the last 14 days of data, as-is (including anomalies,
# since the point is to show what's really happening right now)
recent_start = df["timestamp"].max() - pd.Timedelta(days=13)
recent_source = df[df["timestamp"] >= recent_start]
recent = (recent_source.groupby(["building_id", "is_weekend", "hour"])
         ["consumption_kwh"].mean().round(1))

forecast_path = DATA_DIR / "forecast_results.json"
forecast_data = {}
forecast_meta = {}
if forecast_path.exists():
    with open(forecast_path) as f:
        _fc = json.load(f)
    forecast_data = _fc["buildings"]
    forecast_meta = {"forecast_start": _fc["forecast_start"],
                     "first_24h_index": _fc["first_24h_index"]}

alerts_path = DATA_DIR / "anomaly_alerts.csv"
alert_counts = {}
if alerts_path.exists():
    alerts = pd.read_csv(alerts_path, dtype={"building_id": str}, parse_dates=["timestamp"])
    # Only count alerts within the same recent window the chart shows -
    # otherwise a full year of the detector's expected ~1% background
    # false-positive rate dilutes the real signal (every building ends up
    # with 100+ "alerts" and the genuine problems don't stand out).
    recent_alerts = alerts[alerts["timestamp"] >= recent_start]
    alert_counts = recent_alerts["building_id"].value_counts().to_dict()

result = []
for _, b in meta.iterrows():
    bid = b["building_id"]

    def series(src, is_wknd):
        vals = []
        for h in range(24):
            key = (bid, is_wknd, h)
            vals.append(float(src.get(key, 0)))
        return vals

    result.append({
        "id": str(bid),
        "name": b["name_en"],
        "category": b["category"],
        "x": float(b["x_pct"]),
        "y": float(b["y_pct"]),
        "weekday_typical": series(typical, 0),
        "weekend_typical": series(typical, 1),
        "weekday_actual": series(recent, 0),
        "weekend_actual": series(recent, 1),
        "alert_count": int(alert_counts.get(bid, 0)),
        "forecast": forecast_data.get(bid, None),
        "forecast_start": forecast_meta.get("forecast_start"),
        "forecast_first_24h_index": forecast_meta.get("first_24h_index"),
    })

out_json = json.dumps(result)
(DATA_DIR / "dashboard_data.json").write_text(out_json)
print(f"Wrote {DATA_DIR / 'dashboard_data.json'} ({len(result)} buildings)")

if DASHBOARD_DIR.exists():
    (DASHBOARD_DIR / "data.json").write_text(out_json)
    print(f"Also updated {DASHBOARD_DIR / 'data.json'}")

flagged = [r for r in result if r["alert_count"] > 0]
print(f"\nBuildings with at least one anomaly alert: {len(flagged)}")
for r in sorted(flagged, key=lambda r: -r["alert_count"])[:6]:
    print(f"  {r['name']} ({r['id']}): {r['alert_count']} alerts")
