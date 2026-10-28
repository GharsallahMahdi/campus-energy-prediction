"""
detect_room_anomalies.py
--------------------------
Same idea as anomaly_detection.py, but for individual ROOMS instead of
whole buildings: compares each room's actual usage to its OWN typical
pattern (same room, same season, day-type, hour) - never to another
room or a category average, which would cause false alarms.

Reads:  ../data/rooms_simulated.csv
Writes: ../data/room_anomaly_alerts.json

Run from the project root:  python3 src/detect_room_anomalies.py
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ROOM_IDS = ["9-101", "9-105", "9-110", "9-115"]

HOURLY_THRESHOLD = 1.35
MIN_TYPICAL_KWH = 0.3
MIN_ABS_DIFF_KWH = 0.25
LOOKBACK_DAYS = 30
PRICE_PER_KWH = 0.53  # RMB, same real Shanghai State Grid Tier-1 rate used in anomaly_detection.py

df = pd.read_csv(DATA_DIR / "rooms_simulated.csv", dtype={"room_id": str}, parse_dates=["timestamp"])
df = df[df.room_id.isin(ROOM_IDS)]

# "Typical" excludes injected test anomalies AND academic-break days,
# and is grouped by SEASON too - skipping season grouping was an earlier
# bug here: it let winter's naturally higher usage look like a false
# anomaly against a blended year-round average.
baseline = df[(df.is_injected_anomaly == 0) & (df.is_break == 0)]
typical = baseline.groupby(["room_id", "season", "is_weekend", "hour"])["consumption_kwh"].mean()
df = df.merge(typical.rename("typical_kwh"), on=["room_id", "season", "is_weekend", "hour"], how="left")
df["ratio"] = df["consumption_kwh"] / df["typical_kwh"].clip(lower=0.05)
df["pct_above_typical"] = (df["ratio"] - 1) * 100
df["abs_diff"] = df["consumption_kwh"] - df["typical_kwh"]

recent_start = df["timestamp"].max() - pd.Timedelta(days=LOOKBACK_DAYS - 1)
recent = df[df["timestamp"] >= recent_start]
flagged = recent[(recent.ratio >= HOURLY_THRESHOLD)
                 & (recent.typical_kwh >= MIN_TYPICAL_KWH)
                 & (recent.abs_diff >= MIN_ABS_DIFF_KWH)].sort_values("pct_above_typical", ascending=False)

# Extra cost of each flagged hour, same approach as anomaly_detection.py
flagged = flagged.copy()
flagged["extra_kwh"] = (flagged["consumption_kwh"] - flagged["typical_kwh"]).clip(lower=0)
flagged["extra_cost_rmb"] = (flagged["extra_kwh"] * PRICE_PER_KWH).round(2)

room_alerts = {}
for rid in ROOM_IDS:
    rf = flagged[flagged.room_id == rid]
    total_extra_cost = round(float(rf["extra_cost_rmb"].sum()), 2)
    room_alerts[rid] = {
        "alert_count": int(len(rf)),
        "top_message": None,
        "total_extra_cost_rmb": total_extra_cost,
        "cost_sentence": None,
    }
    if len(rf) > 0:
        top = rf.iloc[0]
        room_alerts[rid]["top_message"] = (
            f"\u26a0 Room {rid.split('-')[1]} on {top.timestamp.strftime('%b %d, %H:%M')} is "
            f"{top.pct_above_typical:.0f}% above typical ({top.consumption_kwh:.2f} kWh vs "
            f"{top.typical_kwh:.2f} kWh expected, an extra \u00a5{top.extra_cost_rmb:.2f} for "
            f"that hour) \u2014 check if AC/equipment was left on."
        )
        room_alerts[rid]["cost_sentence"] = (
            f"Room {rid.split('-')[1]}'s flagged anomalies cost an estimated "
            f"\u00a5{total_extra_cost:.2f} extra over the last {LOOKBACK_DAYS} days."
        )

with open(DATA_DIR / "room_anomaly_alerts.json", "w") as f:
    json.dump(room_alerts, f)

print(f"Saved {DATA_DIR / 'room_anomaly_alerts.json'}")
for rid in ROOM_IDS:
    print(f"  {rid}: {room_alerts[rid]['alert_count']} alerts in the last {LOOKBACK_DAYS} days, "
         f"\u00a5{room_alerts[rid]['total_extra_cost_rmb']:.2f} total extra estimated cost")
    if room_alerts[rid]["top_message"]:
        print(f"    {room_alerts[rid]['top_message']}")
