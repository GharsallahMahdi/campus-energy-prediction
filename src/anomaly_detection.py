"""
anomaly_detection.py
---------------------
Detects abnormal electricity usage and generates plain-language,
actionable recommendations - supervisor's deliverable (3): "an anomaly
alert and AI-powered energy recommendation module."

Two kinds of alerts, matching the two examples given:
  1. HOURLY ALERTS: a building using much more than typical for its
     category/season/day-type/hour (e.g. "Classroom 301 at 2pm is 40%
     above the typical weekday pattern").
  2. MONTHLY BILL ALERTS: a building's projected monthly bill (based on
     its most recent week of usage) is well above the seasonal average
     for buildings of its type.

"Typical" usage comes from typical_profiles.csv (built by
typical_profiles.py). Electricity price uses Shanghai's real published
Tier-1 residential rate (State Grid Shanghai, 2026: ¥0.53/kWh) as a
stated, citable assumption - campus rates may differ and should be
confirmed if real billing data becomes available.

Reads:  ../data/campus_energy_simulated.csv
       ../data/typical_profiles.csv
       ../data/building_metadata.csv
Writes: ../data/anomaly_alerts.csv
       ../data/monthly_bill_alerts.csv

Run from the project root:  python3 src/anomaly_detection.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

HOURLY_THRESHOLD = 1.35       # flag if actual > 35% above typical
MIN_TYPICAL_KWH = 2.0         # ignore near-zero typical values (avoids noise)
BILL_THRESHOLD = 1.30         # flag if projected monthly bill > 30% above seasonal avg
PRICE_PER_KWH = 0.53          # RMB, Shanghai State Grid Tier-1 residential rate (2026)

df = pd.read_csv(DATA_DIR / "campus_energy_simulated.csv",
                 dtype={"building_id": str}, parse_dates=["timestamp"])
profiles = pd.read_csv(DATA_DIR / "typical_profiles.csv", dtype={"category": str})
meta = pd.read_csv(DATA_DIR / "building_metadata.csv", dtype={"building_id": str})
df = df.merge(meta[["building_id", "name_en", "category"]], on="building_id", how="left")

# ======================================================================
# PART 1: HOURLY ANOMALY ALERTS
# ======================================================================
# IMPORTANT: compare each building to its OWN typical pattern (same
# building, same season/day-type/hour), not the category average.
# A category like "teaching" mixes very different building types (a
# library behaves nothing like a lecture hall even though both are
# "teaching"), so a category-level average produces false alarms for any
# building whose own normal pattern differs from the category's mix.
# Injected test anomalies AND academic-break days (Jan/Jul/Aug, when usage
# drops 60-70%, simulated in simulate_data.py) are excluded here, so
# neither contaminates what "typical" means for a normal operating day -
# otherwise break days drag the average down and make every normal day
# look like a false anomaly by comparison.
baseline_source = df[(df["is_injected_anomaly"] == 0) & (df["is_break"] == 0)]
own_profile = (baseline_source.groupby(["building_id", "season", "is_weekend", "hour"])
              ["consumption_kwh"].mean().rename("typical_kwh").reset_index())
df = df.merge(own_profile, on=["building_id", "season", "is_weekend", "hour"], how="left")
df["ratio"] = df["consumption_kwh"] / df["typical_kwh"].clip(lower=0.1)
df["pct_above_typical"] = (df["ratio"] - 1) * 100

flagged = df[(df["ratio"] >= HOURLY_THRESHOLD) & (df["typical_kwh"] >= MIN_TYPICAL_KWH)].copy()


def hourly_message(row):
    pct = round(row["pct_above_typical"])
    time_str = row["timestamp"].strftime("%b %d, %H:%M")
    day_kind = "weekend" if row["is_weekend"] else "weekday"
    hint = "check if equipment (AC, lights, computers) was left on" \
        if pct < 150 else "check immediately - this is a large, unusual spike"
    return (f"{row['name_en']} ({row['building_id']}) at {time_str} is {pct}% above "
           f"the typical {day_kind} pattern for this time ({row['consumption_kwh']:.1f} kWh "
           f"vs. {row['typical_kwh']:.1f} kWh expected) - {hint}.")


flagged["message"] = flagged.apply(hourly_message, axis=1)
flagged_out = flagged[["building_id", "name_en", "timestamp", "consumption_kwh",
                       "typical_kwh", "pct_above_typical", "is_injected_anomaly", "message"]]
flagged_out = flagged_out.sort_values("pct_above_typical", ascending=False)
flagged_out.to_csv(DATA_DIR / "anomaly_alerts.csv", index=False)

print(f"=== Hourly anomaly alerts: {len(flagged_out)} flagged hours "
     f"(threshold: {int((HOURLY_THRESHOLD-1)*100)}% above typical) ===\n")
for _, r in flagged_out.head(8).iterrows():
    tag = "  [INJECTED TEST CASE]" if r["is_injected_anomaly"] else ""
    print(f"- {r['message']}{tag}")

n_caught = flagged_out["is_injected_anomaly"].sum()
n_total_injected = df["is_injected_anomaly"].sum()
n_injected_hours = df.loc[df["is_injected_anomaly"] == 1, "building_id"].count()
print(f"\nDetector sanity check: caught {n_caught} of the injected test "
     f"anomaly hours (out of {n_injected_hours} injected).")

# ======================================================================
# PART 2: MONTHLY BILL PROJECTION ALERTS
# ======================================================================
# "Typical" monthly kWh per BUILDING (not category - see note above),
# using that building's own average daily usage in the current season.
current_season = df.loc[df["timestamp"] == df["timestamp"].max(), "season"].iloc[0]
own_season_daily_avg = (baseline_source[baseline_source["season"] == current_season]
                        .groupby("building_id")["consumption_kwh"].mean() * 24)
typical_monthly_kwh = (own_season_daily_avg * 30).rename("typical_monthly_kwh").reset_index()

last_7_days_end = df["timestamp"].max()
last_7_days_start = last_7_days_end - pd.Timedelta(days=6)
recent = df[df["timestamp"] >= last_7_days_start]
recent_daily_avg = recent.groupby("building_id")["consumption_kwh"].mean() * 24
projected_monthly_kwh = (recent_daily_avg * 30).rename("projected_monthly_kwh").reset_index()

bill = projected_monthly_kwh.merge(meta[["building_id", "name_en", "category"]], on="building_id")
bill = bill.merge(typical_monthly_kwh, on="building_id", how="left")
bill["projected_bill_rmb"] = (bill["projected_monthly_kwh"] * PRICE_PER_KWH).round(0)
bill["typical_bill_rmb"] = (bill["typical_monthly_kwh"] * PRICE_PER_KWH).round(0)
bill["pct_above_typical"] = ((bill["projected_monthly_kwh"] / bill["typical_monthly_kwh"] - 1) * 100).round(0)

bill_flagged = bill[bill["pct_above_typical"] >= (BILL_THRESHOLD - 1) * 100].copy()


def bill_message(row):
    tips = [
        "turn off AC/heating when the room is empty",
        "unplug chargers and devices on standby",
        "use natural light during the day instead of overhead lighting",
    ]
    tip_text = "; ".join(tips[:3])
    return (f"{row['name_en']} ({row['building_id']})'s projected monthly bill is "
           f"\u00a5{row['projected_bill_rmb']:.0f}, {row['pct_above_typical']:.0f}% above "
           f"the seasonal average (\u00a5{row['typical_bill_rmb']:.0f}). Tips: {tip_text}.")


bill_flagged["message"] = bill_flagged.apply(bill_message, axis=1)
bill_flagged = bill_flagged.sort_values("pct_above_typical", ascending=False)
bill_flagged.to_csv(DATA_DIR / "monthly_bill_alerts.csv", index=False)

print(f"\n=== Monthly bill alerts: {len(bill_flagged)} buildings flagged "
     f"(projected from the last 7 days, threshold: {int((BILL_THRESHOLD-1)*100)}% above seasonal avg) ===\n")
for _, r in bill_flagged.head(5).iterrows():
    print(f"- {r['message']}")

print(f"\nSaved {DATA_DIR / 'anomaly_alerts.csv'} and {DATA_DIR / 'monthly_bill_alerts.csv'}")
