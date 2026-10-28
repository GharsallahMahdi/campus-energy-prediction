"""
generate_insights.py
----------------------
Builds the "AI-based analysis module" the supervisor asked for: goes
beyond raw forecasting to produce actionable advice for facility
managers. For each entity (Computer Building J, Dormitory 9, and its
two tracked rooms), this:

  1. Identifies PEAK-LOAD hours from the typical weekday/weekend pattern.
  2. Suggests LOAD-SHIFTING measures, tailored by building type (dorm
     vs teaching/office) and the actual peak timing found.
  3. Pulls in ABNORMAL USAGE status from the anomaly detectors already
     built (anomaly_detection.py / detect_room_anomalies.py) - not
     recomputed, just referenced.
  4. Writes a short PLAIN-LANGUAGE paragraph combining all of the above,
     the way a human energy analyst might summarize findings for a
     non-technical manager.

Reads:  ../data/campus_energy_simulated.csv, ../data/rooms_simulated.csv,
       ../data/building_metadata.csv, ../data/anomaly_alerts.csv,
       ../data/room_anomaly_alerts.json
Writes: ../data/insights.json

Run from the project root:  python3 src/generate_insights.py
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BUILDING_IDS = ["J", "9"]
ROOM_IDS = ["9-101", "9-105", "9-110", "9-115"]

PEAK_N = 2  # how many peak hours to report per entity


def typical_weekday_profile(df, id_col, entity_id):
    """Average weekday hourly consumption, excluding break days and
    injected test anomalies, so peaks reflect NORMAL operation."""
    sub = df[(df[id_col] == entity_id) & (df["is_break"] == 0) & (df["is_weekend"] == 0)]
    if "is_injected_anomaly" in sub.columns:
        sub = sub[sub["is_injected_anomaly"] == 0]
    return sub.groupby("hour")["consumption_kwh"].mean()


def find_peak_hours(profile, n=PEAK_N):
    top = profile.sort_values(ascending=False).head(n)
    return [(int(h), round(float(v), 2)) for h, v in top.items()]


def hour_range_label(hours):
    """Turns [9, 10] into '9:00-11:00', etc - groups consecutive hours."""
    hours = sorted(hours)
    return ", ".join(f"{h:02d}:00-{(h + 1) % 24:02d}:00" for h in hours)


def load_shift_suggestion(category, peak_hours):
    peak_set = set(h for h, _ in peak_hours)
    evening = any(18 <= h <= 22 for h in peak_set)
    daytime = any(8 <= h <= 17 for h in peak_set)
    overnight = any(h <= 5 or h >= 23 for h in peak_set)

    if category == "dormitory":
        if evening:
            return ("Peak load falls in the evening, when students return and use "
                   "appliances together. Consider encouraging laundry, water heaters "
                   "and chargers to run before 18:00 or after 22:00 to flatten this peak.")
        if overnight:
            return ("Unusually high overnight usage detected - this often points to "
                   "equipment left running (AC, lights) rather than normal occupancy "
                   "patterns. Worth a manual check.")
        return ("Load is relatively spread through the day; no single dominant peak "
                "to target for load-shifting.")
    else:  # teaching / office
        if daytime:
            return ("Peak load aligns with class/office hours, as expected. Scheduling "
                   "non-critical equipment (cleaning machinery, non-urgent maintenance, "
                   "battery charging for shared devices) outside these hours would avoid "
                   "adding to the daytime peak.")
        if evening or overnight:
            return ("Usage outside normal operating hours was detected for a teaching/"
                   "office building - worth checking whether equipment was left on "
                   "after hours.")
        return ("Load is relatively flat; no strong peak-hour target for load-shifting.")


def build_entity_insight(name, category, weekday_profile, alert_count, alert_message):
    peak_hours = find_peak_hours(weekday_profile)
    peak_label = hour_range_label([h for h, _ in peak_hours])
    peak_val = max(v for _, v in peak_hours)
    suggestion = load_shift_suggestion(category, peak_hours)

    if alert_count and alert_count > 0:
        anomaly_sentence = (f"{alert_count} anomaly alert(s) were flagged for {name} in the "
                           f"last 14-30 days. {alert_message or ''}")
    else:
        anomaly_sentence = f"No anomalies currently flagged for {name}."

    summary = (f"{name}: typical weekday peak around {peak_label} "
              f"(~{peak_val:.1f} kWh). {suggestion} {anomaly_sentence}")

    return {
        "peak_hours": peak_hours,
        "peak_label": peak_label,
        "load_shift_suggestion": suggestion,
        "anomaly_count": int(alert_count or 0),
        "anomaly_sentence": anomaly_sentence,
        "summary": summary,
    }


def main():
    buildings_df = pd.read_csv(DATA_DIR / "campus_energy_simulated.csv",
                               dtype={"building_id": str}, parse_dates=["timestamp"])
    rooms_df = pd.read_csv(DATA_DIR / "rooms_simulated.csv",
                           dtype={"room_id": str}, parse_dates=["timestamp"])
    meta = pd.read_csv(DATA_DIR / "building_metadata.csv", dtype={"building_id": str}).set_index("building_id")

    building_alert_counts = {}
    if (DATA_DIR / "anomaly_alerts.csv").exists():
        alerts = pd.read_csv(DATA_DIR / "anomaly_alerts.csv", dtype={"building_id": str},
                             parse_dates=["timestamp"])
        recent_start = buildings_df["timestamp"].max() - pd.Timedelta(days=13)
        recent = alerts[alerts["timestamp"] >= recent_start]
        building_alert_counts = recent["building_id"].value_counts().to_dict()

    room_alerts = {}
    if (DATA_DIR / "room_anomaly_alerts.json").exists():
        with open(DATA_DIR / "room_anomaly_alerts.json") as f:
            room_alerts = json.load(f)

    insights = {}

    for bid in BUILDING_IDS:
        row = meta.loc[bid]
        profile = typical_weekday_profile(buildings_df, "building_id", bid)
        insights[bid] = build_entity_insight(
            row["name_en"], row["category"], profile,
            building_alert_counts.get(bid, 0), None,
        )

    for rid in ROOM_IDS:
        profile = typical_weekday_profile(rooms_df, "room_id", rid)
        alert_info = room_alerts.get(rid, {"alert_count": 0, "top_message": None})
        insights[rid] = build_entity_insight(
            f"Room {rid.split('-')[1]}", "dormitory", profile,
            alert_info["alert_count"], alert_info["top_message"],
        )

    with open(DATA_DIR / "insights.json", "w") as f:
        json.dump(insights, f, indent=2)

    print(f"Saved {DATA_DIR / 'insights.json'}\n")
    for eid, ins in insights.items():
        print(f"--- {eid} ---")
        print(ins["summary"])
        print()


if __name__ == "__main__":
    main()
