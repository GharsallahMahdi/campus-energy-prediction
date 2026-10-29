"""
build_map2_data.py
--------------------
Assembles data.json for the focused, 2-marker campus map: Computer
Building (J) and Dormitory 9, with Dormitory 9 further drilling into
two specific rooms (9-101 clean baseline, 9-105 with the injected
"AC left on" anomaly).

Reads:
  ../data/campus_energy_simulated.csv, ../data/rooms_simulated.csv,
  ../data/building_metadata.csv, ../data/forecast_results.json,
  ../data/forecast_rooms_results.json, ../data/room_anomaly_alerts.json,
  ../data/anomaly_alerts.csv
Writes: ../data/map2_data.json
       ../dashboard-map2/data.json (if that folder exists)

Run from the project root:  python3 src/build_map2_data.py
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAP2_DIR = Path(__file__).resolve().parent.parent / "dashboard-map2"

BUILDING_IDS = ["J", "9"]
ROOM_IDS = ["9-101", "9-105", "9-110", "9-115"]

buildings_df = pd.read_csv(DATA_DIR / "campus_energy_simulated.csv",
                           dtype={"building_id": str}, parse_dates=["timestamp"])
rooms_df = pd.read_csv(DATA_DIR / "rooms_simulated.csv",
                       dtype={"building_id": str, "room_id": str}, parse_dates=["timestamp"])
meta = pd.read_csv(DATA_DIR / "building_metadata.csv", dtype={"building_id": str}).set_index("building_id")

with open(DATA_DIR / "forecast_results.json") as f:
    building_forecasts = json.load(f)["buildings"]
with open(DATA_DIR / "forecast_rooms_results.json") as f:
    room_forecast_raw = json.load(f)
with open(DATA_DIR / "room_anomaly_alerts.json") as f:
    room_alerts = json.load(f)

comparison_path = DATA_DIR / "unified_model_comparison.csv"
per_entity_performance = {}
ashrae_performance = []
if comparison_path.exists():
    cmp_df = pd.read_csv(comparison_path)
    sim = cmp_df[cmp_df.dataset == "Simulated STIEI"]
    for entity_id, g in sim.groupby("entity"):
        entity_id = str(entity_id)
        by_horizon = {}
        for horizon, hg in g.groupby("horizon"):
            by_horizon[horizon] = [
                {"model": r["model"], "MAE": r["MAE"], "RMSE": r["RMSE"],
                 "MAPE_pct": r["MAPE_pct"], "R2": r["R2"]}
                for _, r in hg.iterrows()
            ]
        per_entity_performance[entity_id] = by_horizon

    ashrae = cmp_df[cmp_df.dataset == "Real ASHRAE/BDG2"]
    for _, r in ashrae.iterrows():
        ashrae_performance.append({
            "model": r["model"], "MAE": r["MAE"], "RMSE": r["RMSE"],
            "MAPE_pct": r["MAPE_pct"], "R2": r["R2"],
            "entity": r["entity"], "evaluation_method": r["evaluation_method"], "unit": r["unit"],
        })
    print(f"Loaded model performance for {len(per_entity_performance)} entities "
         f"and {len(ashrae_performance)} ASHRAE rows from unified_model_comparison.csv")
else:
    print(f"No {comparison_path.name} found - Model Performance tab will be empty. "
         f"Run build_comparison_table.py first if you want it populated.")

insights_path = DATA_DIR / "insights.json"
insights = {}
if insights_path.exists():
    with open(insights_path) as f:
        insights = json.load(f)

weather_path = DATA_DIR / "weather_test_week.json"
weather_temp = None
weather_source_type = "simulated"
weather_source_description = ""
if weather_path.exists():
    with open(weather_path) as f:
        _wd = json.load(f)
    weather_temp = _wd["temperature_c"]
    weather_source_type = _wd.get("source_type", "simulated")
    weather_source_description = _wd.get("source_description", "")

building_alert_counts = {}
building_alert_costs = {}
alerts_path = DATA_DIR / "anomaly_alerts.csv"
if alerts_path.exists():
    alerts = pd.read_csv(alerts_path, dtype={"building_id": str}, parse_dates=["timestamp"])
    recent_start = buildings_df["timestamp"].max() - pd.Timedelta(days=13)
    recent_alerts = alerts[alerts["timestamp"] >= recent_start]
    building_alert_counts = recent_alerts["building_id"].value_counts().to_dict()
    if "extra_cost_rmb" in recent_alerts.columns:
        building_alert_costs = recent_alerts.groupby("building_id")["extra_cost_rmb"].sum().round(2).to_dict()


SEASONS = ["winter", "spring", "summer", "fall"]


def typical_profile(df, id_col, entity_id):
    """Returns the typical weekday/weekend pattern POOLED across the whole
    year (kept for backward compatibility - the Insights tab uses this),
    PLUS a by_season breakdown (winter/spring/summer/fall x weekday/weekend)
    for the dashboard's season selector."""
    sub = df[(df[id_col] == entity_id) & (df["is_break"] == 0)]
    if "is_injected_anomaly" in sub.columns:
        sub = sub[sub["is_injected_anomaly"] == 0]

    prof = sub.groupby(["is_weekend", "hour"])["consumption_kwh"].mean().round(2)
    by_season = {}
    for season in SEASONS:
        s_sub = sub[sub["season"] == season] if "season" in sub.columns else sub
        s_prof = s_sub.groupby(["is_weekend", "hour"])["consumption_kwh"].mean().round(2)
        by_season[season] = {
            "weekday": [float(s_prof.get((0, h), 0)) for h in range(24)],
            "weekend": [float(s_prof.get((1, h), 0)) for h in range(24)],
        }

    return {
        "weekday": [float(prof.get((0, h), 0)) for h in range(24)],
        "weekend": [float(prof.get((1, h), 0)) for h in range(24)],
        "by_season": by_season,
    }


def last_30_days(df, id_col, entity_id):
    sub = df[df[id_col] == entity_id].copy()
    sub["date"] = sub["timestamp"].dt.date
    daily = sub.groupby("date")["consumption_kwh"].sum().round(2)
    daily = daily.sort_index().tail(30)
    return {"dates": [str(d) for d in daily.index], "values": [float(v) for v in daily.values]}


entities = {}

# ---- Buildings (map markers) ----
for bid in BUILDING_IDS:
    row = meta.loc[bid]
    entities[bid] = {
        "id": bid, "type": "building",
        "name": row["name_en"], "category": row["category"],
        "x": float(row["x_pct"]), "y": float(row["y_pct"]),
        "profile": typical_profile(buildings_df, "building_id", bid),
        "history_30d": last_30_days(buildings_df, "building_id", bid),
        "forecast": building_forecasts.get(bid),
        "alert_count": int(building_alert_counts.get(bid, 0)),
        "alert_message": None,
        "total_extra_cost_rmb": float(building_alert_costs.get(bid, 0)),
        "temperature_c": weather_temp,
        "insight": insights.get(bid),
        "model_performance": per_entity_performance.get(bid),
    }
    print(f"Built building entity {bid} ({row['name_en']})")

# ---- Rooms (drill-down inside Dormitory 9, no map position of their own) ----
for rid in ROOM_IDS:
    room_number = rooms_df.loc[rooms_df.room_id == rid, "room_number"].iloc[0]
    alert_info = room_alerts.get(rid, {"alert_count": 0, "top_message": None})
    entities[rid] = {
        "id": rid, "type": "room",
        "name": f"Room {room_number}", "parent_building": "9",
        "profile": typical_profile(rooms_df, "room_id", rid),
        "history_30d": last_30_days(rooms_df, "room_id", rid),
        "forecast": room_forecast_raw["rooms"].get(rid),
        "alert_count": alert_info["alert_count"],
        "alert_message": alert_info["top_message"],
        "total_extra_cost_rmb": float(alert_info.get("total_extra_cost_rmb", 0)),
        "temperature_c": weather_temp,
        "insight": insights.get(rid),
        "model_performance": per_entity_performance.get(rid),
    }
    print(f"Built room entity {rid} (alerts: {alert_info['alert_count']})")

with open(DATA_DIR / "forecast_results.json") as f:
    _fc_meta = json.load(f)

output = {
    "building_ids": BUILDING_IDS,
    "dorm_building_id": "9",
    "room_ids": ROOM_IDS,
    "forecast_start": _fc_meta["forecast_start"],
    "first_24h_index": _fc_meta["first_24h_index"],
    "weather_source_type": weather_source_type,
    "weather_source_description": weather_source_description,
    "ashrae_performance": ashrae_performance,
    "entities": entities,
}

out_json = json.dumps(output)
(DATA_DIR / "map2_data.json").write_text(out_json)
print(f"\nWrote {DATA_DIR / 'map2_data.json'}")

if MAP2_DIR.exists():
    (MAP2_DIR / "data.json").write_text(out_json)
    print(f"Also updated {MAP2_DIR / 'data.json'}")
