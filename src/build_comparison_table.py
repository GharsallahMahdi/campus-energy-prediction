"""
build_comparison_table.py
---------------------------
Combines results that currently live in 3 separate CSV files into ONE
unified table: your own simulated STIEI campus results (buildings and
rooms) next to the real ASHRAE/BDG2 validation results - exactly the
"(3) model performance comparison tables (on both our own data and
public datasets)" deliverable the supervisor asked for.

Reads:  ../data/forecast_buildings_j9_metrics.csv
       ../data/forecast_rooms_metrics.csv
       ../data/ashrae_results.csv          (from ashrae_experiment.py)
Writes: ../data/unified_model_comparison.csv

Run from the project root:  python3 src/build_comparison_table.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# NOTE on comparability (read before trusting a side-by-side number):
# STIEI and ASHRAE are NOT evaluated the same way, so a lower MAE on one
# side does not mean "better" in an apples-to-apples sense:
#   - STIEI: a genuine RECURSIVE, no-leakage forecast. Once a lag feature
#     would need a value from inside the forecast horizon, the model uses
#     its own earlier prediction instead of the real value.
#   - ASHRAE: a time-based train/test split (train on Jan-Oct, test on
#     Nov-Dec), but the lag features (lag_24, lag_168) are computed from
#     the REAL historical meter readings, including ones that fall inside
#     the test period itself. This is closer to the "oracle" style
#     benchmark from the STIEI experiments, not the strict recursive one.
# The evaluation_method column below makes this explicit in the table
# itself, so it can't be misread as a fair head-to-head.
STIEI_METHOD = "Recursive, no-leakage multi-step forecast"
ASHRAE_METHOD = "Time-based split; lag features use real historical values (oracle-style, not recursive)"

# NOTE on units: STIEI's kWh is defined by our own simulation, so it is
# exactly kWh with no ambiguity. ASHRAE's meter_reading column is
# DOCUMENTED as kWh for electricity meters, but the competition is known
# to have had a unit inconsistency for some site_id 0 buildings in the
# original release (later patched by organizers). We have not re-verified
# this for the specific 30 buildings sampled here, so ASHRAE's unit is
# labeled with that caveat rather than asserted as a clean kWh value.
STIEI_UNIT = "kWh"
ASHRAE_UNIT = "kWh (per ASHRAE's documentation; not independently re-verified for this sample)"

rows = []

# ---------- Simulated STIEI: buildings ----------
b_path = DATA_DIR / "forecast_buildings_j9_metrics.csv"
if b_path.exists():
    b = pd.read_csv(b_path)
    for _, r in b.iterrows():
        rows.append({
            "dataset": "Simulated STIEI", "entity": r["building_id"], "entity_type": "building",
            "model": r["model"], "horizon": r["horizon"], "evaluation_method": STIEI_METHOD,
            "MAE": r["MAE_kWh"], "RMSE": r["RMSE_kWh"], "unit": STIEI_UNIT,
            "MAPE_pct": r["MAPE_pct"], "R2": r["R2"],
        })
    print(f"Loaded {len(b)} rows from forecast_buildings_j9_metrics.csv")
else:
    print(f"Skipped (not found): {b_path.name}")

# ---------- Simulated STIEI: rooms ----------
r_path = DATA_DIR / "forecast_rooms_metrics.csv"
if r_path.exists():
    r_df = pd.read_csv(r_path)
    for _, r in r_df.iterrows():
        rows.append({
            "dataset": "Simulated STIEI", "entity": r["room_id"], "entity_type": "room",
            "model": r["model"], "horizon": r["horizon"], "evaluation_method": STIEI_METHOD,
            "MAE": r["MAE_kWh"], "RMSE": r["RMSE_kWh"], "unit": STIEI_UNIT,
            "MAPE_pct": r["MAPE_pct"], "R2": r["R2"],
        })
    print(f"Loaded {len(r_df)} rows from forecast_rooms_metrics.csv")
else:
    print(f"Skipped (not found): {r_path.name}")

# ---------- Real ASHRAE / BDG2 validation ----------
a_path = DATA_DIR / "ashrae_results.csv"
if a_path.exists():
    a = pd.read_csv(a_path, index_col=0)
    # Only keep the real trained models, not the 3 naive baselines -
    # this table is about comparing OUR models across datasets, and the
    # baselines are already discussed separately in the ASHRAE write-up.
    model_rows = [m for m in a.index if "baseline" not in m.lower()]
    for model_name in model_rows:
        row = a.loc[model_name]
        rows.append({
            "dataset": "Real ASHRAE/BDG2",
            "entity": "30 selected educational buildings (pooled)",
            "entity_type": "real buildings", "model": model_name,
            "horizon": "Nov-Dec 2016 (held-out)", "evaluation_method": ASHRAE_METHOD,
            "MAE": row.get("MAE (kWh)"), "RMSE": row.get("RMSE (kWh)"), "unit": ASHRAE_UNIT,
            "MAPE_pct": row.get("MAPE (%)"), "R2": row.get("R2"),
        })
    print(f"Loaded {len(model_rows)} model rows from ashrae_results.csv")
else:
    print(f"Skipped (not found): {a_path.name} - run ashrae_experiment.py first if you want it included.")

if not rows:
    raise SystemExit("No result files found - run the forecast/ASHRAE scripts first.")

table = pd.DataFrame(rows)
table = table.round({"MAE": 3, "RMSE": 3, "MAPE_pct": 2, "R2": 3})
table.to_csv(DATA_DIR / "unified_model_comparison.csv", index=False)

print(f"\nSaved {DATA_DIR / 'unified_model_comparison.csv'} ({len(table)} rows)\n")
print("=== Simulated STIEI (next 7 days only, for a quick overview) ===")
print(table[(table.dataset == "Simulated STIEI") & (table.horizon == "next_7_days")]
     [["entity", "model", "MAE", "RMSE", "MAPE_pct", "R2"]].to_string(index=False))
print("\n=== Real ASHRAE/BDG2 ===")
print(table[table.dataset == "Real ASHRAE/BDG2"]
     [["model", "MAE", "RMSE", "MAPE_pct", "R2"]].to_string(index=False))
print("\nNote: see the evaluation_method and unit columns in the saved CSV "
     "before comparing MAE/RMSE directly across the two datasets.")
