"""
typical_profiles.py
--------------------
Generates "typical load profile" charts: the average 24-hour consumption
curve for each building category (dormitory vs teaching/office), split by
season and by weekday/weekend. This is deliverable (1) your supervisor
asked for.

Reads:  ../data/campus_energy_simulated.csv, ../data/building_metadata.csv
Writes: ../data/typical_profiles.png  (one grid of charts)
        ../data/typical_profiles.csv (the underlying numbers)

Run from the project root:  python3 src/typical_profiles.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_DIR / "campus_energy_simulated.csv", dtype={"building_id": str})
meta = pd.read_csv(DATA_DIR / "building_metadata.csv", dtype={"building_id": str})
df = df.merge(meta[["building_id", "category"]], on="building_id", how="left")

# Average consumption per category, season, weekday/weekend, hour
profiles = (
    df.groupby(["category", "season", "is_weekend", "hour"])["consumption_kwh"]
    .mean()
    .reset_index()
)
profiles.to_csv(DATA_DIR / "typical_profiles.csv", index=False)

seasons = ["winter", "spring", "summer", "fall"]
categories = sorted(df["category"].unique())
colors = {"dormitory": "#2f7d5f", "teaching": "#3d7ab5", "office": "#c77b2e"}

fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
day_labels = {0: "Weekday", 1: "Weekend"}

for row, is_weekend in enumerate([0, 1]):
    for col, season in enumerate(seasons):
        ax = axes[row, col]
        for cat in categories:
            sub = profiles[(profiles.category == cat) & (profiles.season == season)
                          & (profiles.is_weekend == is_weekend)].sort_values("hour")
            ax.plot(sub["hour"], sub["consumption_kwh"], label=cat,
                   color=colors.get(cat, "gray"), linewidth=2)
        ax.set_title(f"{season.capitalize()} - {day_labels[is_weekend]}", fontsize=11)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("Hour")
        if col == 0:
            ax.set_ylabel("Avg. consumption (kWh)")
        ax.grid(alpha=0.25)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(categories),
          bbox_to_anchor=(0.5, 1.04), frameon=False)
fig.suptitle("Typical Load Profiles by Building Category, Season and Day Type",
            fontsize=14, y=1.1)
fig.tight_layout()
fig.savefig(DATA_DIR / "typical_profiles.png", dpi=150, bbox_inches="tight")

print("Saved typical_profiles.png and typical_profiles.csv")
print("\nPeak hour and value per category (across all seasons/days):")
peak = (profiles.groupby("category")
       .apply(lambda g: g.loc[g["consumption_kwh"].idxmax(),
                               ["hour", "season", "consumption_kwh"]])
       )
print(peak.round(2).to_string())
