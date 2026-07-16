"""
Premier League Home Advantage & Match Outcome Analysis
EDA Script — Python
Produces 6 charts saved as PNG files for use in the datafolio and final report.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────────────────────────
COLOURS = {"H": "#2E4057", "D": "#048A81", "A": "#E94F37"}
SEASON_COLOURS = ["#2E4057", "#048A81", "#F4A261"]
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 14,
                     "axes.labelsize": 12, "legend.fontsize": 10})

# ── Load & combine data ───────────────────────────────────────────────────────
files = {
    "2021-22": "E0__2_.csv",
    "2022-23": "E0__1_.csv",
    "2023-24": "E0.csv",
}

dfs = []
for season, fname in files.items():
    df = pd.read_csv(fname)
    df["Season"] = season
    dfs.append(df)

raw = pd.concat(dfs, ignore_index=True)

# Select columns of interest
cols = ["Season", "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
        "HTHG", "HTAG", "HTR",
        "HS", "AS", "HST", "AST",
        "HF", "AF", "HC", "AC",
        "HY", "AY", "HR", "AR"]

df = raw[cols].copy()
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df["TotalGoals"] = df["FTHG"] + df["FTAG"]
df["GoalDiff"] = df["FTHG"] - df["FTAG"]   # positive = home win

print("=" * 60)
print("DATA PROFILING SUMMARY")
print("=" * 60)
print(f"Total matches: {len(df)}")
print(f"Seasons: {df['Season'].unique().tolist()}")
print(f"Null values in analytical columns: {df[cols].isnull().sum().sum()}")
print(f"\nFull-time results across all seasons:")
print(df["FTR"].value_counts())
pct = df["FTR"].value_counts(normalize=True) * 100
print(f"\nHome win:  {pct['H']:.1f}%")
print(f"Draw:      {pct['D']:.1f}%")
print(f"Away win:  {pct['A']:.1f}%")

# ── Chart 1: Overall win/draw/loss distribution ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
labels = ["Home Win", "Draw", "Away Win"]
values = [pct["H"], pct["D"], pct["A"]]
bars = ax.bar(labels, values,
              color=[COLOURS["H"], COLOURS["D"], COLOURS["A"]],
              width=0.55, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8, f"{val:.1f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=13)
ax.set_ylim(0, 60)
ax.set_ylabel("Percentage of Matches (%)")
ax.set_title("Match Outcomes Across 3 Premier League Seasons (2021–24)", pad=14)
ax.axhline(33.3, color="grey", linestyle="--", linewidth=1, alpha=0.5,
           label="Equal probability (33.3%)")
ax.legend()
plt.tight_layout()
plt.savefig("chart1_overall_outcomes.png")
plt.close()
print("\nSaved chart1_overall_outcomes.png")

# ── Chart 2: Win rates by season ──────────────────────────────────────────────
season_results = df.groupby(["Season", "FTR"]).size().unstack(fill_value=0)
season_pct = season_results.div(season_results.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(season_pct))
width = 0.28
ax.bar(x - width, season_pct["H"], width, label="Home Win",
       color=COLOURS["H"], edgecolor="white")
ax.bar(x,         season_pct["D"], width, label="Draw",
       color=COLOURS["D"], edgecolor="white")
ax.bar(x + width, season_pct["A"], width, label="Away Win",
       color=COLOURS["A"], edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(season_pct.index)
ax.set_ylabel("Percentage of Matches (%)")
ax.set_title("Match Outcomes by Season — Is Home Advantage Stable?", pad=14)
ax.legend()
ax.set_ylim(0, 60)
plt.tight_layout()
plt.savefig("chart2_outcomes_by_season.png")
plt.close()
print("Saved chart2_outcomes_by_season.png")

# ── Chart 3: Shots on target vs result ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, col, label, title in [
    (axes[0], "HST", "Home Shots on Target", "Home Shots on Target by Result"),
    (axes[1], "AST", "Away Shots on Target", "Away Shots on Target by Result"),
]:
    data = [df[df["FTR"] == r][col].dropna() for r in ["H", "D", "A"]]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    for patch, colour in zip(bp["boxes"], [COLOURS["H"], COLOURS["D"], COLOURS["A"]]):
        patch.set_facecolor(colour)
    ax.set_xticklabels(["Home Win", "Draw", "Away Win"])
    ax.set_ylabel(label)
    ax.set_title(title, pad=10)
plt.suptitle("Shots on Target Distribution by Match Result", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("chart3_shots_on_target.png")
plt.close()
print("Saved chart3_shots_on_target.png")

# ── Chart 4: Correlation heatmap ──────────────────────────────────────────────
df["Result_num"] = df["FTR"].map({"H": 1, "D": 0, "A": -1})
stat_cols = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
             "HY", "AY", "Result_num"]
rename = {"HS": "Home Shots", "AS": "Away Shots",
          "HST": "Home SOT", "AST": "Away SOT",
          "HF": "Home Fouls", "AF": "Away Fouls",
          "HC": "Home Corners", "AC": "Away Corners",
          "HY": "Home Yellows", "AY": "Away Yellows",
          "Result_num": "Result\n(H=1,A=-1)"}
corr = df[stat_cols].rename(columns=rename).corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Correlation Matrix: Match Statistics vs Result", pad=14, fontsize=14)
plt.tight_layout()
plt.savefig("chart4_correlation_heatmap.png")
plt.close()
print("Saved chart4_correlation_heatmap.png")

# ── Chart 5: Home advantage by team ──────────────────────────────────────────
team_home = df.groupby("HomeTeam")["FTR"].apply(
    lambda x: (x == "H").sum() / len(x) * 100).reset_index()
team_home.columns = ["Team", "HomeWin%"]
team_home = team_home.sort_values("HomeWin%", ascending=True)

fig, ax = plt.subplots(figsize=(9, 10))
colours = [COLOURS["H"] if v >= 50 else COLOURS["A"] for v in team_home["HomeWin%"]]
ax.barh(team_home["Team"], team_home["HomeWin%"], color=colours, edgecolor="white")
ax.axvline(45.8, color="grey", linestyle="--", linewidth=1.2,
           label=f"Overall avg (45.8%)")
ax.set_xlabel("Home Win Rate (%)")
ax.set_title("Home Win Rate by Team (2021–24)", pad=14)
ax.legend()
plt.tight_layout()
plt.savefig("chart5_home_win_by_team.png")
plt.close()
print("Saved chart5_home_win_by_team.png")

# ── Chart 6: Average goals — home vs away ────────────────────────────────────
avg_goals = df.groupby("Season")[["FTHG", "FTAG"]].mean()
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(avg_goals))
width = 0.35
ax.bar(x - width/2, avg_goals["FTHG"], width, label="Avg Home Goals",
       color=COLOURS["H"], edgecolor="white")
ax.bar(x + width/2, avg_goals["FTAG"], width, label="Avg Away Goals",
       color=COLOURS["A"], edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(avg_goals.index)
ax.set_ylabel("Average Goals per Match")
ax.set_title("Average Goals Scored — Home vs Away by Season", pad=14)
ax.legend()
plt.tight_layout()
plt.savefig("chart6_avg_goals.png")
plt.close()
print("Saved chart6_avg_goals.png")

print("\n" + "=" * 60)
print("ALL CHARTS SAVED. EDA COMPLETE.")
print("=" * 60)

# ── Print key stats for the report ───────────────────────────────────────────
print("\nKEY FINDINGS FOR REPORT:")
print(f"- Home win rate: {pct['H']:.1f}%  |  Draw: {pct['D']:.1f}%  |  Away win: {pct['A']:.1f}%")
print(f"- Avg home goals per match: {df['FTHG'].mean():.2f}")
print(f"- Avg away goals per match: {df['FTAG'].mean():.2f}")
print(f"- Avg home shots on target: {df['HST'].mean():.2f}")
print(f"- Avg away shots on target: {df['AST'].mean():.2f}")
corr_result = df[["HST","AST","HC","AC","HF","AF","Result_num"]].corr()["Result_num"].drop("Result_num")
print(f"\nCorrelations with result (H=1, A=-1):")
print(corr_result.sort_values(ascending=False).to_string())
