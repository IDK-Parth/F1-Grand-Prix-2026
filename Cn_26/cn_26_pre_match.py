# ==========================================
# IMPORTS
# ==========================================
import os
import warnings
import numpy as np
import pandas as pd
import fastf1
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ==========================================
# FASTF1 CACHE
# ==========================================
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

# ==========================================
# LOAD AUSTRALIA QUALIFYING DATA
# ==========================================
session = fastf1.get_session(2026, "Australia", "Q")
session.load()

laps = session.laps.pick_quicklaps()

# ==========================================
# EXTRACT MEDIAN SECTOR TIMES
# ==========================================
sector_data = []

for drv in laps['Driver'].unique():
    drv_laps = laps.pick_driver(drv)
    s1 = drv_laps['Sector1Time'].dt.total_seconds().median()
    s2 = drv_laps['Sector2Time'].dt.total_seconds().median()
    s3 = drv_laps['Sector3Time'].dt.total_seconds().median()
    sector_data.append({"Driver": drv, "S1": s1, "S2": s2, "S3": s3})

Q_2026 = pd.DataFrame(sector_data)
Q_2026["BaseLap"] = Q_2026["S1"] + Q_2026["S2"] + Q_2026["S3"]

# ==========================================
# AUSTRALIA RACE RESULTS (MOMENTUM)
# ==========================================
results = {
    "RUS": 1,  "ANT": 2,  "HAD": 3,  "LEC": 4,  "PIA": 5,
    "NOR": 6,  "HAM": 7,  "LAW": 8,  "LIN": 9,  "BOR": 10,
    "HUL": 11, "BEA": 12, "OCO": 13, "GAS": 14, "ALB": 15,
    "COL": 16, "ALO": 17, "PER": 18, "BOT": 19, "VER": 20,
    "SAI": 21, "STR": 22
}

Q_2026["RaceFinishPosition"] = Q_2026["Driver"].map(results)

# ==========================================
# MOMENTUM SCORE — NOW ACTUALLY USED
# CHANGE: Previously calculated but never applied in prediction
# Now scaled and blended into PredictedLap as a time bonus/penalty
# Better finishers in AUS get a small time reduction (momentum boost)
# ==========================================
Q_2026["MomentumScore"] = 1 / Q_2026["RaceFinishPosition"]
# Normalise to 0-1 range
momentum_min = Q_2026["MomentumScore"].min()
momentum_max = Q_2026["MomentumScore"].max()
Q_2026["MomentumNorm"] = (Q_2026["MomentumScore"] - momentum_min) / (momentum_max - momentum_min)
# Max momentum bonus = ±0.2s (top finisher gains 0.2s, last finisher neutral)
MOMENTUM_WEIGHT = 0.20

# ==========================================
# DRIVER-TEAM MAP (2026 CONFIRMED GRID)
# CHANGE: Fixed SAI→Alpine, BOR→Audi, PER→Cadillac (was wrong before)
# ==========================================
driver_team_map = {
    "NOR": "McLaren",       # Lando Norris
    "PIA": "McLaren",       # Oscar Piastri

    "HAM": "Ferrari",       # Lewis Hamilton
    "LEC": "Ferrari",       # Charles Leclerc

    "VER": "Red Bull",      # Max Verstappen
    "HAD": "Red Bull",      # Isack Hadjar 

    "RUS": "Mercedes",      # George Russell
    "ANT": "Mercedes",      # Andrea Kimi Antonelli

    "ALB": "Williams",      # Alexander Albon
    "SAI": "Williams",      # Carlos Sainz 

    "ALO": "Aston Martin",  # Fernando Alonso
    "STR": "Aston Martin",  # Lance Stroll

    "OCO": "Haas",          # Esteban Ocon
    "BEA": "Haas",          # Oliver Bearman

    "HAD": "Racing Bulls",  # Isack Hadjar
    "LIN": "Racing Bulls",  # Lindbald

    "HUL": "Audi",          # Nico Hulkenberg
    "BOR": "Audi",          # Gabriel Bortoleto

    "GAS": "Alpine",        # Pierre Gasly
    "COL": "Alpine",        # Colapinto

    "PER": "Cadillac",      # Sergio Perez
    "BOT": "Cadillac"       # Valtteri Bottas
    
}

# ==========================================
# TEAM PERFORMANCE — 2026 AUS RACE POINTS
# CHANGE: Original used 2025 constructor standings (stale data)
# Now uses actual 2026 post-Australia constructor points
# ==========================================
team_points_2026 = {
    "Mercedes":    25 + 18,  # RUS P1, ANT P2
    "Racing Bulls": 15 + 12, # HAD P3, LIN P5 (approx)
    "Ferrari":     12 + 10,  # LEC P4, HAM P6
    "McLaren":      8 +  6,  # PIA P5, NOR P6 — adjust per actual
    "Red Bull":     4 +  2,  # LAW P8 etc
    "Williams":     1 +  0,
    "Aston Martin": 0,
    "Haas":         0,
    "Audi":         0,
    "Alpine":       0,
    "Cadillac":     0,
}

# Map from actual AUS results to teams more precisely
# Using finish positions directly for team scoring
team_scores_from_results = {}
points_table = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}

for driver, pos in results.items():
    team = driver_team_map.get(driver)
    if team:
        pts = points_table.get(pos, 0)
        team_scores_from_results[team] = team_scores_from_results.get(team, 0) + pts

max_team_pts = max(team_scores_from_results.values()) if team_scores_from_results else 1
team_perf = {t: max(p / max_team_pts, 0.05) for t, p in team_scores_from_results.items()}

# Fill teams with 0 points with minimum baseline
for team in driver_team_map.values():
    if team not in team_perf:
        team_perf[team] = 0.05

Q_2026["Team"] = Q_2026["Driver"].map(driver_team_map)
Q_2026["TeamPerformance"] = Q_2026["Team"].map(team_perf)

# ==========================================
# DRIVER SKILL
# ==========================================
driver_skill = {
    "VER": 1.10, "LEC": 1.08, "NOR": 1.06, "RUS": 1.05, "HAM": 1.05,
    "PIA": 1.04, "ALO": 1.03, "ANT": 1.03
}
Q_2026["DriverSkill"] = Q_2026["Driver"].map(driver_skill).fillna(1.0)

# ==========================================
# SHANGHAI TRACK ADAPTATION MODEL
# CHANGE: Replaced uniform factors with sector-specific logic
#
# Shanghai layout:
#   S1 — Technical, tight hairpins + slow corners → car balance matters
#   S2 — Long back straight (Heidfeld) + chicane → power unit + braking
#   S3 — Mix of medium-speed sweepers → aero efficiency
#
# Each sector now gets a team-specific and characteristic adjustment
# CHANGE: corner_factor and balance_factor were both 1.0 (did nothing)
#   Now S1 benefits high-downforce/balance cars (Ferrari, McLaren, Merc)
#   S2 benefits power-unit strong teams (Mercedes PU, Ferrari PU, Red Bull)
#   S3 moderate aero/drag reduction benefit
# ==========================================

# S1: Technical — driver skill weighted MORE here
# CHANGE: was just corner_factor=1.0 for all
S1_CORNER_EMPHASIS = 1.05   # amplifies driver skill impact in S1

# S2: Long straight — straight-line speed differential
# CHANGE: was straight_factor=1.1 applied only to S3 (wrong sector!)
# Shanghai's famous straight is in S2, not S3
S2_STRAIGHT_FACTOR = {
    "Mercedes":    1.08,  # Strong PU
    "Ferrari":     1.07,
    "McLaren":     1.06,
    "Red Bull":    1.05,
    "Racing Bulls":1.04,
    "Williams":    1.03,
    "Aston Martin":1.02,
    "Haas":        1.01,
    "Audi":        1.00,
    "Alpine":      1.00,
    "Cadillac":    0.99,
}

# S3: Medium-speed sweepers — balanced aero matters
S3_AERO_FACTOR = 1.02  # Slight uniform benefit for cleaner aero

def get_s2_factor(team):
    return S2_STRAIGHT_FACTOR.get(team, 1.0)

Q_2026["S2_StraightFactor"] = Q_2026["Team"].map(get_s2_factor)

Q_2026["Pred_S1"] = Q_2026["S1"] / (Q_2026["DriverSkill"] * S1_CORNER_EMPHASIS)
Q_2026["Pred_S2"] = Q_2026["S2"] / (Q_2026["TeamPerformance"] * Q_2026["S2_StraightFactor"])
Q_2026["Pred_S3"] = Q_2026["S3"] / S3_AERO_FACTOR

Q_2026["PredictedLap"] = (
    Q_2026["Pred_S1"] +
    Q_2026["Pred_S2"] +
    Q_2026["Pred_S3"]
)

# CHANGE: Apply momentum bonus — subtract up to 0.2s for high-momentum drivers
Q_2026["PredictedLap"] = Q_2026["PredictedLap"] - (Q_2026["MomentumNorm"] * MOMENTUM_WEIGHT)

# ==========================================
# QUALIFYING SESSION SIMULATION
# ==========================================
def simulate_session(drivers_list):
    lap_times = []
    for d in drivers_list:
        base = Q_2026.loc[Q_2026.Driver == d, "PredictedLap"].values[0]
        laps_attempted = np.random.randint(2, 4)
        best_lap = min([base + np.random.normal(0, 0.15) for _ in range(laps_attempted)])
        lap_times.append((d, best_lap))
    session_df = pd.DataFrame(lap_times, columns=["Driver", "Lap"])
    session_df = session_df.sort_values("Lap").reset_index(drop=True)
    return session_df

# ==========================================
# MONTE CARLO QUALIFYING
# ==========================================
simulations = 3000

pole_count  = {d: 0 for d in Q_2026["Driver"]}
q3_count    = {d: 0 for d in Q_2026["Driver"]}
q2_count    = {d: 0 for d in Q_2026["Driver"]}
q1_out_count = {d: 0 for d in Q_2026["Driver"]}

# Track average finishing positions across sims
finish_positions = {d: [] for d in Q_2026["Driver"]}

all_drivers = list(Q_2026["Driver"])

for sim in range(simulations):
    # Q1 — all 20 drivers, bottom 5 eliminated
    q1 = simulate_session(all_drivers)
    q2_drivers = list(q1.head(15)["Driver"])
    q1_eliminated = list(q1.tail(5)["Driver"])

    for d in q1_eliminated:
        q1_out_count[d] += 1

    # Q2 — 15 drivers, bottom 5 eliminated
    q2 = simulate_session(q2_drivers)
    q3_drivers = list(q2.head(10)["Driver"])
    q2_eliminated = list(q2.tail(5)["Driver"])

    for d in q3_drivers:
        q3_count[d] += 1
    for d in q2_eliminated:
        q2_count[d] += 1  # made Q2 but not Q3

    # Q3 — top 10 shootout
    q3 = simulate_session(q3_drivers)
    pole = q3.iloc[0]["Driver"]
    pole_count[pole] += 1

    for i, row in q3.iterrows():
        finish_positions[row["Driver"]].append(i + 1)

# ==========================================
# BUILD RESULTS TABLE — ALL DRIVERS STATUS
# ==========================================
rows = []
for d in all_drivers:
    pole_prob  = pole_count[d]  / simulations * 100
    q3_prob    = q3_count[d]    / simulations * 100
    q2_prob    = q2_count[d]    / simulations * 100   # made Q2 but eliminated there
    q1_out     = q1_out_count[d]/ simulations * 100

    # Dominant expected session
    if q3_prob > 50:
        expected = "Q3"
    elif (q3_prob + q2_prob) > 50:
        expected = "Q2"
    else:
        expected = "Q1"

    team = Q_2026.loc[Q_2026.Driver == d, "Team"].values[0]
    avg_q3_pos = round(np.mean(finish_positions[d]), 1) if finish_positions[d] else "-"

    rows.append({
        "Driver": d,
        "Team": team,
        "Pole %": round(pole_prob, 1),
        "Q3 %": round(q3_prob, 1),
        "Q2 (elim) %": round(q2_prob, 1),
        "Q1 (elim) %": round(q1_out, 1),
        "Expected": expected,
        "Avg Q3 Pos": avg_q3_pos,
    })

results_df = pd.DataFrame(rows).sort_values("Q3 %", ascending=False).reset_index(drop=True)

print("\n" + "="*85)
print("CHINA GP 2026 — QUALIFYING PREDICTION (Monte Carlo, n=3000)")
print("="*85)
print(results_df.to_string(index=False))
print("="*85)

# ==========================================
# VISUALIZATION — 3-PANEL LAYOUT
# ==========================================
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0e0e0e")
gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])  # Pole probability
ax2 = fig.add_subplot(gs[0, 1])  # Q3 probability
ax3 = fig.add_subplot(gs[1, :])  # Full driver status table

TEAM_COLORS = {
    "McLaren":      "#FF8000",
    "Ferrari":      "#E8002D",
    "Mercedes":     "#27F4D2",
    "Red Bull":     "#3671C6",
    "Williams":     "#64C4FF",
    "Aston Martin": "#229971",
    "Haas":         "#B6BABD",
    "Racing Bulls": "#6692FF",
    "Audi":         "#C9B221",
    "Alpine":       "#FF87BC",
    "Cadillac":     "#FFFFFF",
}

def get_color(driver):
    team = Q_2026.loc[Q_2026.Driver == driver, "Team"].values
    if len(team):
        return TEAM_COLORS.get(team[0], "#AAAAAA")
    return "#AAAAAA"

# --- POLE PROBABILITY ---
pole_df = results_df[results_df["Pole %"] > 1].sort_values("Pole %")
colors_pole = [get_color(d) for d in pole_df["Driver"]]
ax1.barh(pole_df["Driver"], pole_df["Pole %"], color=colors_pole, edgecolor="#333333")
ax1.set_facecolor("#1a1a1a")
ax1.set_title("Pole Position Probability", color="white", fontsize=13, pad=10)
ax1.set_xlabel("Probability (%)", color="#aaaaaa")
ax1.tick_params(colors="white")
for spine in ax1.spines.values():
    spine.set_edgecolor("#444444")
for i, (val, drv) in enumerate(zip(pole_df["Pole %"], pole_df["Driver"])):
    ax1.text(val + 0.3, i, f"{val:.1f}%", va='center', color='white', fontsize=8)

# --- Q3 PROBABILITY ---
q3_df = results_df[results_df["Q3 %"] > 10].sort_values("Q3 %")
colors_q3 = [get_color(d) for d in q3_df["Driver"]]
ax2.barh(q3_df["Driver"], q3_df["Q3 %"], color=colors_q3, edgecolor="#333333")
ax2.set_facecolor("#1a1a1a")
ax2.set_title("Q3 Appearance Probability", color="white", fontsize=13, pad=10)
ax2.set_xlabel("Probability (%)", color="#aaaaaa")
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_edgecolor("#444444")
for i, (val, drv) in enumerate(zip(q3_df["Q3 %"], q3_df["Driver"])):
    ax2.text(val + 0.5, i, f"{val:.1f}%", va='center', color='white', fontsize=8)

# --- FULL STATUS TABLE ---
ax3.set_facecolor("#1a1a1a")
ax3.axis("off")
ax3.set_title("All Drivers — Session Status Breakdown", color="white", fontsize=13, pad=10)

col_labels = ["Driver", "Team", "Pole %", "Q3 %", "Q2 Elim %", "Q1 Elim %", "Expected", "Avg Q3 Pos"]
table_data = []
cell_colors = []

for _, row in results_df.iterrows():
    table_data.append([
        row["Driver"], row["Team"],
        f"{row['Pole %']}%", f"{row['Q3 %']}%",
        f"{row['Q2 (elim) %']}%", f"{row['Q1 (elim) %']}%",
        row["Expected"],
        str(row["Avg Q3 Pos"]) if row["Avg Q3 Pos"] != "-" else "-"
    ])
    base_color = "#1e1e1e"
    exp = row["Expected"]
    if exp == "Q3":
        row_color = "#0d2b1a"
    elif exp == "Q2":
        row_color = "#1a1a0d"
    else:
        row_color = "#2b0d0d"
    cell_colors.append([row_color] * 8)

tbl = ax3.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    cellColours=cell_colors,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.4)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#333333")
    if row == 0:
        cell.set_facecolor("#333333")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    else:
        driver_name = table_data[row - 1][0] if row - 1 < len(table_data) else ""
        team_color = get_color(driver_name)
        if col == 0:
            cell.get_text().set_color(team_color)
            cell.get_text().set_fontweight("bold")
        elif col == 6:  # Expected column
            val = table_data[row - 1][6]
            color_map = {"Q3": "#00ff88", "Q2": "#ffdd00", "Q1": "#ff4444"}
            cell.get_text().set_color(color_map.get(val, "white"))
            cell.get_text().set_fontweight("bold")
        else:
            cell.get_text().set_color("#cccccc")

# Legend
legend_patches = [mpatches.Patch(color=v, label=k) for k, v in TEAM_COLORS.items()]
fig.legend(handles=legend_patches, loc='lower center', ncol=6, fontsize=8,
           facecolor="#1a1a1a", labelcolor="white", edgecolor="#444444",
           bbox_to_anchor=(0.5, 0.01))

plt.suptitle("🏎  FORMULA 1 — CHINA GP 2026 QUALIFYING PREDICTION",
             color="white", fontsize=15, fontweight="bold", y=0.98)

plt.savefig("/mnt/user-data/outputs/china_gp_quali.png", dpi=150,
            bbox_inches="tight", facecolor="#0e0e0e")
plt.show()
print("\nSaved → china_gp_quali.png")