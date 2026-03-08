# ==============================
# Imports
# ==============================
import os
import warnings
import pandas as pd
import fastf1
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ==============================
# FastF1 Cache
# ==============================
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

# Q = Qualifying session
session = fastf1.get_session(2026, 'Australia', 'Q')
session.load()

# ==============================
# Qualifying Dataset
# ==============================
Q_2026 = pd.DataFrame({
    "Driver": [
        "RUS","ANT","HAD","LEC","PIA","NOR","HAM",
        "LAW","LIN","BOR","HUL","BEA","OCO","GAS",
        "ALB","COL","ALO","PER","BOT","VER","SAI","STR"
    ],

    "QualifyingTime (s)": [
        78.518,78.811,79.303,79.327,79.380,79.475,79.478,
        79.994,81.247,80.221,80.303,80.311,80.491,80.501,
        80.941,81.270,81.969,82.605,83.244,82.500,83.000,85.000
    ],

    "GridPosition": list(range(1,23))
})

# ==============================
# Team Performance
# ==============================
team_points_2025 = {
    "McLaren":800,
    "Mercedes":459,
    "Red Bull":429,
    "Ferrari":382,
    "Williams":137,
    "Aston Martin":80,
    "Haas":73,
    "Racing Bulls":92,
    "Audi":68,
    "Alpine":22,
    "Cadillac":5
}

max_pts = max(team_points_2025.values())
team_performance_score = {t:p/max_pts for t,p in team_points_2025.items()}

# ==============================
# Driver-Team Mapping
# ==============================
driver_team_map = {
"NOR":"McLaren","PIA":"McLaren",
"HAM":"Ferrari","LEC":"Ferrari",
"VER":"Red Bull","LAW":"Red Bull",
"RUS":"Mercedes","ANT":"Mercedes",
"ALB":"Williams","COL":"Williams",
"ALO":"Aston Martin","STR":"Aston Martin",
"OCO":"Haas","BEA":"Haas",
"HAD":"Racing Bulls","LIN":"Racing Bulls",
"HUL":"Audi","BOT":"Audi",
"GAS":"Alpine","SAI":"Alpine",
"PER":"Cadillac","BOR":"Cadillac"
}

Q_2026["Team"] = Q_2026["Driver"].map(driver_team_map)
Q_2026["TeamPerformance"] = Q_2026["Team"].map(team_performance_score)

# ==============================
# Regulation Boost
# ==============================
reg_change_boost = {
"Mercedes":1.15,"Ferrari":1.05,"Red Bull":0.95,
"McLaren":1.00,"Williams":0.80,"Aston Martin":0.70,
"Haas":0.85,"Racing Bulls":0.88,"Audi":0.83,
"Alpine":0.80,"Cadillac":0.70
}

Q_2026["RegChangeBoost"] = Q_2026["Team"].map(reg_change_boost)

Q_2026["AdjustedTeamScore"] = (
    Q_2026["TeamPerformance"] * Q_2026["RegChangeBoost"]
)

# ==============================
# Weather Factors
# ==============================
rain_probability = 0.15
temperature = 22.0

Q_2026["RainProbability"] = rain_probability
Q_2026["Temperature"] = temperature

# ==============================
# Race Simulation Features
# ==============================
Q_2026["GridPenalty (s)"] = (Q_2026["GridPosition"] - 1) * 0.15

pole_time = Q_2026["QualifyingTime (s)"].min()
Q_2026["GapFromPole (s)"] = Q_2026["QualifyingTime (s)"] - pole_time

# ==============================
# Target Variable
# ==============================
qualifying_to_race_factor = 1.055
merged = Q_2026.copy()

merged["TargetTime (s)"] = (
    merged["QualifyingTime (s)"] * qualifying_to_race_factor
)

# ==============================
# Features
# ==============================
feature_cols = [
"QualifyingTime (s)",
"GapFromPole (s)",
"AdjustedTeamScore",
"GridPenalty (s)",
"RainProbability",
"Temperature"
]

X = merged[feature_cols]
y = merged["TargetTime (s)"]

# ==============================
# Handle Missing Values
# ==============================
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# ==============================
# Scale Important Features
# ==============================
qualifying_scale = 3.0
X_scaled = X_imputed.copy()

X_scaled[:,0] *= qualifying_scale
X_scaled[:,1] *= qualifying_scale

# ==============================
# Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.2, random_state=42
)

# ==============================
# Model
# ==============================
model = XGBRegressor(
n_estimators=100,
learning_rate=0.05,
max_depth=4,
random_state=42,
subsample=0.8,
colsample_bytree=0.8
)

model.fit(X_train, y_train)

# ==============================
# Predictions
# ==============================
merged["PredictedRaceTime (s)"] = model.predict(X_scaled)

final_results = merged.sort_values(
by=["PredictedRaceTime (s)","GridPosition"]
).reset_index(drop=True)

# ==============================
# Podium Prediction
# ==============================
podium = final_results.head(3)

print("\n"+"="*65)
print("PREDICTED PODIUM - 2026 AUSTRALIAN GP")
print("="*65)

print(f"P1: {podium.iloc[0]['Driver']} ({podium.iloc[0]['Team']})")
print(f"P2: {podium.iloc[1]['Driver']} ({podium.iloc[1]['Team']})")
print(f"P3: {podium.iloc[2]['Driver']} ({podium.iloc[2]['Team']})")

# ==============================
# Model Evaluation
# ==============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print("\nModel MAE:", round(mae,3),"seconds")

# ==============================
# Visualization
# ==============================
fig, axes = plt.subplots(1,2,figsize=(16,6))

fig.suptitle("2026 Australian GP Prediction")

# Feature Importance
importance = model.feature_importances_

feature_labels = [
"QualifyingTime",
"GapFromPole",
"AdjustedTeamScore",
"GridPenalty",
"RainProbability",
"Temperature"
]

axes[0].barh(feature_labels, importance)
axes[0].set_title("Feature Importance")

# Predicted Race Times
axes[1].barh(final_results["Driver"], final_results["PredictedRaceTime (s)"])
axes[1].set_title("Predicted Race Time")

plt.tight_layout()
plt.show()