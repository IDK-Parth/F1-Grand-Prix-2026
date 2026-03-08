# ==========================================
# IMPORTS
# ==========================================
import os
import warnings
import numpy as np
import pandas as pd
import fastf1
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ==========================================
# FASTF1 CACHE
# ==========================================
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

session = fastf1.get_session(2026, "Australia", "Q")
session.load()

# ==========================================
# QUALIFYING DATA
# ==========================================
Q_2026 = pd.DataFrame({
"Driver":[
"RUS","ANT","HAD","LEC","PIA","NOR","HAM",
"LAW","LIN","BOR","HUL","BEA","OCO","GAS",
"ALB","COL","ALO","PER","BOT","VER","SAI","STR"
],

"QualifyingTime":[
78.518,78.811,79.303,79.327,79.380,79.475,79.478,
79.994,81.247,80.221,80.303,80.311,80.491,80.501,
80.941,81.270,81.969,82.605,83.244,82.500,83.000,85.000
],

"GridPosition":list(range(1,23))
})

# ==========================================
# TEAM PERFORMANCE
# ==========================================
team_points = {
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

max_pts = max(team_points.values())
team_perf = {t:p/max_pts for t,p in team_points.items()}

# ==========================================
# DRIVER TEAM MAP
# ==========================================
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
Q_2026["TeamPerformance"] = Q_2026["Team"].map(team_perf)

# ==========================================
# REGULATION BOOST
# ==========================================
reg_boost = {
"Mercedes":1.15,"Ferrari":1.05,"Red Bull":0.95,
"McLaren":1.00,"Williams":0.80,"Aston Martin":0.70,
"Haas":0.85,"Racing Bulls":0.88,"Audi":0.83,
"Alpine":0.80,"Cadillac":0.70
}

Q_2026["RegBoost"] = Q_2026["Team"].map(reg_boost)

Q_2026["AdjustedTeamScore"] = (
Q_2026["TeamPerformance"] * Q_2026["RegBoost"]
)

# ==========================================
# WEATHER
# ==========================================
rain_probability = 0.15
temperature = 22

Q_2026["RainProbability"] = rain_probability
Q_2026["Temperature"] = temperature

# ==========================================
# RACE SIMULATION FEATURES
# ==========================================
Q_2026["GridPenalty"] = (Q_2026["GridPosition"]-1)*0.15

pole = Q_2026["QualifyingTime"].min()

Q_2026["GapFromPole"] = Q_2026["QualifyingTime"] - pole

# ==========================================
# SYNTHETIC RACE TIME TARGET
# ==========================================
fuel_effect = np.random.normal(2.5,0.3,len(Q_2026))
driver_variance = np.random.normal(0,0.15,len(Q_2026))
weather_effect = rain_probability * 1.2

Q_2026["TargetRaceTime"] = (
Q_2026["QualifyingTime"]
+ fuel_effect
+ driver_variance
+ weather_effect
)

# ==========================================
# FEATURES
# ==========================================
features = [
"QualifyingTime",
"GapFromPole",
"AdjustedTeamScore",
"GridPenalty",
"RainProbability",
"Temperature"
]

X = Q_2026[features]
y = Q_2026["TargetRaceTime"]

# ==========================================
# HANDLE MISSING
# ==========================================
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# ==========================================
# SCALING
# ==========================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================================
# TRAIN TEST
# ==========================================
X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42
)

# ==========================================
# MODELS
# ==========================================
models = {

"XGBoost":XGBRegressor(
n_estimators=200,
learning_rate=0.05,
max_depth=4
),

"RandomForest":RandomForestRegressor(
n_estimators=200,
max_depth=6
),

"GradientBoost":GradientBoostingRegressor(),

"LinearRegression":LinearRegression()

}

# ==========================================
# TRAIN MODELS
# ==========================================
results = {}

for name,model in models.items():

    model.fit(X_train,y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test,preds)

    results[name] = mae

    print(name,"MAE:",round(mae,3))

# ==========================================
# BEST MODEL
# ==========================================
best_name = min(results,key=results.get)
model = models[best_name]

print("\nBest Model:",best_name)

# ==========================================
# PREDICT RACE TIME
# ==========================================
Q_2026["PredictedRaceTime"] = model.predict(X)

final_results = Q_2026.sort_values(
by="PredictedRaceTime"
).reset_index(drop=True)

final_results["PredictedPosition"] = final_results.index+1

# ==========================================
# PODIUM
# ==========================================
podium = final_results.head(3)

print("\n==============================")
print("PREDICTED PODIUM")
print("==============================")

for i,row in podium.iterrows():
    print(f"P{i+1}: {row.Driver} ({row.Team})")

# ==========================================
# MONTE CARLO SIMULATION
# ==========================================
simulations = 1000

win_count = {d:0 for d in Q_2026["Driver"]}

for i in range(simulations):

    noise = np.random.normal(0,0.2,len(final_results))

    sim_time = final_results["PredictedRaceTime"] + noise

    winner = final_results.iloc[np.argmin(sim_time)]["Driver"]

    win_count[winner] += 1

print("\nWIN PROBABILITY")

for d in win_count:

    prob = win_count[d]/simulations*100

    if prob > 0:
        print(d,":",round(prob,2),"%")

# ==========================================
# VISUALIZATION
# ==========================================
plt.figure(figsize=(10,6))

plt.barh(
final_results["Driver"],
final_results["PredictedRaceTime"]
)

plt.title("Predicted Race Time")
plt.xlabel("Time (s)")
plt.show()

# ==========================================
# CREATE FINISHING ORDER
# ==========================================

final_results = Q_2026.sort_values(
    by="PredictedRaceTime"
).reset_index(drop=True)

final_results["PredictedPosition"] = final_results.index + 1

# ==========================================
# PODIUM
# ==========================================

podium = final_results.head(3)

p1 = podium.iloc[0]
p2 = podium.iloc[1]
p3 = podium.iloc[2]

print("\n==============================")
print("PREDICTED PODIUM - AUSTRALIA GP")
print("==============================")

print(f"1st : {p1.Driver} ({p1.Team})")
print(f"2nd : {p2.Driver} ({p2.Team})")
print(f"3rd : {p3.Driver} ({p3.Team})")

