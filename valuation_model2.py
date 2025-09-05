# valuation_model2.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np
from config import BASE_DIR, VALUATION_MODEL_PATH

# Load engineered features
feature_path = os.path.join(BASE_DIR, "feature_engineering.csv")
data = pd.read_csv(feature_path)

# Normalize GW column
gw_col = None
for c in ["gw", "GW", "round", "event"]:
    if c in data.columns:
        gw_col = c
        break
if gw_col is None:
    raise KeyError("No GW column found in feature_engineering.csv")

# Calculate target: valuation change per gameweek
if "value" not in data.columns:
    raise KeyError("'value' column missing from feature_engineering.csv")

data["valuation_change"] = (
    data.groupby("name")["value"].diff().fillna(0)
)

# Define candidate features
candidate_features = [
    "minutes", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "bonus", "bps", "influence",
    "creativity", "threat", "ict_index",
    "yellow_cards", "red_cards"
]

# Optional rolling/cumulative features
optional_features = ["bps_rolling_avg", "bonus_cumulative"]

# Keep only those that actually exist
features = [f for f in candidate_features + optional_features if f in data.columns]

if not features:
    raise ValueError("No valid features found in feature_engineering.csv")

X = data[features].fillna(0)
y = data["valuation_change"].fillna(0)

# Train/test split
if len(data) > 10:  # guard for small dataset early season
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = X, X, y, y

# Train model
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE for Valuation Model: {rmse:.4f}")

# Predict for full dataset
data["predicted_valuation_change"] = rf.predict(X)

# Save predictions
prediction_path = os.path.join(BASE_DIR, "valuation_predictions.csv")
data[["name", gw_col, "predicted_valuation_change"]].to_csv(
    prediction_path, index=False
)
print(f"Saved valuation predictions → {prediction_path}")

# Save trained model
with open(VALUATION_MODEL_PATH, "wb") as f:
    pickle.dump(rf, f)
print(f"Saved valuation model → {VALUATION_MODEL_PATH}")
