import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

from config import BASE_DIR, OPPONENT_MODEL_PATH

# Load features
feature_path = os.path.join(BASE_DIR, "feature_engineering.csv")
data = pd.read_csv(feature_path)

# Compute opponent-related features
data['opposition_overall_strength'] = np.where(
    data['was_home'],
    data['strength_overall_away'],
    data['strength_overall_home']
)

data['opposition_attack_strength'] = np.where(
    data['was_home'],
    data['strength_attack_away'],
    data['strength_attack_home']
)

data['opposition_defence_strength'] = np.where(
    data['was_home'],
    data['strength_defence_away'],
    data['strength_defence_home']
)

# Define model inputs
features = [
    'opposition_overall_strength',
    'opposition_attack_strength',
    'opposition_defence_strength'
]

X = data[features]
y = data['total_points']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE for Opponent Model: {rmse:.4f}")

# Predict across all data
data['predicted_points_opponent'] = model.predict(X)

# Save predictions
prediction_path = os.path.join(BASE_DIR, "opponent_predictions.csv")
data[['name', 'GW', 'predicted_points_opponent']].to_csv(prediction_path, index=False)

# Save model
with open(OPPONENT_MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
