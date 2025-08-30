import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

from config import BASE_DIR, ICT_MODEL_PATH

# Load preprocessed feature data
feature_path = os.path.join(BASE_DIR, "feature_engineering.csv")
players = pd.read_csv(feature_path)



# Prepare ICT data
features_ict = ['influence', 'creativity', 'threat', 'ict_index', 'bps_rolling_avg']
ict_data = players[['name', 'GW', features_ict, 'total_points']].copy()

# Convert ICT fields to numeric
ict_fields = ['influence', 'creativity', 'threat', 'ict_index']
ict_data[ict_fields] = ict_data[ict_fields].apply(pd.to_numeric, errors='coerce')

# Shift total_points to get next gameweek's target
ict_data['next_gw_points'] = ict_data.groupby('name')['total_points'].shift(-1)
ict_data.fillna(0, inplace=True)

# Features and target
X = ict_data[ict_fields]
y = ict_data['next_gw_points']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print("RMSE for ICT model:", rmse)

# Predict on all rows for export
ict_data['predicted_points'] = model.predict(X)

# Save predictions
prediction_path = os.path.join(BASE_DIR, "ict_predictions.csv")
ict_data[['name', 'GW', 'predicted_points']].to_csv(prediction_path, index=False)

# Save model
with open(ICT_MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
