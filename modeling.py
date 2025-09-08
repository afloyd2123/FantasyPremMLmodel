import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import joblib

from config import BASE_DIR, ICT_MODEL_PATH, VALUATION_MODEL_PATH, OPPONENT_MODEL_PATH


class PlayerPerformanceModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        """Train the Linear Regression model."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict player performance."""
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """Evaluate the model’s R^2 score."""
        return self.model.score(X_test, y_test)


def retrain_models():
    """Retrain ICT, Valuation, and Opponent models on all available historical data."""
    all_data = []
    data_dir = BASE_DIR / "data"
    for season in os.listdir(data_dir):
        gw_path = data_dir / season / "gws"
        if gw_path.exists():
            for f in gw_path.glob("gw*.csv"):
                df = pd.read_csv(f)
                df["season"] = season
                all_data.append(df)

    if not all_data:
        raise RuntimeError("No gameweek data found to retrain models!")

    df_all = pd.concat(all_data, ignore_index=True)

    # --- Feature sets ---
    features_ict = ["influence", "creativity", "threat", "ict_index", "bps"]
    features_val = ["minutes", "goals_scored", "assists", "clean_sheets", "bonus", "bps"]
    features_opp = ["opponent_difficulty"]

    y = df_all["total_points"]

    # --- Train ICT model ---
    X_ict = df_all[features_ict].fillna(0)
    ict_model = LinearRegression().fit(X_ict, y)
    joblib.dump(ict_model, ICT_MODEL_PATH)

    # --- Train Valuation model ---
    X_val = df_all[features_val].fillna(0)
    val_model = LinearRegression().fit(X_val, y)
    joblib.dump(val_model, VALUATION_MODEL_PATH)

    # --- Train Opponent model ---
    X_opp = df_all[features_opp].fillna(0)
    opp_model = LinearRegression().fit(X_opp, y)
    joblib.dump(opp_model, OPPONENT_MODEL_PATH)

    print("✅ Models retrained on historical data (all seasons).")
