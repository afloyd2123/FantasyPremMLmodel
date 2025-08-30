import os
import pandas as pd
import pickle

from config import (
    BASE_DIR,
    ICT_MODEL_PATH,
    VALUATION_MODEL_PATH,
    OPPONENT_MODEL_PATH
)

def get_predictions():
    # Load input features
    feature_path = os.path.join(BASE_DIR, "feature_engineering.csv")
    features = pd.read_csv(feature_path)

    # Load models
    with open(ICT_MODEL_PATH, 'rb') as f:
        ict_model = pickle.load(f)

    with open(VALUATION_MODEL_PATH, 'rb') as f:
        valuation_model = pickle.load(f)

    with open(OPPONENT_MODEL_PATH, 'rb') as f:
        opponent_model = pickle.load(f)

    # Predict
    features['ict_prediction'] = ict_model.predict(features)
    features['valuation_prediction'] = valuation_model.predict(features)
    features['opponent_prediction'] = opponent_model.predict(features)

    # Return prediction columns with player name
    return features[['name', 'ict_prediction', 'valuation_prediction', 'opponent_prediction']]
