# ICT_model.py — robust training for early-season + older sklearn
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from config import BASE_DIR, ICT_MODEL_PATH

FEATURES_CSV = os.path.join(BASE_DIR, "feature_engineering.csv")
PREDICT_OUT  = os.path.join(BASE_DIR, "ict_predictions.csv")

def _rmse(y_true, y_pred):
    # compatibility with older sklearn (no "squared" kw)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _detect_gw_col(df: pd.DataFrame) -> str:
    for c in ("gw", "GW", "round", "event"):
        if c in df.columns:
            if c != "gw":
                df["gw"] = df[c]
            # make gw numeric if possible
            df["gw"] = pd.to_numeric(df["gw"], errors="coerce")
            return "gw"
    raise KeyError("No gameweek column found (expected one of: 'gw','GW','round','event').")

def main():
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Feature file not found at: {FEATURES_CSV}")

    players = pd.read_csv(FEATURES_CSV)

    # --- choose features (allow missing bps_rolling_avg early in season)
    raw_features = ['influence', 'creativity', 'threat', 'ict_index', 'bps_rolling_avg']
    available_features = [f for f in raw_features if f in players.columns]
    if len(available_features) == 0:
        raise ValueError(
            "None of the expected ICT features are present. "
            f"Expected any of: {raw_features}. Columns in file: {players.columns.tolist()}"
        )

    gw_col = _detect_gw_col(players)

    # subset the frame
    cols = ['name', gw_col, 'total_points', *available_features]
    missing = [c for c in cols if c not in players.columns]
    if missing:
        raise KeyError(f"Missing columns in feature file: {missing}")

    ict_data = players[cols].copy()

    # coerce numerics
    ict_data[available_features + ['total_points']] = ict_data[available_features + ['total_points']].apply(
        pd.to_numeric, errors='coerce'
    )

    # target = next GW total points
    ict_data['next_gw_points'] = ict_data.groupby('name')['total_points'].shift(-1)

    # early season: lots of NaNs on the last GW row for each player; drop where target is NaN
    train_df = ict_data.dropna(subset=['next_gw_points']).copy()

    # if everything was dropped (e.g., only GW1 exists), fallback to using current total_points
    if train_df.empty:
        # minimal fallback to let the pipeline proceed; not statistically ideal
        train_df = ict_data.dropna(subset=available_features + ['total_points']).copy()
        train_df['next_gw_points'] = train_df['total_points']

    # final NA clean
    train_df = train_df.dropna(subset=available_features + ['next_gw_points'])

    X = train_df[available_features]
    y = train_df['next_gw_points'].astype(float)

    # small-data guardrails
    n = len(train_df)
    if n < 3:
        # train on all, eval on train (best we can do with tiny data)
        X_train, y_train = X, y
        X_val, y_val = X, y
        eval_on_train = True
    else:
        # keep validation size reasonable for small n
        test_size = 0.2 if n >= 10 else max(1 / n, 0.1)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
        eval_on_train = False

    # model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_val)
    rmse = _rmse(y_val, y_pred)
    where = "train set" if eval_on_train else "validation split"
    print(f"RMSE for ICT model on {where}: {rmse:.4f}")

    # predict on all rows we can (use all rows with features)
    all_X = ict_data[available_features].copy()
    all_X = all_X.fillna(0)
    ict_data['predicted_points'] = model.predict(all_X)

    # save predictions
    out_cols = ['name', gw_col, 'predicted_points']
    ict_data[out_cols].to_csv(PREDICT_OUT, index=False)
    print(f"Saved ICT predictions → {PREDICT_OUT}")

    # save model
    with open(ICT_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved ICT model → {ICT_MODEL_PATH}")

if __name__ == "__main__":
    main()
