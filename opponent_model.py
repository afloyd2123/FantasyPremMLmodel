# opponent_model.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from config import BASE_DIR, DATA_DIR, OPPONENT_MODEL_PATH, MERGED_GW_PATH

FEATURES_CSV = os.path.join(BASE_DIR, "feature_engineering.csv")
PREDICT_OUT  = os.path.join(BASE_DIR, "opponent_predictions.csv")

def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _canon_gw(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("gw", "GW", "round", "event"):
        if c in df.columns:
            if c != "gw":
                df["gw"] = df[c]
            df["gw"] = pd.to_numeric(df["gw"], errors="coerce")
            return df
    raise KeyError("No gameweek column found.")

def _ensure_opponent_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """If opponent_difficulty is missing, rebuild it from fixtures.csv and merged_gw.csv."""
    if "opponent_difficulty" in df.columns:
        return df

    fixtures_path = os.path.join(DATA_DIR, "fixtures.csv")
    if not os.path.exists(fixtures_path) or not os.path.exists(MERGED_GW_PATH):
        raise FileNotFoundError("Missing fixtures.csv or merged_gw.csv to rebuild opponent_difficulty.")

    fixtures = pd.read_csv(fixtures_path)
    gws      = pd.read_csv(MERGED_GW_PATH)

    fixtures = _canon_gw(fixtures)
    gws      = _canon_gw(gws)

    need = {"gw", "opponent_team", "was_home"}
    if not need.issubset(gws.columns):
        raise KeyError(f"merged_gw.csv missing required columns: {list(need - set(gws.columns))}")

    gws["opponent_team"] = pd.to_numeric(gws["opponent_team"], errors="coerce").astype("Int64")
    fixtures["team_h"]   = pd.to_numeric(fixtures["team_h"],   errors="coerce").astype("Int64")
    fixtures["team_a"]   = pd.to_numeric(fixtures["team_a"],   errors="coerce").astype("Int64")

    gws = gws.merge(
        fixtures[["gw", "team_h", "team_h_difficulty"]],
        left_on=["gw", "opponent_team"],
        right_on=["gw", "team_h"],
        how="left"
    ).rename(columns={"team_h_difficulty": "opp_home_difficulty"}).drop(columns=["team_h"])

    gws = gws.merge(
        fixtures[["gw", "team_a", "team_a_difficulty"]],
        left_on=["gw", "opponent_team"],
        right_on=["gw", "team_a"],
        how="left"
    ).rename(columns={"team_a_difficulty": "opp_away_difficulty"}).drop(columns=["team_a"])

    gws["opponent_difficulty"] = np.where(
        gws["was_home"].astype(bool),
        gws["opp_away_difficulty"],
        gws["opp_home_difficulty"]
    )

    # Merge the new column back onto df by (name, gw) if both exist; else by (gw, team/opponent_team) fallback
    if {"name", "gw"}.issubset(df.columns) and {"name", "gw"}.issubset(gws.columns):
        df = df.merge(gws[["name", "gw", "opponent_difficulty"]].drop_duplicates(), on=["name", "gw"], how="left")
    elif {"gw", "opponent_team"}.issubset(df.columns) and {"gw", "opponent_team"}.issubset(gws.columns):
        df = df.merge(gws[["gw", "opponent_team", "opponent_difficulty"]].drop_duplicates(), on=["gw", "opponent_team"], how="left")
    else:
        # last resort: merge on gw only (coarse)
        df = df.merge(gws[["gw", "opponent_difficulty"]].drop_duplicates(), on="gw", how="left")

    return df

def main():
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Feature file not found: {FEATURES_CSV}")
    data = pd.read_csv(FEATURES_CSV)

    # Normalize GW
    data = _canon_gw(data)

    # Make sure opponent_difficulty exists (rebuild if necessary)
    data = _ensure_opponent_difficulty(data)

    if "opponent_difficulty" not in data.columns:
        raise KeyError("opponent_difficulty is still missing after rebuild attempt.")

    features = ["opponent_difficulty"]
    target   = "total_points"
    if target not in data.columns:
        raise KeyError(f"Missing target column '{target}' in features")

    train_df = data.dropna(subset=features + [target])
    if train_df.empty:
        raise ValueError("No valid rows to train opponent model")

    X = train_df[features]
    y = train_df[target].astype(float)

    n = len(train_df)
    if n < 3:
        X_train, y_train = X, y
        X_val,   y_val   = X, y
        eval_on_train = True
    else:
        test_size = 0.2 if n >= 10 else max(1/n, 0.1)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
        eval_on_train = False

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = _rmse(y_val, y_pred)
    where = "train set" if eval_on_train else "validation split"
    print(f"RMSE for Opponent model on {where}: {rmse:.4f}")

    # Predict everywhere
    data["predicted_points_opponent"] = model.predict(data[features].fillna(0))
    out_cols = [c for c in ["name", "gw", "predicted_points_opponent"] if c in data.columns]
    data[out_cols].to_csv(PREDICT_OUT, index=False)
    print(f"Saved opponent predictions → {PREDICT_OUT}")

    with open(OPPONENT_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved opponent model → {OPPONENT_MODEL_PATH}")

if __name__ == "__main__":
    main()
