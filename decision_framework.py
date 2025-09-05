# decision_framework.py
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from config import MERGED_GW_PATH, USE_MANUAL_WEIGHTS, MANUAL_WEIGHTS

def get_optimal_weights(predictions_df: pd.DataFrame, current_gw: int):
    """Compute optimal ensemble weights by regressing predictions against actuals."""
    # Load actuals
    actuals_df = pd.read_csv(MERGED_GW_PATH)

    # Normalize gameweek column
    gw_col = None
    for c in ["gw", "GW", "round", "event"]:
        if c in actuals_df.columns:
            gw_col = c
            break
    if gw_col is None:
        raise KeyError("No gameweek column found in merged_gw.csv")

    # Filter actuals
    if "total_points" not in actuals_df.columns:
        raise KeyError("'total_points' not found in merged_gw.csv")
    actuals_df = actuals_df[[gw_col, "element", "total_points"]]
    actuals_df = actuals_df[actuals_df[gw_col] == current_gw]

    # Ensure predictions_df has an 'id' for merging
    if "id" not in predictions_df.columns:
        if "element" in predictions_df.columns:
            predictions_df = predictions_df.rename(columns={"element": "id"})
        elif "name" in predictions_df.columns:
            predictions_df["id"] = predictions_df["name"]
        else:
            raise KeyError("predictions_df has no 'id', 'element', or 'name' column to merge on.")

    # Merge predictions with actuals
    merged = pd.merge(
        predictions_df,
        actuals_df,
        left_on="id",
        right_on="element",
        how="inner"
    )

    # Normalize total_points column
    if "total_points_y" in merged.columns:
        merged = merged.rename(columns={"total_points_y": "total_points"})
    elif "total_points_x" in merged.columns:
        merged = merged.rename(columns={"total_points_x": "total_points"})

    if "total_points" not in merged.columns:
        raise KeyError(
            f"Merge failed to bring in total_points. Columns present: {merged.columns.tolist()}"
        )

    # Drop rows with missing values in required columns
    df = merged.dropna(
        subset=["ict_prediction", "valuation_prediction", "opponent_prediction", "total_points"]
    )

    if df.empty:
        print("⚠️ Not enough data to optimize weights, falling back to equal weights.")
        return (1/3, 1/3, 1/3)

    # Fit regression for weights
    X = df[["ict_prediction", "valuation_prediction", "opponent_prediction"]].values
    y = df["total_points"].values

    try:
        reg = LinearRegression(positive=True)
        reg.fit(X, y)
        w_ict, w_val, w_opp = reg.coef_
        total = w_ict + w_val + w_opp
        if total == 0:
            return (1/3, 1/3, 1/3)
        return (w_ict / total, w_val / total, w_opp / total)
    except Exception as e:
        print(f"⚠️ Weight optimization failed: {e}, falling back to equal weights.")
        return (1/3, 1/3, 1/3)


def make_decisions(features: pd.DataFrame, current_gw: int, team_status: dict):
    """Generate strategic FPL decisions based on model predictions and optimal weights."""
    predictions_df = features.copy()

    # 🔧 Ensure predictions_df has an 'id'
    if "id" not in predictions_df.columns:
        if "element" in predictions_df.columns:
            predictions_df = predictions_df.rename(columns={"element": "id"})
        elif "name" in predictions_df.columns:
            predictions_df["id"] = predictions_df["name"]

    if USE_MANUAL_WEIGHTS:
        w_ict = MANUAL_WEIGHTS["ict"]
        w_val = MANUAL_WEIGHTS["valuation"]
        w_opp = MANUAL_WEIGHTS["opponent"]
        print(f"ℹ️ Using manual weights: {MANUAL_WEIGHTS}")
    else:
        w_ict, w_val, w_opp = get_optimal_weights(predictions_df, current_gw)
        print(f"📊 Optimized weights → ICT={w_ict:.2f}, VAL={w_val:.2f}, OPP={w_opp:.2f}")

    # Ensemble score
    predictions_df["ensemble_score"] = (
        w_ict * predictions_df["ict_prediction"] +
        w_val * predictions_df["valuation_prediction"] +
        w_opp * predictions_df["opponent_prediction"]
    )

    # Pick captain and vice
    top_players = predictions_df.sort_values("ensemble_score", ascending=False)
    captain = top_players.iloc[0]
    vice_captain = top_players.iloc[1]

    # Simple transfers
    trade_ins = top_players.head(3)[["id", "name", "ensemble_score"]].to_dict("records")
    trade_outs = top_players.tail(3)[["id", "name", "ensemble_score"]].to_dict("records")

    # Chip logic
    chip = "None"
    if team_status.get("triple_captain_available") and current_gw in [2, 19, 34]:
        chip = "Triple Captain"
    elif team_status.get("bench_boost_available") and current_gw in [19, 34]:
        chip = "Bench Boost"

    return {
        "trade_ins": trade_ins,
        "trade_outs": trade_outs,
        "captain": {"id": captain["id"], "name": captain["name"]},
        "vice_captain": {"id": vice_captain["id"], "name": vice_captain["name"]},
        "chip": chip
    }

