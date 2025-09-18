# orchestrator2.py — PATCH the imports at the top
import os
import pandas as pd
import numpy as np
import pickle
import csv

from config import (
    BASE_DIR, MERGED_GW_PATH,
    ICT_MODEL_PATH, VALUATION_MODEL_PATH, OPPONENT_MODEL_PATH,
    ERROR_METRICS_PATH, HISTORY_PATH
)

# REMOVE any imports of nonexistent functions from data_preprocessing
# from data_preprocessing import fetch_current_gw_data, process_gw_data   # <- delete this

# keep fetch_data import if you want to git pull on run (it runs on import)
import fetch_data

from feature_engineering import create_features_for_current_gw
from decision_framework import make_decisions
# from user_team import get_user_team_for_gw   # if/when you actually have this

def load_models():
    with open(ICT_MODEL_PATH, 'rb') as f:
        ict_model = pickle.load(f)
    with open(VALUATION_MODEL_PATH, 'rb') as f:
        valuation_model = pickle.load(f)
    with open(OPPONENT_MODEL_PATH, 'rb') as f:
        opponent_model = pickle.load(f)
    return ict_model, valuation_model, opponent_model

def _current_gw_from(df: pd.DataFrame) -> int:
    if "gw" in df.columns:
        return int(df["gw"].max())
    if "GW" in df.columns:
        return int(df["GW"].max())
    if "round" in df.columns:
        return int(df["round"].max())
    if "event" in df.columns:
        return int(df["event"].max())
    raise KeyError("No GW column found in merged gw data.")

def run_fpl_pipeline():
    print("🔄 Syncing Vaastav repo (via fetch_data.py)...")

    print("📄 Loading merged gameweek data...")
    gw_data = pd.read_csv(MERGED_GW_PATH)
    current_gw = _current_gw_from(gw_data)
    print(f"📆 Current Gameweek: {current_gw}")

    print("🧠 Building features...")
    features = create_features_for_current_gw(current_gw=current_gw, save=True)

    print("📦 Loading models...")
    ict_model, valuation_model, opponent_model = load_models()

    print("📊 Running predictions...")
    # ✅ Inserted block here
    features_ict = ["influence", "creativity", "threat", "ict_index", "bps_rolling_avg"]
    features_valuation = [
        "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "bonus", "bps", "influence",
        "creativity", "threat", "ict_index",
        "yellow_cards", "red_cards", "bps_rolling_avg", "bonus_cumulative"
    ]
    features_opponent = ["opponent_difficulty"]
    # ICT features requested
    features_ict = ["influence", "creativity", "threat", "ict_index", "bps_rolling_avg"]

    # Keep only columns that exist AND that the model was trained on
    features_ict = [f for f in features_ict if f in ict_model.feature_names_in_]

    features["ict_prediction"] = ict_model.predict(features[features_ict].fillna(0))

    features_valuation = [f for f in features_valuation if f in features.columns]
    features_opponent = [f for f in features_opponent if f in features.columns]

    features["ict_prediction"] = ict_model.predict(features[features_ict].fillna(0))
    features["valuation_prediction"] = valuation_model.predict(features[features_valuation].fillna(0))
    features["opponent_prediction"] = opponent_model.predict(features[features_opponent].fillna(0))

    team_status = {
        "free_hit_available": True,
        "bench_boost_available": True,
        "triple_captain_available": True
    }

    print("🤖 Making strategic decisions...")
    decisions = make_decisions(features, current_gw, team_status)

    print("🎯 Decisions:")
    for category in ["trade_ins", "trade_outs"]:
        print(f"{category.upper()}:")
        for p in decisions[category]:
            print(f"  - {p['name']} (ID: {p['id']})")
    print(f"CAPTAIN: {decisions['captain']['name']} (ID: {decisions['captain']['id']})")
    print(f"VICE:    {decisions['vice_captain']['name']} (ID: {decisions['vice_captain']['id']})")
    print(f"CHIP:    {decisions['chip']}")

    print("📈 Evaluating prediction accuracy (GW-only)...")
    # Join on 'name'; adjust if your identifiers differ
    predicted_df = features[["name", "ict_prediction", "valuation_prediction", "opponent_prediction"]].copy()
    # normalize gw column in gw_data
    if "gw" not in gw_data.columns:
        for c in ["GW", "round", "event"]:
            if c in gw_data.columns:
                gw_data["gw"] = gw_data[c]
                break
    actual_df = gw_data[gw_data["gw"] == current_gw]

    merged = pd.merge(predicted_df, actual_df, on="name", how="inner")

    for model_name in ["ict", "valuation", "opponent"]:
        mae = np.mean(np.abs(merged[f"{model_name}_prediction"] - merged["total_points"]))
        rmse = np.sqrt(np.mean((merged[f"{model_name}_prediction"] - merged["total_points"])**2))
        print(f"{model_name.upper()} → MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    #print("📝 Logging to history.csv and error_metrics.csv...")
    #log_recommendations(current_gw, decisions)
    #log_error_metrics(current_gw, merged)

    print("✅ Pipeline completed successfully.")
    print("🎯 Decisions:")
    print(f"CAPTAIN: {decisions['captain']['name']} "
        f"(proj gap vs vice: {decisions['captain']['confidence_gap']:.2f} pts)")
    print(f"VICE:    {decisions['vice_captain']['name']}")
    print(f"CHIP:    {decisions['chip']}")

    if decisions["trade_ins"]:
        print("🔄 Suggested Trades:")
        for out, inn in zip(decisions["trade_outs"], decisions["trade_ins"]):
            gain = inn["projected_points"] - out["projected_points"]
            print(f"  - {out['name']} ➝ {inn['name']} (+{gain:.1f} pts over next 5 GWs)")
    else:
        print("🔄 No trades suggested.")

    print(f"💰 Bank before: {decisions['bank_before']:.1f}m | after: {decisions['bank_after']:.1f}m")
    print(f"🎟️ Free transfers before: {decisions['free_transfers_before']} | after: {decisions['free_transfers_after']}")
    print(f"📊 Squad Value: {decisions['squad_value']:.1f}m")
    print(f"📈 Projected gain from trades: {decisions['trade_gain']:.1f} pts")

    if decisions["injury_flags"]:
        print("🩺 Injuries/flags:", decisions["injury_flags"])

    if decisions["fixture_info"]:
        print("📅 Fixture Horizon (next 5 GWs):")
        print("   Easier runs:", ", ".join(decisions["fixture_info"]["easiest_runs"]))
        print("   Harder runs:", ", ".join(decisions["fixture_info"]["hardest_runs"]))

    log_decisions(current_gw, decisions)
    print(f"📝 Decisions for GW{current_gw} logged to history_decisions.csv")
    
    if decisions["injury_flags"]["count"] > 0:
        print(f"🩺 Injured players: {', '.join(decisions['injury_flags']['players'])}")


# ---- Logger defined below ----
import csv
from pathlib import Path

HISTORY_LOG = Path(BASE_DIR) / "history_decisions.csv"

def log_decisions(current_gw, decisions):
    file_exists = HISTORY_LOG.exists()
    with open(HISTORY_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "gw", "captain", "vice_captain", "chip",
            "trade_outs", "trade_ins",
            "bank_before", "bank_after",
            "free_transfers_before", "free_transfers_after",
            "trade_gain", "squad_value"
        ])
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "gw": current_gw,
            "captain": decisions["captain"]["name"],
            "vice_captain": decisions["vice_captain"]["name"],
            "chip": decisions["chip"],
            "trade_outs": ", ".join([p["name"] for p in decisions["trade_outs"]]) if decisions["trade_outs"] else "",
            "trade_ins": ", ".join([p["name"] for p in decisions["trade_ins"]]) if decisions["trade_ins"] else "",
            "bank_before": decisions["bank_before"],
            "bank_after": decisions["bank_after"],
            "free_transfers_before": decisions["free_transfers_before"],
            "free_transfers_after": decisions["free_transfers_after"],
            "trade_gain": decisions["trade_gain"],
            "squad_value": decisions["squad_value"],
        })


if __name__ == "__main__":
    run_fpl_pipeline()