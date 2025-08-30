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
from data_preprocessing import fetch_current_gw_data, process_gw_data
from feature_engineering import create_features_for_current_gw
from decision_framework import make_decisions
from FantasyPremML.user_team0 import get_user_team_for_gw


def load_models():
    with open(ICT_MODEL_PATH, 'rb') as f:
        ict_model = pickle.load(f)
    with open(VALUATION_MODEL_PATH, 'rb') as f:
        valuation_model = pickle.load(f)
    with open(OPPONENT_MODEL_PATH, 'rb') as f:
        opponent_model = pickle.load(f)
    return ict_model, valuation_model, opponent_model


def run_fpl_pipeline():
    print("🔄 Fetching & processing latest data...")
    fetch_current_gw_data()
    process_gw_data()

    print("📄 Loading merged gameweek data...")
    gw_data = pd.read_csv(MERGED_GW_PATH)
    current_gw = gw_data['gw'].max()
    print(f"📆 Current Gameweek: {current_gw}")

    print("🧠 Running feature engineering...")
    features = create_features_for_current_gw(current_gw)

    print("📦 Loading models...")
    ict_model, valuation_model, opponent_model = load_models()

    print("📊 Running predictions...")
    features['ict_prediction'] = ict_model.predict(features)
    features['valuation_prediction'] = valuation_model.predict(features)
    features['opponent_prediction'] = opponent_model.predict(features)

    # Decision logic inputs
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


    print("📈 Evaluating prediction accuracy...")
    # Evaluate RMSE/MAE only for players in GW
    predicted_df = features[['name', 'ict_prediction', 'valuation_prediction', 'opponent_prediction']].copy()
    actual_df = gw_data[gw_data['gw'] == current_gw]

    merged = pd.merge(predicted_df, actual_df, on='name', how='inner')

    for model_name in ['ict', 'valuation', 'opponent']:
        mae = np.mean(np.abs(merged[f"{model_name}_prediction"] - merged['total_points']))
        rmse = np.sqrt(np.mean((merged[f"{model_name}_prediction"] - merged['total_points'])**2))
        print(f"{model_name.upper()} → MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    print("📝 Logging to history.csv and error_metrics.csv...")
    log_recommendations(current_gw, decisions)
    log_error_metrics(current_gw, merged)

    print("✅ Pipeline completed successfully.")


def log_recommendations(gw, decisions):
    def format_players(players):
        return "; ".join([f"{p['name']} (ID: {p['id']})" for p in players])

    row = [
        gw,
        format_players(decisions.get("trade_ins", [])),
        format_players(decisions.get("trade_outs", [])),
        f"{decisions['captain']['name']} (ID: {decisions['captain']['id']})",
        f"{decisions['vice_captain']['name']} (ID: {decisions['vice_captain']['id']})",
        decisions.get("chip", "None")
    ]

    with open(HISTORY_PATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)



def log_error_metrics(gw, merged_df):
    data = {
        "game_week": gw,
        "mae_ict": np.mean(np.abs(merged_df["ict_prediction"] - merged_df["total_points"])),
        "mae_valuation": np.mean(np.abs(merged_df["valuation_prediction"] - merged_df["total_points"])),
        "mae_opponent": np.mean(np.abs(merged_df["opponent_prediction"] - merged_df["total_points"])),
        "rmse_ict": np.sqrt(np.mean((merged_df["ict_prediction"] - merged_df["total_points"])**2)),
        "rmse_valuation": np.sqrt(np.mean((merged_df["valuation_prediction"] - merged_df["total_points"])**2)),
        "rmse_opponent": np.sqrt(np.mean((merged_df["opponent_prediction"] - merged_df["total_points"])**2)),
    }
    df = pd.DataFrame([data])
    if os.path.exists(ERROR_METRICS_PATH):
        df.to_csv(ERROR_METRICS_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(ERROR_METRICS_PATH, index=False)


if __name__ == "__main__":
    run_fpl_pipeline()
