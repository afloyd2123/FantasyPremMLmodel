# orchestrator2.py — CLEANED

import os
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BASE_DIR, MERGED_GW_PATH,
    ICT_MODEL_PATH, VALUATION_MODEL_PATH, OPPONENT_MODEL_PATH,
    ERROR_METRICS_PATH, HISTORY_PATH
)

# Pull vaastav data on import (if your fetch_data does that)
import fetch_data

from feature_engineering import create_features_for_current_gw
from decision_framework import make_decisions


# ---- Paths for logging ----
HISTORY_LOG = Path(BASE_DIR) / "history_decisions.csv"
LINEUP_HISTORY_PATH = Path(BASE_DIR) / "history_lineups.csv"


# ---- Logger ----
def log_decisions(current_gw, decisions):
    file_exists = HISTORY_LOG.exists()
    with open(HISTORY_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gw", "captain", "vice_captain", "chip",
                "trade_outs", "trade_ins",
                "bank_before", "bank_after",
                "free_transfers_before", "free_transfers_after",
                "trade_gain", "squad_value",
                "starting_xi_expected", "bench_expected", "xi_total_expected"
            ],
        )
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
            "starting_xi_expected": json.dumps(decisions["starting_xi"]),
            "bench_expected": json.dumps(decisions.get("bench", [])),
            "xi_total_expected": float(sum(
                p.get("expected_points", p.get("ensemble_score", 0.0))
                for p in decisions.get("starting_xi", [])
            )),

        })


def load_models():
    with open(ICT_MODEL_PATH, 'rb') as f:
        ict_model = pickle.load(f)
    with open(VALUATION_MODEL_PATH, 'rb') as f:
        valuation_model = pickle.load(f)
    with open(OPPONENT_MODEL_PATH, 'rb') as f:
        opponent_model = pickle.load(f)
    return ict_model, valuation_model, opponent_model


def _current_gw_from(df: pd.DataFrame) -> int:
    for col in ("gw", "GW", "round", "event"):
        if col in df.columns:
            return int(df[col].max())
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
    # Feature sets
    features_ict = ["influence", "creativity", "threat", "ict_index", "bps_rolling_avg"]
    features_valuation = [
        "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "bonus", "bps", "influence",
        "creativity", "threat", "ict_index",
        "yellow_cards", "red_cards", "bps_rolling_avg", "bonus_cumulative"
    ]
    features_opponent = ["opponent_difficulty"]

    # Keep only columns that exist AND that the model was trained on (for ICT)
    features_ict = [f for f in features_ict if f in getattr(ict_model, "feature_names_in_", [])]

    # Predictions
    features["ict_prediction"] = ict_model.predict(features[features_ict].fillna(0)) if features_ict else 0.0
    features_valuation = [f for f in features_valuation if f in features.columns]
    features_opponent = [f for f in features_opponent if f in features.columns]
    features["valuation_prediction"] = valuation_model.predict(features[features_valuation].fillna(0)) if features_valuation else 0.0
    features["opponent_prediction"] = opponent_model.predict(features[features_opponent].fillna(0)) if features_opponent else 0.0

    team_status = {
        "free_hit_available": True,
        "bench_boost_available": True,
        "triple_captain_available": True
    }

    print("🤖 Making strategic decisions...")
    decisions = make_decisions(features, current_gw, team_status)

    # ---- Summary (all prints INSIDE the function) ----
    print("🎯 Decisions:")
    print(f"CAPTAIN: {decisions['captain']['name']} "
          f"(proj gap vs vice: {decisions['captain']['confidence_gap']:.2f} pts)")
    print(f"VICE:    {decisions['vice_captain']['name']}")
    print(f"CHIP:    {decisions.get('chip', 'None')}")

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

    # 📋 Starting XI (expected points)
    print("📋 Starting XI (expected points):")
    for row in decisions["starting_xi"]:
        opp_diff = row.get("opponent_difficulty", None)
        opp_txt = str(int(opp_diff)) if opp_diff is not None and not pd.isna(opp_diff) else "?"
        xp = row.get("expected_points", row.get("ensemble_score", 0.0))
        print(f"  - {row['name']} ({row['position']})  oppDiff={opp_txt}  xp={xp:.2f}")
    total_xi = sum(row.get("expected_points", row.get("ensemble_score", 0.0)) for row in decisions["starting_xi"])
    print(f"Σ XI expected points: {total_xi:.2f}")

    bench_list = decisions.get("bench", [])
    if bench_list:
        total_bench = sum(row.get("expected_points", row.get("ensemble_score", 0.0)) for row in bench_list)
        print("\n🪑 Bench (expected):")
        for row in bench_list:
            xp = row.get("expected_points", row.get("ensemble_score", 0.0))
            print(f"  - {row['name']} ({row['position']}): {xp:.2f} pts")
        print(f"  Σ Bench total: {total_bench:.2f} pts")


    if decisions.get("injury_flags"):
        print("🩺 Injuries/flags:", decisions["injury_flags"])

    if decisions.get("fixture_info"):
        print("📅 Fixture Horizon (next 5 GWs):")
        print("   Easier runs:", ", ".join(decisions["fixture_info"].get("easiest_runs", [])))
        print("   Harder runs:", ", ".join(decisions["fixture_info"].get("hardest_runs", [])))

    # Persist lineup-level predictions for learning/calibration
    lineup_rows = []
    for row in decisions["starting_xi"]:
        lineup_rows.append({
            "gw": current_gw,
            "player_id": row["id"],
            "player_name": row["name"],
            "position": row.get("position"),
            "opponent_difficulty": row.get("opponent_difficulty"),
            "expected_points": row.get("expected_points", row.get("ensemble_score")),
        })
    file_exists = LINEUP_HISTORY_PATH.exists()
    pd.DataFrame(lineup_rows).to_csv(
        LINEUP_HISTORY_PATH, mode="a", index=False, header=not file_exists
    )
    
    # Log decisions summary CSV
    log_decisions(current_gw, decisions)
    print(f"📝 Decisions for GW{current_gw} logged to history_decisions.csv")

    # ---- Quick per-model error on current GW (diagnostic only) ----
    print("📈 Evaluating prediction accuracy (GW-only)...")
    # normalize gw column in gw_data
    if "gw" not in gw_data.columns:
        for c in ["GW", "round", "event"]:
            if c in gw_data.columns:
                gw_data = gw_data.rename(columns={c: "gw"})
                break
    actual_df = gw_data[gw_data["gw"] == current_gw]
    predicted_df = features[["name", "ict_prediction", "valuation_prediction", "opponent_prediction"]].copy()
    merged = pd.merge(predicted_df, actual_df[["name", "total_points"]], on="name", how="inner")

    for model_name in ["ict", "valuation", "opponent"]:
        pred_col = f"{model_name}_prediction"
        if pred_col in merged.columns:
            mae = float(np.mean(np.abs(merged[pred_col] - merged["total_points"])))
            rmse = float(np.sqrt(np.mean((merged[pred_col] - merged["total_points"]) ** 2)))
            print(f"{model_name.upper()} → MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    print("✅ Pipeline completed successfully.")
    return decisions


if __name__ == "__main__":
    run_fpl_pipeline()
