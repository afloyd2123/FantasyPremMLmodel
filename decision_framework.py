import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from config import ERROR_METRICS_PATH, WEIGHTS_LOG_PATH, MERGED_GW_PATH, USE_MANUAL_WEIGHTS, MANUAL_WEIGHTS

TRADE_THRESHOLD = 3

# =======================
# Optimal Weight Learning
# =======================

def get_optimal_weights(predictions_df, current_gw):
    if USE_MANUAL_WEIGHTS:
        print("⚙️ Using manual override weights from config.")
        return MANUAL_WEIGHTS["ict"], MANUAL_WEIGHTS["valuation"], MANUAL_WEIGHTS["opponent"]

    actuals_df = pd.read_csv(MERGED_GW_PATH)
    merged = pd.merge(predictions_df, actuals_df, left_on='id', right_on='element')

    df = merged.dropna(subset=[
        "ict_prediction", "valuation_prediction", "opponent_prediction", "total_points"
    ])

    if df.empty:
        print("⚠️ Not enough data to optimize weights. Using equal weights.")
        return 1/3, 1/3, 1/3

    X = df[["ict_prediction", "valuation_prediction", "opponent_prediction"]]
    y = df["total_points"]

    model = LinearRegression()
    model.fit(X, y)
    weights = model.coef_

    log_weights(weights, current_gw)

    return weights[0], weights[1], weights[2]



def log_weights(weights, gw):
    row = {
        "gw": gw,
        "ict_weight": round(weights[0], 3),
        "valuation_weight": round(weights[1], 3),
        "opponent_weight": round(weights[2], 3)
    }

    df = pd.DataFrame([row])
    if os.path.exists(WEIGHTS_LOG_PATH):
        df.to_csv(WEIGHTS_LOG_PATH, mode='a', index=False, header=False)
    else:
        df.to_csv(WEIGHTS_LOG_PATH, index=False)


def compute_composite_score(df, w_ict, w_val, w_opp):
    return (
        w_ict * df["ict_prediction"] +
        w_val * df["valuation_prediction"] +
        w_opp * df["opponent_prediction"]
    )


# ===================
# Decision Logic Core
# ===================

def make_decisions(predictions_df, current_gw, team_status):
    decisions = {}

    # Optimize and compute composite score
    w_ict, w_val, w_opp = get_optimal_weights(predictions_df, current_gw)
    predictions_df["composite_score"] = compute_composite_score(predictions_df, w_ict, w_val, w_opp)

    # Trade decisions
    decisions['trade_ins'] = predictions_df.nlargest(5, 'composite_score')[['id', 'name']].to_dict('records')
    decisions['trade_outs'] = predictions_df.nsmallest(5, 'composite_score')[['id', 'name']].to_dict('records')

    # Captain decisions
    top_captains = predictions_df.nlargest(2, 'composite_score')[['id', 'name']]
    decisions['captain'] = top_captains.iloc[0].to_dict()
    decisions['vice_captain'] = top_captains.iloc[1].to_dict()

    # Chip logic
    high_scorers = predictions_df[predictions_df['ict_prediction'] > 6].shape[0]
    decisions['chip'] = 'Bench Boost' if high_scorers > 11 else 'None'

    if current_gw > 30:
        if team_status.get('bench_boost_available'):
            decisions['chip'] = 'Bench Boost'
        elif team_status.get('triple_captain_available'):
            decisions['chip'] = 'Triple Captain'

    return decisions


# ===================
# Trade Validation
# ===================

def is_valid_trade(user_team, player_in, player_out):
    temp_team = [p for p in user_team if p['id'] != player_out['id']]
    temp_team.append(player_in)

    gk = sum(1 for p in temp_team if p['position'] == 'GKP')
    df = sum(1 for p in temp_team if p['position'] == 'DEF')
    md = sum(1 for p in temp_team if p['position'] == 'MID')
    fw = sum(1 for p in temp_team if p['position'] == 'FWD')

    return gk == 2 and df == 5 and md == 5 and fw == 3


def is_within_club_limit(user_team, player_in, max_per_club=3):
    club_count = sum(1 for p in user_team if p['club'] == player_in['club'])
    return club_count < max_per_club


def is_double_gw(gw, fixtures_df):
    count = fixtures_df[fixtures_df['gw'] == gw]['team'].value_counts()
    return any(count > 1)


def is_difficult_gw(gw, fixtures_df, top_teams):
    gw_fixtures = fixtures_df[fixtures_df['gw'] == gw]
    tough_matches = gw_fixtures[
        (gw_fixtures['team'].isin(top_teams)) &
        (gw_fixtures['opponent_team'].isin(top_teams))
    ]
    return len(tough_matches) > len(top_teams) * 0.5


def select_best_trade(user_team, candidates_in, candidates_out, budget, model_predictions):
    best_gain = 0
    best_trade = (None, None)

    for _, player_in in candidates_in.iterrows():
        for _, player_out in candidates_out.iterrows():
            gain = (
                model_predictions.loc[player_in['id'], 'predicted_points'] -
                model_predictions.loc[player_out['id'], 'predicted_points']
            )

            if (
                gain > best_gain and
                player_in['cost'] <= player_out['cost'] + budget and
                is_valid_trade(user_team, player_in.to_dict(), player_out.to_dict()) and
                is_within_club_limit(user_team, player_in.to_dict())
            ):
                best_gain = gain
                best_trade = (player_in.to_dict(), player_out.to_dict())

    return best_trade
