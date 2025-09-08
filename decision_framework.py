# decision_framework.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from config import (
    MERGED_GW_PATH,
    USE_MANUAL_WEIGHTS,
    MANUAL_WEIGHTS,
    CLEANED_PLAYERS_PATH,
    PLAYER_IDLIST_PATH,
)
from user_team_management import read_current_team, get_id_to_name_map
from rules import POSITIONAL_RULES, within_budget


# ---------- Weight Optimization ----------
def get_optimal_weights(predictions_df: pd.DataFrame, current_gw: int):
    """
    Compute optimal ensemble weights by regressing predictions against actuals.
    Uses up to the last 5 gameweeks (or all available if fewer).
    """
    actuals_df = pd.read_csv(MERGED_GW_PATH)

    # Normalize gameweek column
    gw_col = None
    for c in ["gw", "GW", "round", "event"]:
        if c in actuals_df.columns:
            gw_col = c
            break
    if gw_col is None:
        raise KeyError("No gameweek column found in merged_gw.csv")

    if "total_points" not in actuals_df.columns:
        raise KeyError("'total_points' not found in merged_gw.csv")

    # --- Lookback window (max 5 GWs) ---
    lookback = 5
    min_gw = max(1, current_gw - lookback + 1)
    actuals_df = actuals_df[
        (actuals_df[gw_col] >= min_gw) & (actuals_df[gw_col] <= current_gw)
    ][[gw_col, "element", "total_points"]]

    # Ensure predictions_df has an 'id' column
    if "id" not in predictions_df.columns:
        if "name" in predictions_df.columns:
            id_to_name = get_id_to_name_map()
            name_to_id = {v: k for k, v in id_to_name.items()}
            predictions_df["id"] = predictions_df["name"].map(name_to_id)
        else:
            raise KeyError("predictions_df has no 'id' or 'name' column to merge on.")

    merged = pd.merge(
        predictions_df,
        actuals_df,
        left_on="id",
        right_on="element",
        how="inner",
    )

    # Normalize total_points column if suffixed
    if "total_points" not in merged.columns:
        if "total_points_y" in merged.columns:
            merged = merged.rename(columns={"total_points_y": "total_points"})
        elif "total_points_x" in merged.columns:
            merged = merged.rename(columns={"total_points_x": "total_points"})
        else:
            raise KeyError(
                f"Merge failed to bring in total_points. Columns: {merged.columns.tolist()}"
            )

    df = merged.dropna(
        subset=["ict_prediction", "valuation_prediction", "opponent_prediction", "total_points"]
    )

    if df.empty:
        print("⚠️ Not enough data to optimize weights, falling back to equal weights.")
        return (1 / 3, 1 / 3, 1 / 3)

    X = df[["ict_prediction", "valuation_prediction", "opponent_prediction"]].values
    y = df["total_points"].values

    try:
        reg = LinearRegression(positive=True)
        reg.fit(X, y)
        w_ict, w_val, w_opp = reg.coef_
        total = w_ict + w_val + w_opp
        if total == 0:
            return (1 / 3, 1 / 3, 1 / 3)
        return (w_ict / total, w_val / total, w_opp / total)
    except Exception as e:
        print(f"⚠️ Weight optimization failed: {e}, falling back to equal weights.")
        return (1 / 3, 1 / 3, 1 / 3)

def optimize_trades(predictions_df, user_team, position_lookup, cost_lookup,
                    current_gw, lookahead=5, bank=0.0, free_transfers=1):

    """
    Suggest optimal trades over a multi-week horizon.
    - predictions_df: DataFrame with ensemble_score + gw info
    - user_team: DataFrame from user_team.xlsx
    - position_lookup: dict {id: position}
    - cost_lookup: dict {id: cost}
    - current_gw: current gameweek
    - lookahead: number of future GWs to consider (default 5)
    """
    squad_ids = set(user_team["id"].tolist())
    starting_ids = set(user_team.loc[user_team["Starting"], "id"].tolist())

    in_squad = predictions_df[predictions_df["id"].isin(squad_ids)].copy()
    out_squad = predictions_df[~predictions_df["id"].isin(squad_ids)].copy()

    # --- Project over multiple GWs ---
    max_gw = predictions_df["GW"].max() if "GW" in predictions_df.columns else current_gw
    end_gw = min(current_gw + lookahead - 1, max_gw)

    # Filter to relevant window
    window = predictions_df[(predictions_df["GW"] >= current_gw) & (predictions_df["GW"] <= end_gw)]

    def project_points(player_id):
        df = window[window["id"] == player_id]
        return df["ensemble_score"].sum() if not df.empty else 0

    # Baseline: projected team score with current XI
    baseline_score = sum(project_points(pid) for pid in starting_ids)

    trade_outs, trade_ins = [], []
    free_transfers = 2 if user_team.get("saved_transfer", False) else 1
    max_trades = free_transfers

    # Consider all starters as possible trade-outs
    for pid_out in starting_ids:
        pos = position_lookup.get(pid_out, None)
        if not pos:
            continue

        candidates = out_squad[out_squad["position"] == pos]
        if candidates.empty:
            continue

        # Best candidate for this position
        best_in = candidates.sort_values("ensemble_score", ascending=False).iloc[0]
        pid_in = best_in["id"]

        # Projected scores
        out_score = project_points(pid_out)
        in_score = project_points(pid_in)

        improvement = in_score - out_score
        threshold = max(1.0, 0.1 * out_score)  # adaptive threshold

        # Budget check
        current_cost = sum(cost_lookup.get(pid, 0) for pid in squad_ids)
        cost_out = cost_lookup.get(pid_out, 0)
        cost_in = cost_lookup.get(pid_in, 0)
        new_total = current_cost - cost_out + cost_in

        if improvement > threshold and within_budget([new_total]):
            trade_outs.append({"id": pid_out, "name": best_in.get("name", "Unknown"), "projected_points": out_score})
            trade_ins.append({"id": pid_in, "name": best_in["name"], "projected_points": in_score})

        if len(trade_outs) >= max_trades:
            break

    return trade_outs, trade_ins, baseline_score


# ---------- Squad Optimization ----------
def pick_best_starting_xi(predictions_df: pd.DataFrame, position_lookup: dict):
    """
    Pick the best legal starting XI using FPL formation rules.
    Players with unknown positions are excluded earlier.
    """
    df = predictions_df.copy()
    df["position"] = df["id"].map(position_lookup)

    starters = []

    # Always take the top GK
    gks = df[df["position"] == "GK"].sort_values("ensemble_score", ascending=False)
    if not gks.empty:
        starters.append(gks.iloc[0].to_dict())   # ✅ preserve all columns

    # Then pick remaining 10 players
    others = df[df["position"] != "GK"].sort_values("ensemble_score", ascending=False)
    counts = {"DEF": 0, "MID": 0, "FWD": 0}

    for _, row in others.iterrows():
        pos = row["position"]
        if len(starters) == 11:
            break
        if counts[pos] >= POSITIONAL_RULES[pos]:
            continue
        remaining_slots = 11 - len(starters) - 1
        needed_def = max(0, 3 - counts["DEF"])
        needed_mid = max(0, 2 - counts["MID"])
        needed_fwd = max(0, 1 - counts["FWD"])
        if remaining_slots < needed_def + needed_mid + needed_fwd and counts[pos] >= 3:
            continue
        starters.append(row.to_dict())          # ✅ preserve all columns
        counts[pos] += 1

    starters_df = pd.DataFrame(starters)

    # Double-check ensemble_score exists
    if "ensemble_score" not in starters_df.columns:
        raise KeyError(f"Best XI missing ensemble_score. Columns: {starters_df.columns.tolist()}")

    return starters_df
    """
    Pick the best legal starting XI using FPL formation rules.
    Players with unknown positions are excluded earlier.
    """
    df = predictions_df.copy()
    df["position"] = df["id"].map(position_lookup)

    starters = []

    # Always take the top GK
    gks = df[df["position"] == "GK"].sort_values("ensemble_score", ascending=False)
    if not gks.empty:
        starters.append(gks.iloc[0])

    # Then pick remaining 10 players
    others = df[df["position"] != "GK"].sort_values("ensemble_score", ascending=False)
    counts = {"DEF": 0, "MID": 0, "FWD": 0}

    for _, row in others.iterrows():
        pos = row["position"]
        if len(starters) == 11:
            break
        if counts[pos] >= POSITIONAL_RULES[pos]:
            continue
        remaining_slots = 11 - len(starters) - 1
        needed_def = max(0, 3 - counts["DEF"])
        needed_mid = max(0, 2 - counts["MID"])
        needed_fwd = max(0, 1 - counts["FWD"])
        if remaining_slots < needed_def + needed_mid + needed_fwd and counts[pos] >= 3:
            continue
        starters.append(row)

        counts[pos] += 1

    starters_df = pd.DataFrame(starters)
    return starters_df

# ---------- Decision Engine ----------
def make_decisions(features: pd.DataFrame, current_gw: int, team_status: dict):
    """
    Generate FPL decisions based on model predictions, user team, and rules.
    Returns squad, trades, captaincy, chip usage, and helpful summary statistics.
    """
    predictions_df = features.copy()

    # Map names → ids if needed
    id_to_name = get_id_to_name_map()
    name_to_id = {v: k for k, v in id_to_name.items()}
    if "id" not in predictions_df.columns and "name" in predictions_df.columns:
        predictions_df["id"] = predictions_df["name"].map(name_to_id)

    # Load current team
    user_team = read_current_team()
    if "Bank" not in user_team.columns or "Free Transfers" not in user_team.columns:
        raise KeyError("user_team.xlsx must include 'Bank' and 'Free Transfers' columns.")

    bank_value = float(user_team["Bank"].iloc[0])   # e.g. 0.2
    free_transfers = int(user_team["Free Transfers"].iloc[0])

    if "Starting" in user_team.columns:
        user_team["Starting"] = user_team["Starting"].astype(str).str.upper().eq("TRUE")
    else:
        raise KeyError("user_team.xlsx must include a 'Starting' column with TRUE/FALSE values.")

    squad_ids = set(user_team["id"].tolist())

    # --- Position + Cost lookup ---
    idlist_df = pd.read_csv(PLAYER_IDLIST_PATH)
    players_df = pd.read_csv(CLEANED_PLAYERS_PATH)
    idlist_df["name"] = idlist_df["first_name"] + " " + idlist_df["second_name"]
    players_df["name"] = players_df["first_name"] + " " + players_df["second_name"]
    merged = idlist_df.merge(players_df, how="left", on="name")

    def element_to_position(et: int) -> str:
        return {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(et, "UNK")

    if "element_type" in merged.columns:
        element_map = merged.set_index("id")["element_type"].to_dict()
        position_lookup = {pid: element_to_position(et) for pid, et in element_map.items()}
    else:
        position_lookup = {pid: "UNK" for pid in merged["id"]}

    cost_lookup = merged.set_index("id")["now_cost"].to_dict() if "now_cost" in merged.columns else {}

    # --- Ensemble scoring ---
    if USE_MANUAL_WEIGHTS:
        w_ict, w_val, w_opp = (
            MANUAL_WEIGHTS["ict"],
            MANUAL_WEIGHTS["valuation"],
            MANUAL_WEIGHTS["opponent"],
        )
        print(f"ℹ️ Using manual weights: {MANUAL_WEIGHTS}")
    else:
        w_ict, w_val, w_opp = get_optimal_weights(predictions_df, current_gw)
        print(f"📊 Optimized weights → ICT={w_ict:.2f}, VAL={w_val:.2f}, OPP={w_opp:.2f}")

    predictions_df["ensemble_score"] = (
        w_ict * predictions_df["ict_prediction"]
        + w_val * predictions_df["valuation_prediction"]
        + w_opp * predictions_df["opponent_prediction"]
    )

    # --- Split squad ---
    in_squad = predictions_df[predictions_df["id"].isin(squad_ids)].copy()
    in_squad["position"] = in_squad["id"].map(position_lookup)

    starting_ids = set(user_team.loc[user_team["Starting"], "id"].tolist())
    starting_xi = in_squad[in_squad["id"].isin(starting_ids)]
    bench = in_squad[~in_squad["id"].isin(starting_ids)]

    if starting_xi.empty:
        raise ValueError("No starters found — check 'Starting' column in user_team.xlsx.")

    # Captain & Vice
    starting_sorted = starting_xi.sort_values("ensemble_score", ascending=False)
    captain = starting_sorted.iloc[0]
    vice_captain = starting_sorted.iloc[1] if len(starting_sorted) > 1 else captain
    captain_confidence = captain["ensemble_score"] - vice_captain["ensemble_score"]

    # --- Optimize Trades (multi-week) ---
    trade_outs, trade_ins, baseline_score = optimize_trades(
        predictions_df,
        user_team,
        position_lookup,
        cost_lookup,
        current_gw,
        lookahead=5,
        bank=bank_value,
        free_transfers=free_transfers,
    )

    # Projected gain from trades
    trade_gain = 0
    if trade_outs and trade_ins:
        trade_gain = sum(t["projected_points"] for t in trade_ins) - sum(t["projected_points"] for t in trade_outs)

    # Bank + transfers after trades
    total_squad_cost = sum(cost_lookup.get(pid, 0) for pid in squad_ids)
    if trade_outs and trade_ins:
        cost_out = sum(cost_lookup.get(t["id"], 0) for t in trade_outs)
        cost_in = sum(cost_lookup.get(t["id"], 0) for t in trade_ins)
        new_total = total_squad_cost - cost_out + cost_in
        new_bank = (1000 + bank_value * 10 - new_total) / 10.0
    else:
        new_bank = bank_value
    new_free_transfers = max(0, free_transfers - len(trade_outs))

    # Squad value
    squad_value = total_squad_cost / 10.0

    # Injury / Risk summary
    injury_flags = {}
    for col in ["red_cards", "yellow_cards"]:
        if col in in_squad.columns:
            injury_flags[col] = int(in_squad[col].sum())

    # Fixture horizon (easy vs hard runs for YOUR squad only)
    fixture_info = {}
    if "opponent_difficulty" in predictions_df.columns:
        horizon = predictions_df[
            (predictions_df["GW"] >= current_gw) & (predictions_df["GW"] < current_gw + 5)
        ]
        horizon = horizon[horizon["id"].isin(squad_ids)]  # ✅ restrict to your team only

    if not horizon.empty:
        avg_difficulty = horizon.groupby("id")["opponent_difficulty"].mean()
        easiest = avg_difficulty.nsmallest(min(3, len(avg_difficulty))).index.tolist()
        hardest = avg_difficulty.nlargest(min(3, len(avg_difficulty))).index.tolist()
        fixture_info = {
            "easiest_runs": [id_to_name.get(pid, str(pid)) for pid in easiest],
            "hardest_runs": [id_to_name.get(pid, str(pid)) for pid in hardest],
        }

    # Chip logic
    chip = "None"
    if team_status.get("triple_captain_available") and current_gw in [2, 19, 34]:
        chip = "Triple Captain"
    elif team_status.get("bench_boost_available") and current_gw in [19, 34]:
        chip = "Bench Boost"

    return {
        "trade_ins": trade_ins,
        "trade_outs": trade_outs,
        "trade_gain": trade_gain,
        "captain": {"id": captain["id"], "name": captain["name"], "confidence_gap": captain_confidence},
        "vice_captain": {"id": vice_captain["id"], "name": vice_captain["name"]},
        "chip": chip,
        "starting_xi": starting_sorted[["id", "name", "position", "ensemble_score"]].to_dict("records"),
        "bench": bench[["id", "name", "position", "ensemble_score"]].to_dict("records"),
        "baseline_score_next5": baseline_score,
        "bank_before": bank_value,
        "bank_after": new_bank,
        "free_transfers_before": free_transfers,
        "free_transfers_after": new_free_transfers,
        "squad_value": squad_value,
        "injury_flags": injury_flags,
        "fixture_info": fixture_info,
    }
