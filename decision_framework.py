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

def optimize_trades(
    pred_df,
    squad_ids,
    bank_m,
    free_transfers,
    position_lookup,
    cost_lookup,
    horizon_weeks=5,
    # accept legacy/alternate kwargs:
    lookahead=None,
    projection_col=None,
    **kwargs,        # ignore any other incidental kwargs
):
    """
    Pick the single best trade within free transfers that increases projected points.

    Returns: (trade_outs, trade_ins, bank_after, free_after, trade_gain)
    All bank values are in millions.
    """
    # allow caller to override horizon via 'lookahead'
    if lookahead is not None:
        try:
            horizon_weeks = int(lookahead)
        except Exception:
            pass

    # ---------- SAFE DEFAULTS ----------
    best_gain   = 0.0
    best_out    = None
    best_in     = None
    bank_after  = float(bank_m)          # unchanged if no trade
    free_after  = int(free_transfers)    # unchanged if no trade
    trade_outs  = []
    trade_ins   = []

    # Which column represents expected points?
    # 1) explicit projection_col if provided
    # 2) xp_next_{horizon_weeks}
    # 3) projected_points
    # 4) ensemble_score
    if projection_col and projection_col in pred_df.columns:
        points_col = projection_col
    else:
        points_col = None
        for c in (f"xp_next_{horizon_weeks}", "projected_points", "ensemble_score"):
            if c in pred_df.columns:
                points_col = c
                break
        if points_col is None:
            pred_df = pred_df.copy()
            pred_df["ensemble_score"] = pred_df.get("ensemble_score", 0.0)
            points_col = "ensemble_score"

    squad_ids_set = set(int(x) for x in squad_ids)
    in_squad_df   = pred_df[pred_df["id"].isin(squad_ids_set)].copy()
    out_squad_df  = pred_df[~pred_df["id"].isin(squad_ids_set)].copy()

    for _, outrow in in_squad_df.iterrows():
        out_id      = int(outrow["id"])
        out_pos     = position_lookup.get(out_id)
        out_price_m = float(cost_lookup.get(out_id, 0.0))
        out_xp      = float(outrow.get(points_col, 0.0))
        if out_pos is None:
            continue

        def _affordable(pid: int) -> bool:
            in_price_m = float(cost_lookup.get(int(pid), 0.0))
            return (in_price_m - out_price_m) <= float(bank_m) + 1e-9

        same_pos_mask = out_squad_df["id"].map(lambda pid: position_lookup.get(int(pid)) == out_pos)
        afford_mask   = out_squad_df["id"].map(_affordable)
        candidates    = out_squad_df[same_pos_mask & afford_mask]

        for _, inrow in candidates.iterrows():
            in_id      = int(inrow["id"])
            in_xp      = float(inrow.get(points_col, 0.0))
            gain       = in_xp - out_xp
            if gain > best_gain + 1e-9:
                in_price_m = float(cost_lookup.get(in_id, 0.0))
                best_gain  = float(gain)
                best_out   = {
                    "id": out_id,
                    "name": outrow.get("name", ""),
                    "projected_points": out_xp,
                    "price_m": out_price_m,
                }
                best_in    = {
                    "id": in_id,
                    "name": inrow.get("name", ""),
                    "projected_points": in_xp,
                    "price_m": in_price_m,
                }
                bank_after = float(bank_m - (in_price_m - out_price_m))
                free_after = max(0, int(free_transfers) - 1)

    # Safety: never trade out someone not in the current squad
    if best_out and best_out["id"] not in squad_ids_set:
        best_out = None
        best_in = None
        best_gain = 0.0
        bank_after = float(bank_m)
        free_after = int(free_transfers)

    if best_out and best_in and best_gain > 0:
        trade_outs = [best_out]
        trade_ins  = [best_in]
        trade_gain = float(best_gain)
    else:
        trade_outs = []
        trade_ins  = []
        trade_gain = 0.0
        bank_after = float(bank_m)
        free_after = int(free_transfers)

    return trade_outs, trade_ins, bank_after, free_after, trade_gain

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
from config import PLAYER_IDLIST_PATH, CLEANED_PLAYERS_PATH, USE_MANUAL_WEIGHTS, MANUAL_WEIGHTS, MERGED_GW_PATH

def _element_to_position(et):
    try:
        et = int(et)
    except Exception:
        return "UNK"
    return {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(et, "UNK")

def make_decisions(features: pd.DataFrame, current_gw: int, team_status: dict):
    """
    Generate FPL decisions based on model predictions, user team, and rules.
    - Uses ensemble of ICT, Valuation, Opponent models.
    - Picks captain/vice from starting XI in user_team.xlsx.
    - Suggests trades using optimize_trades().
    """
  # decision_framework.py  (inside make_decisions)
    predictions_df = features.copy()

    # 🔒 ID hygiene: prefer the canonical 'element' column if available
    if "id" not in predictions_df.columns and "element" in predictions_df.columns:
        predictions_df["id"] = predictions_df["element"]

    # Map names → ids only if still missing
    id_to_name = get_id_to_name_map()
    name_to_id = {v: k for k, v in id_to_name.items()}
    if "id" not in predictions_df.columns and "name" in predictions_df.columns:
        predictions_df["id"] = predictions_df["name"].map(name_to_id)

    # Load current team
    user_team = read_current_team()

    if "Bank" not in user_team.columns:
        raise KeyError("user_team.xlsx must include a 'Bank' column")
    if "Free Transfers" not in user_team.columns:
        raise KeyError("user_team.xlsx must include a 'Free Transfers' column")
    if "Starting" not in user_team.columns:
        raise KeyError("user_team.xlsx must include a 'Starting' column with TRUE/FALSE values.")

    bank_value = float(user_team["Bank"].iloc[0])
    free_transfers = int(user_team["Free Transfers"].iloc[0])
    user_team["Starting"] = user_team["Starting"].astype(str).str.upper().eq("TRUE")

    # Optional Injured
    if "Injured" in user_team.columns:
        user_team["Injured"] = user_team["Injured"].astype(str).str.upper().eq("TRUE")
    else:
        user_team["Injured"] = False

    squad_ids = set(user_team["id"].dropna().astype(int).tolist())
    starting_ids = set(user_team.loc[user_team["Starting"], "id"].dropna().astype(int).tolist())
    injured_ids = set(user_team.loc[user_team["Injured"], "id"].dropna().astype(int).tolist())

        # --- POSITIONS + COST LOOKUP ---
    idlist_df = pd.read_csv(PLAYER_IDLIST_PATH, encoding="latin1")
    players_df = pd.read_csv(CLEANED_PLAYERS_PATH, encoding="latin1")

    # Prefer joining on ID if both have it
    if "id" in idlist_df.columns and "id" in players_df.columns:
        merged = idlist_df.merge(
            players_df[["id", "element_type", "now_cost"]],
            on="id",
            how="left"
        )
    else:
        # Fallback: normalize and join on name
        for df in [idlist_df, players_df]:
            for col in ["first_name", "second_name"]:
                if col in df.columns:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.normalize("NFKD")
                        .str.encode("ascii", errors="ignore")
                        .str.decode("utf-8")
                    )
            if "first_name" in df.columns and "second_name" in df.columns:
                df["name"] = df["first_name"].str.strip() + " " + df["second_name"].str.strip()

        merged = idlist_df.merge(
            players_df[["name", "element_type", "now_cost"]],
            on="name",
            how="left"
        )

    # Build lookups
    element_map = merged.set_index("id")["element_type"].dropna().to_dict()
    cost_lookup = merged.set_index("id")["now_cost"].dropna().to_dict()
    position_lookup = {pid: _element_to_position(et) for pid, et in element_map.items()}

    def element_to_position(et: int) -> str:
        return {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(et, "UNK")

    element_map = merged.set_index("id")["element_type"].to_dict()
    position_lookup = {pid: element_to_position(et) for pid, et in element_map.items()}
    cost_lookup = merged.set_index("id")["now_cost"].to_dict()

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

    starting_xi = in_squad[in_squad["id"].isin(starting_ids)]
    bench = in_squad[~in_squad["id"].isin(starting_ids)]

    # Captain & Vice = top two projected from OUTFIELD starters (no GKs)
    starting_xi = starting_xi.copy()
    if "position" not in starting_xi.columns:
        starting_xi["position"] = starting_xi["id"].map(position_lookup)

    outfield = starting_xi[~starting_xi["position"].eq("GK")]
    pool = outfield if not outfield.empty else starting_xi  # fallback if something is weird

    starting_sorted = pool.sort_values("ensemble_score", ascending=False)
    captain = starting_sorted.iloc[0]
    vice_captain = starting_sorted.iloc[1] if len(starting_sorted) > 1 else captain
    captain_confidence = captain["ensemble_score"] - vice_captain["ensemble_score"]

    # Captain & Vice = top two projected from starters
    starting_sorted = starting_xi.sort_values("ensemble_score", ascending=False)
    captain = starting_sorted.iloc[0]
    vice_captain = starting_sorted.iloc[1] if len(starting_sorted) > 1 else captain
    captain_confidence = captain["ensemble_score"] - vice_captain["ensemble_score"]
    if starting_xi.empty:
        raise ValueError("No starters found — check 'Starting' column in user_team.xlsx.")

    # --- Optimize Trades ---
    trade_outs, trade_ins, bank_after_m, free_after, trade_gain = optimize_trades(
    pred_df=predictions_df,
    squad_ids=squad_ids,
    bank_m=bank_value,
    free_transfers=free_transfers,
    position_lookup=position_lookup,
    cost_lookup=cost_lookup,
    lookahead= 5
)


    # --- Chip logic ---
    chip = "None"
    if team_status.get("triple_captain_available") and current_gw in [2, 19, 34]:
        chip = "Triple Captain"
    elif team_status.get("bench_boost_available") and current_gw in [19, 34]:
        chip = "Bench Boost"
# 🧾 Loggable view of the XI: id, name, position, opponent_difficulty, expected points
    id_to_name = get_id_to_name_map()
    xi_view = (
        starting_xi[["id", "position", "opponent_difficulty", "ensemble_score"]]
        .rename(columns={"ensemble_score": "expected_points"})
        .assign(name=lambda d: d["id"].map(id_to_name))
        .sort_values("expected_points", ascending=False)
    )
    xi_records = xi_view.to_dict(orient="records")
    return {
        "trade_ins": trade_ins,
        "trade_outs": trade_outs,
        "captain": {"id": captain["id"], "name": captain["name"], "confidence_gap": captain_confidence},
        "vice_captain": {"id": vice_captain["id"], "name": vice_captain["name"]},
        "chip": chip,
        "starting_xi": xi_records,          # 👈 new
        "expected_points_total": float(xi_view["expected_points"].sum()),  # 👈 new        "bench": bench[["id", "name", "position", "ensemble_score"]].to_dict("records"),
        "bank_before": bank_value,
        "bank_after": bank_after_m,
        "free_transfers_before": free_transfers,
        "free_transfers_after": free_after,
        "squad_value": sum(cost_lookup.get(pid, 0) for pid in squad_ids) / 10.0,
        "trade_gain": trade_gain,
        "injury_flags": {
            "players": [id_to_name.get(pid, str(pid)) for pid in user_team.loc[user_team["Injured"], "id"]],
            "count": int(user_team["Injured"].sum())
        },
        "fixture_info": {
            "easiest_runs": [],   # placeholder until fixture difficulty is calculated
            "hardest_runs": []
}

    }