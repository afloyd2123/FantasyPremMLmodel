# feature_engineering.py
import os
import numpy as np
import pandas as pd
from config import BASE_DIR, DATA_DIR, MERGED_GW_PATH

FEATURES_OUT = os.path.join(BASE_DIR, "feature_engineering.csv")

def _read_csv_required(path: str, what: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} not found at: {path}")
    return pd.read_csv(path)

def _canon_gw(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a numeric 'gw' column exists by copying from common alternatives."""
    for c in ("gw", "GW", "round", "event"):
        if c in df.columns:
            if c != "gw":
                df["gw"] = df[c]
            df["gw"] = pd.to_numeric(df["gw"], errors="coerce").astype("Int64")
            return df
    raise KeyError("No gameweek column found (expected one of: 'gw','GW','round','event').")

def create_features_for_current_gw(current_gw: int | None = None, save: bool = True) -> pd.DataFrame:
    teams_path    = os.path.join(DATA_DIR, "teams.csv")
    fixtures_path = os.path.join(DATA_DIR, "fixtures.csv")
    players_path  = os.path.join(BASE_DIR, "data_preprocessing.csv")

    teams   = _read_csv_required(teams_path,    "teams.csv")
    fixtures= _read_csv_required(fixtures_path, "fixtures.csv")
    gws     = _read_csv_required(MERGED_GW_PATH,"merged_gw.csv")
    players = _read_csv_required(players_path,  "data_preprocessing.csv")

    # Fill NA
    for df in (teams, gws, players, fixtures):
        df.fillna(0, inplace=True)

    # Normalize GW column in both sources
    gws      = _canon_gw(gws)
    fixtures = _canon_gw(fixtures)

    # Drop duplicate player names if needed
    if "name" in players.columns:
        dup_names = players["name"].value_counts()
        dup_names = dup_names[dup_names > 1].index.tolist()
        if dup_names:
            players = players[~players["name"].isin(dup_names)]

    # Map team id -> name (optional, not required for difficulties)
    if {"id", "name"}.issubset(teams.columns) and "team" in gws.columns:
        team_map = teams.set_index("id")["name"].to_dict()
        gws["team_name"] = gws["team"].map(team_map)

    # --- Opponent difficulty merge (robust, works for home and away opponents)
    # Need these columns present in gws
    required_gws_cols = {"gw", "opponent_team", "was_home"}
    if not required_gws_cols.issubset(gws.columns):
        missing = list(required_gws_cols - set(gws.columns))
        raise KeyError(f"merged_gw.csv missing required columns: {missing}")

    # Ensure integer dtypes for join keys
    gws["opponent_team"] = pd.to_numeric(gws["opponent_team"], errors="coerce").astype("Int64")
    fixtures["team_h"]   = pd.to_numeric(fixtures["team_h"],   errors="coerce").astype("Int64")
    fixtures["team_a"]   = pd.to_numeric(fixtures["team_a"],   errors="coerce").astype("Int64")

    # Merge when opponent is the home team
    gws = gws.merge(
        fixtures[["gw", "team_h", "team_h_difficulty"]],
        left_on=["gw", "opponent_team"],
        right_on=["gw", "team_h"],
        how="left"
    ).rename(columns={"team_h_difficulty": "opp_home_difficulty"}).drop(columns=["team_h"])

    # Merge when opponent is the away team
    gws = gws.merge(
        fixtures[["gw", "team_a", "team_a_difficulty"]],
        left_on=["gw", "opponent_team"],
        right_on=["gw", "team_a"],
        how="left"
    ).rename(columns={"team_a_difficulty": "opp_away_difficulty"}).drop(columns=["team_a"])

    # Compose a single opponent difficulty value depending on our home/away
    gws["opponent_difficulty"] = np.where(
        gws["was_home"].astype(bool),
        gws["opp_away_difficulty"],
        gws["opp_home_difficulty"]
    )

    # --- Rolling features (safe with small data)
    if "total_points" in gws.columns:
        gws["rolling_avg_points"] = gws.groupby("name")["total_points"].transform(lambda x: x.rolling(5, 1).mean())
    if "value" in gws.columns:
        gws["rolling_avg_value"]  = gws.groupby("name")["value"].transform(lambda x: x.rolling(5, 1).mean())
    if "bps" in gws.columns:
        gws["bps_rolling_avg"]    = gws.groupby("name")["bps"].transform(lambda x: x.rolling(5, 1).mean())

    # Merge player position if present
    if {"name", "element_type"}.issubset(players.columns):
        gws = pd.merge(gws, players[["name", "element_type"]], on="name", how="left")

    # Drop only truly-unneeded columns (KEEP difficulties)
    drop_cols = ["team_avg_points", "id"]
    gws.drop(columns=[c for c in drop_cols if c in gws.columns], inplace=True, errors="ignore")

    # Save
    if save:
        gws.to_csv(FEATURES_OUT, index=False)

    if current_gw is not None:
        return gws[gws["gw"] == int(current_gw)].copy()

    return gws

if __name__ == "__main__":
    df = create_features_for_current_gw(save=True)
    print(f"Feature frame saved to: {FEATURES_OUT}, shape={df.shape}")
    print("Columns now include:", [c for c in df.columns if "difficulty" in c or c in ("gw","was_home","opponent_team","opponent_difficulty")])
