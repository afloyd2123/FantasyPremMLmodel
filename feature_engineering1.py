import os
import numpy as np
import pandas as pd
from config import BASE_DIR, DATA_DIR, GWS_DIR, MERGED_GW_PATH

FEATURES_OUT = os.path.join(BASE_DIR, "feature_engineering.csv")
def _read_csv_required(path: str, what: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} not found at: {path}")
    return pd.read_csv(path)

def _canon_gw_col(df: pd.DataFrame, prefer="gw") -> str:
    """
    Return a column name that contains the gameweek number.
    Tries: 'gw', 'GW', 'round', 'event'.
    Also creates a lowercase 'gw' copy for downstream merges.
    """
    for c in ["gw", "GW", "round", "event"]:
        if c in df.columns:
            if c != "gw":
                df["gw"] = df[c]
            # make sure it's integer-like if possible
            with pd.option_context('mode.chained_assignment', None):
                df["gw"] = pd.to_numeric(df["gw"], errors="coerce").astype("Int64")
            return "gw"
    raise KeyError("Could not find a GW column among ['gw','GW','round','event'].")

def create_features_for_current_gw(current_gw: int | None = None, save: bool = True) -> pd.DataFrame:
    """
    Build model features from Vaastav data. Returns a DataFrame of features.
    If current_gw is provided, the returned frame is filtered to that GW.
    """
# Load datasets using centralized paths
teams_path = os.path.join(DATA_DIR, "teams.csv")
fixtures_path = os.path.join(DATA_DIR, "fixtures.csv")
#gws_path = os.path.join(DATA_DIR, "merged_gw.csv")
players_path = os.path.join(BASE_DIR, "data_preprocessing.csv")

teams   = _read_csv_required(teams_path, "teams.csv")
gws     = _read_csv_required(MERGED_GW_PATH, "merged_gw.csv")
players = _read_csv_required(players_path, "data_preprocessing.csv")

# fixtures are optional; don't hard fail if missing structure
fixtures = pd.read_csv(fixtures_path) if os.path.exists(fixtures_path) else pd.DataFrame()

# Clean up nulls
for df in (teams, gws, players):
    df.fillna(0, inplace=True)

# Drop duplicate names
duplicate_names = players['name'].value_counts()
duplicate_names = duplicate_names[duplicate_names > 1].index.tolist()
players = players[~players['name'].isin(duplicate_names)]

# Merge team names into GWs
team_map = teams.set_index('id').name.to_dict()
gws['team_name'] = gws['team'].map(team_map)

fixtures['home_team'] = fixtures['team_a'].map(team_map)
fixtures['away_team'] = fixtures['team_h'].map(team_map)

gws = gws.merge(fixtures, left_on=['gw', 'team_name'], right_on=['gw', 'home_team'], how='left')
gws['match_location'] = np.where(gws['team_name'] == gws['home_team'], 'home', 'away')
gws['home'] = (gws['match_location'] == 'home').astype(int)

# Rolling averages
gws['rolling_avg_points'] = gws.groupby('name')['total_points'].transform(lambda x: x.rolling(5, 1).mean())
gws['rolling_avg_value'] = gws.groupby('name')['value'].transform(lambda x: x.rolling(5, 1).mean())

# Merge player positions
gws = pd.merge(gws, players[['name', 'element_type']], on='name', how='left')

# Team average points for relative performance
team_avg = gws.groupby('team')['total_points'].mean().reset_index()
team_avg.columns = ['team', 'team_avg_points']
gws = gws.merge(team_avg, on='team', how='left')
gws['relative_performance'] = gws['total_points'] - gws['team_avg_points']

# Player and team form
gws['form_last_3'] = gws.groupby('name')['total_points'].rolling(window=3).mean().reset_index(0, drop=True)

team_form_last_3 = gws.groupby(['team', 'round'])['total_points'].sum().rolling(window=3).mean().reset_index()
team_form_last_3.columns = ['team', 'round', 'team_form_last_3']
gws = gws.merge(team_form_last_3, on=['team', 'round'], how='left')

# Opponent strength via fixture difficulty
difficulty_cols = ['team_h_difficulty', 'team_a_difficulty']
gws = gws.merge(fixtures[['id'] + difficulty_cols], left_on='opponent_team', right_on='id', how='left')

gws['opponent_strength'] = np.where(
    gws['was_home'],
    gws['team_a_difficulty'],
    gws['team_h_difficulty']
)

# Add rolling BPS average
if 'bps' in gws.columns:
    gws['bps_rolling_avg'] = gws.groupby('name')['bps'].rolling(window=5).mean().reset_index(0, drop=True)

# Cumulative bonus points (useful as an overall performance proxy)
if 'bonus' in gws.columns:
    gws['bonus_cumulative'] = gws.groupby('name')['bonus'].cumsum()


# Drop unneeded columns
drop_cols = ['team_avg_points', 'id', 'team_h_difficulty', 'team_a_difficulty']
gws.drop(columns=[col for col in drop_cols if col in gws.columns], inplace=True)

# Save enhanced feature dataset
output_path = os.path.join(BASE_DIR, "feature_engineering.csv")
gws.to_csv(output_path, index=False)
