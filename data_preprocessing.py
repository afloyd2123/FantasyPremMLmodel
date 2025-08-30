import pandas as pd
import os
from config import CLEANED_PLAYERS_PATH, BASE_DIR

# Load this season's cleaned player dataset
players = pd.read_csv(CLEANED_PLAYERS_PATH)

# Generate the name column
players['name'] = players['first_name'] + ' ' + players['second_name']


def compute_form(player_data, window=5):
    player_data['form'] = (
        player_data.groupby('name')['total_points']
        .rolling(window=window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return player_data


def compute_home_away_advantage(player_data):
    home_avg = player_data[player_data['was_home']].groupby('name')['total_points'].mean()
    away_avg = player_data[~player_data['was_home']].groupby('name')['total_points'].mean()
    advantage = home_avg - away_avg
    player_data['home_advantage'] = player_data['name'].map(advantage)
    return player_data


def compute_team_strength(player_data, team_data):
    team_strength = team_data.set_index('id')['strength_overall_home'].to_dict()
    player_data['team_strength'] = player_data['team'].map(team_strength)
    return player_data


def compute_opponent_strength(player_data, team_data):
    opponent_strength = team_data.set_index('id')['strength_defence_away'].to_dict()
    player_data['opponent_strength'] = player_data['opponent_team'].map(opponent_strength)
    return player_data


def compute_position_based_features(player_data):
    positions = pd.get_dummies(player_data['element_type'], prefix='position')
    player_data = pd.concat([player_data, positions], axis=1)
    return player_data


def compute_value_efficiency(player_data):
    player_data['value_efficiency'] = player_data['total_points'] / player_data['value']
    return player_data


def compute_rolling_averages(player_data, windows=[2, 3, 4]):
    for window in windows:
        col_name = f"avg_last_{window}_gws"
        player_data[col_name] = (
            player_data.groupby('name')['total_points']
            .rolling(window=window)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return player_data


def preprocess_data(player_data, team_data):
    player_data = compute_form(player_data)
    player_data = compute_home_away_advantage(player_data)
    player_data = compute_team_strength(player_data, team_data)
    player_data = compute_opponent_strength(player_data, team_data)
    player_data = compute_position_based_features(player_data)
    player_data = compute_value_efficiency(player_data)
    player_data = compute_rolling_averages(player_data)
    return player_data


# Save processed data
output_path = os.path.join(BASE_DIR, "data_preprocessing.csv")
players.to_csv(output_path, index=False)
