import os

# 🌍 Base directory of your project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 📆 Choose the season to work with here
SEASON = "2025-26"  # Change this once to test a different season

# 📁 Data repository path (from Vaastav GitHub)
REPO_DIR = os.path.join(BASE_DIR, "Fantasy-Premier-League")
DATA_DIR = os.path.join(REPO_DIR, "data", SEASON)
GWS_DIR = os.path.join(DATA_DIR, "gws")
# 📄 Common data files
MERGED_GW_PATH = os.path.join(GWS_DIR, "merged_gw.csv")
CLEANED_PLAYERS_PATH = os.path.join(DATA_DIR, "cleaned_players.csv")
LINEUP_HISTORY_PATH = os.path.join(BASE_DIR, "history_lineups.csv")

#USER_TEAM_PATH = os.path.join(BASE_DIR, "user_team.xlsx")
USER_PERFORMANCE_PATH = os.path.join(BASE_DIR, "user_performance.xlsx")

WEIGHTS_LOG_PATH = os.path.join(BASE_DIR, "weights_log.csv")
# Optional: Manual override for ensemble weights
USE_MANUAL_WEIGHTS = False  # Set to True to bypass regression
MANUAL_WEIGHTS = {
    "ict": 0.4,
    "valuation": 0.4,
    "opponent": 0.2
}

# 📁 Model paths
ICT_MODEL_PATH = os.path.join(BASE_DIR, "ICT_model.pkl")
VALUATION_MODEL_PATH = os.path.join(BASE_DIR, "valuation_model2.pkl")
OPPONENT_MODEL_PATH = os.path.join(BASE_DIR, "opponent_model.pkl")

# User files
USER_TEAM_PATH = os.path.join(BASE_DIR, "user_team.xlsx")
USER_PERFORMANCE_PATH = os.path.join(BASE_DIR, "user_performance.xlsx")
PLAYER_IDLIST_PATH = os.path.join(DATA_DIR, "player_idlist.csv")
#MERGED_GW_PATH = os.path.join(DATA_DIR, "merged_gw.csv")


# 📄 Output logs
ERROR_METRICS_PATH = os.path.join(BASE_DIR, "error_metrics.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "history.csv")
