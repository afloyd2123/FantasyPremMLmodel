import pandas as pd
from config import (
    USER_TEAM_PATH,
    USER_PERFORMANCE_PATH,
    MERGED_GW_PATH,
    PLAYER_IDLIST_PATH
)

# ========== ID to Name Lookup ==========
def load_player_name_lookup():
    idlist = pd.read_csv(PLAYER_IDLIST_PATH)
    return idlist.set_index('id').apply(lambda x: f"{x['first_name']} {x['second_name']}", axis=1).to_dict()


# ========== Performance Tracker ==========
def update_user_performance(gw):
    """
    Updates the user's performance tracking sheet after a gameweek.
    Uses player IDs and logs captaincy, points, and transfer penalties.
    """
    # Load data
    user_team = pd.read_excel(USER_TEAM_PATH, sheet_name='Team')
    merged_gw = pd.read_csv(MERGED_GW_PATH)
    id_to_name = load_player_name_lookup()

    try:
        user_performance = pd.read_excel(USER_PERFORMANCE_PATH)
    except FileNotFoundError:
        user_performance = pd.DataFrame(columns=[
            "Game Week", "Points", "Transfers Made", "Captain", "Vice Captain"
        ])

    # Filter this gameweek's data
    gw_data = merged_gw[merged_gw['gw'] == gw]
    user_ids = user_team['id'].tolist()
    user_gw_data = gw_data[gw_data['element'].isin(user_ids)]

    # Total points from starters
    total_points = user_gw_data['total_points'].sum()

    # Captain logic
    captain_id = user_team[user_team['captain'] == True]['id'].iloc[0]
    captain_points = user_gw_data[gw_data['element'] == captain_id]['total_points'].values
    if captain_points.size > 0:
        total_points += captain_points[0]  # double counted

    # Transfer penalty
    transfers_made = user_team['transfer_in'].notna().sum()
    if transfers_made > 1:
        penalty = (transfers_made - 1) * 4
        total_points -= penalty
    else:
        penalty = 0

    # Get names for captain and vice
    captain_name = id_to_name.get(captain_id, "Unknown")
    vice_id = user_team[user_team['vice_captain'] == True]['id'].iloc[0]
    vice_name = id_to_name.get(vice_id, "Unknown")

    # Log performance
    new_row = {
        "Game Week": gw,
        "Points": total_points,
        "Transfers Made": transfers_made,
        "Captain": captain_name,
        "Vice Captain": vice_name
    }

    user_performance = pd.concat([user_performance, pd.DataFrame([new_row])], ignore_index=True)
    user_performance.to_excel(USER_PERFORMANCE_PATH, index=False)

    print(f"✅ GW{gw} logged: {total_points} pts (Captain: {captain_name}, Transfers: {transfers_made}, Penalty: {penalty})")
