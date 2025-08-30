import pandas as pd
from config import (
    USER_TEAM_PATH,
    USER_PERFORMANCE_PATH,
    PLAYER_IDLIST_PATH,
    MERGED_GW_PATH
)

# ===== Load ID → Name Mapping =====
def get_id_to_name_map():
    idlist = pd.read_csv(PLAYER_IDLIST_PATH)
    return idlist.set_index('id').apply(lambda x: f"{x['first_name']} {x['second_name']}", axis=1).to_dict()


# ===== Read Your Current Team =====
def read_current_team():
    return pd.read_excel(USER_TEAM_PATH, sheet_name='Team')


# ===== Update Your Team After Trades =====
def update_user_team_with_trades(player_in, player_out_id):
    team = read_current_team()

    # Remove player by ID
    team = team[team['id'] != player_out_id]

    # Add traded-in player (assume you look up their metadata externally)
    new_player = {
        "id": player_in['id'],
        "first_name": player_in['first_name'],
        "second_name": player_in['second_name'],
        "position": player_in.get("position", "TBD"),
        "status": "Starting",
        "captain": False,
        "vice_captain": False,
        "transfer_in": f"Traded in GWX"
    }

    team = pd.concat([team, pd.DataFrame([new_player])], ignore_index=True)
    team.to_excel(USER_TEAM_PATH, sheet_name='Team', index=False)
    print(f"🔁 Team updated: OUT {player_out_id}, IN {new_player['id']}")


# ===== Update User Performance After GW =====
def update_user_performance(gw):
    user_team = read_current_team()
    merged_gw = pd.read_csv(MERGED_GW_PATH)
    id_to_name = get_id_to_name_map()

    try:
        performance = pd.read_excel(USER_PERFORMANCE_PATH)
    except FileNotFoundError:
        performance = pd.DataFrame(columns=["Game Week", "Player", "Position", "Points", "Status"])

    players = user_team["id"].tolist()
    team_lookup = {row["id"]: row for _, row in user_team.iterrows()}

    new_rows = []

    for pid in players:
        player_row = merged_gw[(merged_gw["element"] == pid) & (merged_gw["gw"] == gw)]

        if not player_row.empty:
            points = player_row["total_points"].values[0]
            position = player_row["element_type"].values[0]
        else:
            points = 0
            position = "N/A"

        status = team_lookup[pid]["status"]
        player_name = id_to_name.get(pid, "Unknown")

        new_rows.append({
            "Game Week": gw,
            "Player": player_name,
            "Position": position,
            "Points": points,
            "Status": status
        })

    performance = pd.concat([performance, pd.DataFrame(new_rows)], ignore_index=True)
    performance.to_excel(USER_PERFORMANCE_PATH, index=False)
    print(f"📈 GW{gw} performance logged for {len(new_rows)} players.")
