# rules.py — 2025–26 Fantasy Premier League Constants

# ============ Squad Rules ============
BUDGET = 100.0  # in millions
MAX_PLAYERS = 15

POSITIONAL_RULES = {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3
}

MAX_PLAYERS_FROM_ONE_TEAM = 3


# ============ Transfer Rules ============
FREE_TRANSFERS = 1
MAX_FREE_TRANSFERS = 2
TRANSFER_PENALTY = 4  # points per extra transfer beyond free limit


def compute_transfer_penalty(transfers_made, accumulated_transfers):
    """
    Computes point deduction for excess transfers.
    """
    total_available = min(accumulated_transfers + FREE_TRANSFERS, MAX_FREE_TRANSFERS)
    extra = max(0, transfers_made - total_available)
    return extra * TRANSFER_PENALTY


# ============ Chip Rules ============
CHIPS_USAGE = {
    "Wildcard_1": False,
    "Wildcard_2": False,
    "Triple Captain": False,
    "Bench Boost": False,
    "Free Hit": False
}


def use_chip(chip_name):
    """
    Marks a chip as used.
    """
    if chip_name in CHIPS_USAGE and not CHIPS_USAGE[chip_name]:
        CHIPS_USAGE[chip_name] = True
        return True
    return False


# ============ Captaincy ============
CAPTAIN_MULTIPLIER = 2
VICE_CAPTAIN_MULTIPLIER = 2  # only if captain does not play


# ============ Point Scoring ============
POINT_RULES = {
    "play_up_to_60": 1,
    "play_60_or_more": 2,
    "goal_by_gk_or_def": 6,
    "goal_by_mid": 5,
    "goal_by_fwd": 4,
    "assist": 3,
    "clean_sheet_by_gk_or_def": 4,
    "clean_sheet_by_mid": 1,
    "penalty_save": 5,
    "penalty_missed": -2,
    "2_goals_conceded_by_gk_or_def": -1,
    "yellow_card": -1,
    "red_card": -3,
    "own_goal": -2,
    "3_saves_by_gk": 1,
    # bonus points are handled separately from BPS data
}


# ============ Validators ============

def valid_team_structure(player_positions):
    """
    Checks team positional constraints.
    """
    counts = {pos: player_positions.count(pos) for pos in POSITIONAL_RULES}
    return all(counts.get(pos, 0) <= max_count for pos, max_count in POSITIONAL_RULES.items())


def within_budget(player_costs):
    """
    Verifies total squad value is within budget.
    """
    return sum(player_costs) <= BUDGET


def valid_team_from_one_club(player_club_counts):
    """
    Enforces max club player limit.
    """
    return all(count <= MAX_PLAYERS_FROM_ONE_TEAM for count in player_club_counts.values())
