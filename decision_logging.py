import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HISTORY_LOG = BASE_DIR / "history_decisions.csv"

def log_decisions(current_gw, decisions):
    """
    Append this week's decisions to a persistent CSV log.
    """
    file_exists = HISTORY_LOG.exists()
    with open(HISTORY_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "gw", "captain", "vice_captain", "chip",
            "trade_outs", "trade_ins",
            "bank_before", "bank_after",
            "free_transfers_before", "free_transfers_after",
            "trade_gain", "squad_value"
        ])
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "gw": current_gw,
            "captain": decisions["captain"]["name"],
            "vice_captain": decisions["vice_captain"]["name"],
            "chip": decisions["chip"],
            "trade_outs": ", ".join([p["name"] for p in decisions["trade_outs"]]) if decisions["trade_outs"] else "",
            "trade_ins": ", ".join([p["name"] for p in decisions["trade_ins"]]) if decisions["trade_ins"] else "",
            "bank_before": decisions["bank_before"],
            "bank_after": decisions["bank_after"],
            "free_transfers_before": decisions["free_transfers_before"],
            "free_transfers_after": decisions["free_transfers_after"],
            "trade_gain": decisions["trade_gain"],
            "squad_value": decisions["squad_value"],
        })
