import json
import csv
from pathlib import Path
from collections import defaultdict

EVENTS_DIR = Path("open-data-projects/data/events")
OUTPUT_FILE = Path("open-data-projects/data/processed/defending.csv")


def get_name(value):
    if isinstance(value, dict):
        return value.get("name")
    return value


def get_player(event):
    player = event.get("player") or {}
    return player.get("id"), player.get("name")


def normalize(value):
    if value is None:
        return ""
    return str(value).strip().lower().replace("_", " ")


def get_outcome(event):
    # Duel outcomes are stored inside the duel object.
    duel = event.get("duel") or {}
    return normalize(get_name(duel.get("outcome")))


def is_won(outcome):
    return outcome in {
        "won",
        "success",
        "success in play",
        "success out",
    }


def is_lost(outcome):
    return outcome in {
        "lost",
        "lost in play",
        "lost out",
    }


def calculate_percentage(numerator, denominator):
    """Return a percentage, or None when the denominator is zero."""
    if denominator == 0:
        return None
    return round((numerator / denominator) * 100, 2)


def process_aerial_duel(event, event_by_id, player_stats):
    duel = event.get("duel") or {}
    duel_type = normalize(get_name(duel.get("type")))

    if duel_type != "aerial lost":
        return

    loser_id, loser_name = get_player(event)

    if loser_id is None:
        return

    # The player attached to "Aerial Lost" is the loser.
    if loser_id not in player_stats:
        player_stats[loser_id] = new_player_stats(
            loser_id, loser_name
        )

    player_stats[loser_id]["aerial_lost"] += 1

    # Look up the linked event to identify the player who gained
    # possession after the aerial duel.
    for related_id in event.get("related_events", []):
        related_event = event_by_id.get(related_id)

        if not related_event:
            continue

        winner_id, winner_name = get_player(related_event)

        if winner_id is None or winner_id == loser_id:
            continue

        loser_team = (event.get("team") or {}).get("id")
        winner_team = (related_event.get("team") or {}).get("id")

        if loser_team is None or winner_team is None:
            continue

        if loser_team == winner_team:
            continue

        # Only attribute a win when the linked event is a
        # possession/action event by the opposing player.
        winner_event_type = get_name(related_event.get("type"))

        if winner_event_type not in {
            "Pass",
            "Carry",
            "Clearance",
            "Shot",
            "Ball Recovery",
            "Interception",
        }:
            continue

        if winner_id not in player_stats:
            player_stats[winner_id] = new_player_stats(
                winner_id, winner_name
            )

        player_stats[winner_id]["aerial_won"] += 1
        break


def new_player_stats(player_id, player_name):
    return {
        "player_id": player_id,
        "player_name": player_name,

        "pressures": 0,
        "interceptions": 0,
        "blocks": 0,

        "clearances": 0,
        "aerial_clearances_won": 0,

        # Aerial duel counts. `aerial_lost` now means actual lost duels,
        # rather than every event whose duel type is "Aerial Lost".
        "aerial_won": 0,
        "aerial_lost": 0,

        "tackles_won": 0,
        "tackles_lost": 0,

        "ball_recoveries": 0,
        "dribbled_past": 0,
        "dispossessed": 0,

        "fouls": 0,
        "yellow_cards": 0,
        "red_cards": 0,
    }


def process_event(event, player_stats):
    player_id, player_name = get_player(event)

    if player_id is None:
        return

    if player_id not in player_stats:
        player_stats[player_id] = new_player_stats(
            player_id,
            player_name
        )

    stats = player_stats[player_id]

    # Update player name if it was previously missing.
    if not stats["player_name"] and player_name:
        stats["player_name"] = player_name

    event_type = get_name(event.get("type"))

    # --------------------------------------------------
    # Individual defensive actions
    # --------------------------------------------------

    if event_type == "Pressure":
        stats["pressures"] += 1

    elif event_type == "Interception":
        stats["interceptions"] += 1

    elif event_type == "Block":
        stats["blocks"] += 1

    elif event_type == "Clearance":
        stats["clearances"] += 1

        clearance = event.get("clearance") or {}

        # This is an aerial clearance, not a general aerial duel win.
        if clearance.get("aerial_won") is True:
            stats["aerial_clearances_won"] += 1

    elif event_type == "Ball Recovery":
        stats["ball_recoveries"] += 1

    elif event_type == "Dribbled Past":
        stats["dribbled_past"] += 1

    elif event_type == "Dispossessed":
        stats["dispossessed"] += 1

    # --------------------------------------------------
    # Duels: tackles and aerial duels
    # --------------------------------------------------

    elif event_type == "Duel":
        duel = event.get("duel") or {}
        duel_type = normalize(get_name(duel.get("type")))
        outcome = get_outcome(event)

        if duel_type == "tackle":
            if is_won(outcome):
                stats["tackles_won"] += 1
            elif is_lost(outcome):
                stats["tackles_lost"] += 1


    # --------------------------------------------------
    # Fouls and cards
    # --------------------------------------------------

    elif event_type == "Foul Committed":
        stats["fouls"] += 1

        foul = event.get("foul_committed") or {}
        card = normalize(get_name(foul.get("card")))

        if card == "yellow card":
            stats["yellow_cards"] += 1

        elif card in {"red card", "second yellow"}:
            stats["red_cards"] += 1

            # A second yellow is also a yellow booking.
            if card == "second yellow":
                stats["yellow_cards"] += 1

    elif event_type == "Bad Behaviour":
        bad_behaviour = event.get("bad_behaviour") or {}
        card = normalize(get_name(bad_behaviour.get("card")))

        if card == "yellow card":
            stats["yellow_cards"] += 1

        elif card in {"red card", "second yellow"}:
            stats["red_cards"] += 1

            if card == "second yellow":
                stats["yellow_cards"] += 1


def add_derived_metrics(row):
    # Tackle success: won tackles / all tackles with a known outcome.
    tackle_attempts = row["tackles_won"] + row["tackles_lost"]
    row["tackle_success_pct"] = calculate_percentage(
        row["tackles_won"],
        tackle_attempts
    )

    # Aerial duel success: won aerial duels / all aerial duels with
    # a known outcome. Aerial clearances are deliberately excluded.
    aerial_duels = row["aerial_won"] + row["aerial_lost"]
    row["aerial_duel_win_pct"] = calculate_percentage(
        row["aerial_won"],
        aerial_duels
    )

    # Approximation of how often the defender is beaten in recorded
    # defensive 1v1 events. The denominator combines tackle duels
    # (won + lost) with separate "Dribbled Past" events.
    # It is not a complete measure of all defensive 1v1 situations.
    defensive_1v1_events = (
        row["tackles_won"]
        + row["tackles_lost"]
        + row["dribbled_past"]
    )
    row["dribbled_past_pct"] = calculate_percentage(
        row["dribbled_past"],
        defensive_1v1_events
    )

    return row


def main():
    player_stats = defaultdict(dict)

    files = list(EVENTS_DIR.rglob("*.json"))

    print(f"Found {len(files)} event files.")

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                events = json.load(file)

            if not isinstance(events, list):
                continue

            event_by_id = {
                event["id"]: event
                for event in events
                if event.get("id")
            }

            for event in events:
                process_event(event, player_stats)

            for event in events:
                process_aerial_duel(
                    event,
                    event_by_id,
                    player_stats
                )

        except (json.JSONDecodeError, OSError) as error:
            print(f"Could not process {file_path}: {error}")

    # --------------------------------------------------
    # Convert to rows, calculate derived metrics and save
    # --------------------------------------------------

    columns = [
        "player_id",
        "player_name",
        "pressures",
        "interceptions",
        "blocks",
        "clearances",
        "aerial_clearances_won",
        "aerial_won",
        "aerial_lost",
        "aerial_duel_win_pct",
        "tackles_won",
        "tackles_lost",
        "tackle_success_pct",
        "ball_recoveries",
        "dribbled_past",
        "dribbled_past_pct",
        "dispossessed",
        "fouls",
        "yellow_cards",
        "red_cards",
    ]

    rows = [
        add_derived_metrics(dict(row))
        for row in player_stats.values()
    ]

    rows = sorted(
        rows,
        key=lambda row: (
            row["tackles_won"],
            row["interceptions"],
            row["pressures"],
        ),
        reverse=True,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} players to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
