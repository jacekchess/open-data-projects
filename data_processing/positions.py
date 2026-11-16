
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


def parse_player_appearances(json_path):
    """Extract player appearances and their playing time from one match."""

    with open(json_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    def event_time(event):
        minute = event.get("minute", 0)
        second = event.get("second", 0)

        timestamp = event.get("timestamp", "00:00:00.000")
        milliseconds = 0

        if "." in timestamp:
            milliseconds = int(timestamp.split(".")[-1]) / 1000

        return minute * 60 + second + milliseconds

    events = sorted(events, key=lambda e: e.get("index", 0))

    match_end = max(
        (event_time(e) for e in events),
        default=0
    )

    appearances = []
    active_players = {}

    def add_player(player, team, start_time, position=None):
        if not player or not team:
            return

        key = (team["id"], player["id"])

        appearance = {
            "player_id": player["id"],
            "player_name": player["name"],
            "team": team["name"],
            "start_seconds": start_time,
            "end_seconds": None,
            "end_position": position or "Unknown",
        }

        appearances.append(appearance)
        active_players[key] = appearance

    for event in events:
        event_type = event.get("type", {}).get("name")
        team = event.get("team")
        player = event.get("player")
        position = event.get("position")

        current_time = event_time(event)

        # Starting XI
        if event_type == "Starting XI":
            for item in event.get("tactics", {}).get("lineup", []):
                player_position = item.get("position", {}).get("name")

                add_player(
                    player=item.get("player"),
                    team=team,
                    start_time=0,
                    position=player_position,
                )

        # Substitution
        elif event_type == "Substitution" and team and player:
            team_id = team["id"]
            outgoing_id = player["id"]

            outgoing = active_players.get((team_id, outgoing_id))

            if outgoing:
                if position:
                    outgoing["end_position"] = position["name"]

                outgoing["end_seconds"] = current_time
                del active_players[(team_id, outgoing_id)]

            replacement = event.get("substitution", {}).get("replacement")

            if replacement:
                add_player(
                    player=replacement,
                    team=team,
                    start_time=current_time,
                )

        # Update the latest recorded position of active players
        elif player and team:
            key = (team["id"], player["id"])
            appearance = active_players.get(key)

            if appearance and position:
                appearance["end_position"] = position["name"]

    # Close appearances of players still on the pitch
    for appearance in appearances:
        if appearance["end_seconds"] is None:
            appearance["end_seconds"] = match_end

    # Calculate duration
    for appearance in appearances:
        appearance["minutes_played"] = max(
            0,
            (appearance["end_seconds"] - appearance["start_seconds"]) / 60
        )

    return appearances


def aggregate_all_matches(events_dir):
    """Parse all match files and aggregate player statistics."""

    events_dir = Path(events_dir)
    all_appearances = []

    files = sorted(events_dir.rglob("*.json"))

    print(f"Found {len(files)} match files.")

    for i, file_path in enumerate(files, start=1):
        try:
            appearances = parse_player_appearances(file_path)

            for appearance in appearances:
                appearance["match_file"] = file_path.name

            all_appearances.extend(appearances)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        if i % 100 == 0:
            print(f"Processed {i}/{len(files)} files")

    print(f"Extracted {len(all_appearances)} player appearances.")

    if not all_appearances:
        return pd.DataFrame()

    df = pd.DataFrame(all_appearances)

    # Aggregate minutes by player and position.
    position_stats = (
        df.groupby(
            ["player_id", "player_name", "end_position"],
            as_index=False
        )
        .agg(position_minutes=("minutes_played", "sum"))
    )

    # Find the position with the most accumulated minutes.
    position_stats = position_stats.sort_values(
        ["player_id", "position_minutes", "end_position"],
        ascending=[True, False, True]
    )

    dominant_positions = (
        position_stats
        .drop_duplicates(subset=["player_id"])
        .rename(columns={
            "end_position": "position"
        })[
            ["player_id", "position", "position_minutes"]
        ]
    )

    # Aggregate total player statistics.
    player_stats = (
        df.groupby(
            ["player_id", "player_name"],
            as_index=False
        )
        .agg(
            total_minutes=("minutes_played", "sum"),
            matches_played=("match_file", "nunique"),
        )
    )

    # Add dominant position.
    result = player_stats.merge(
        dominant_positions,
        on="player_id",
        how="left"
    )

    # Add a readable breakdown of minutes by position.
    position_breakdown = (
        position_stats
        .groupby(["player_id"])
        .apply(
            lambda group: "; ".join(
                f"{row['end_position']}: {row['position_minutes']:.1f}"
                for _, row in group.iterrows()
            ),
            include_groups=False
        )
        .rename("positions_breakdown")
        .reset_index()
    )

    result = result.merge(
        position_breakdown,
        on="player_id",
        how="left"
    )

    result["total_minutes"] = result["total_minutes"].round(2)
    result["position_minutes"] = result["position_minutes"].round(2)

    result = result.sort_values(
        "total_minutes",
        ascending=False
    ).reset_index(drop=True)

    return result


# --------------------------------------------------
# RUN
# --------------------------------------------------

events_directory = "open-data-projects/data/events/"

players = aggregate_all_matches(events_directory)

# Save aggregated player statistics
players.to_csv(
    "open-data-projects/data/processed/players_positions.csv",
    index=False,
    encoding="utf-8-sig"
)

print(players.head(20).to_string(index=False))
print(f"\nTotal unique players: {len(players)}")
