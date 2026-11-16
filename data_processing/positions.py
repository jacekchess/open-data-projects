
import json
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd


def load_player_nicknames(lineups_dir):
    """Load player nicknames from StatsBomb lineup JSON files."""

    lineups_dir = Path(lineups_dir)
    nicknames = {}

    files = sorted(lineups_dir.rglob("*.json"))

    print(f"Found {len(files)} lineup files.")

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                teams = json.load(f)

            if isinstance(teams, dict):
                teams = [teams]

            for team in teams:
                for player in team.get("lineup", []):
                    player_id = player.get("player_id")
                    nickname = player.get("player_nickname")

                    if player_id is not None and nickname:
                        nicknames[str(player_id)] = nickname

        except (json.JSONDecodeError, OSError, TypeError) as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Loaded {len(nicknames)} player nicknames.")

    return nicknames



def parse_player_appearances(json_path, competition_lookup):
    """Extract player appearances and playing time from one match."""

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

    match_id = json_path.stem
    competition_name = competition_lookup.get(match_id, "Unknown")

    appearances = []
    active_players = {}

    def add_player(player, team, start_time, position=None):
        if not player or not team:
            return

        player_id = player.get("id")
        team_id = team.get("id")

        if player_id is None or team_id is None:
            return

        key = (team_id, player_id)

        appearance = {
            "player_id": player_id,
            "player_name": player.get("name"),
            "nick_name": player.get("nickname"),
            "team": team.get("name"),
            "competition_name": competition_name,
            "match_file": json_path.name,
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
                player_data = item.get("player")
                player_position = item.get("position", {}).get("name")

                add_player(
                    player=player_data,
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

        # Update latest recorded position of active players
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


def load_competition_lookup(matches_dir):
    """
    Build match_id -> competition_name lookup
    from StatsBomb matches JSON files.
    """

    matches_dir = Path(matches_dir)
    competition_lookup = {}

    files = list(matches_dir.rglob("*.json"))

    print(f"Found {len(files)} match metadata files.")

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                matches = json.load(f)

            if not isinstance(matches, list):
                continue

            for match in matches:
                match_id = match.get("match_id")

                competition = match.get("competition") or {}
                competition_name = competition.get("competition_name")

                if match_id is not None and competition_name:
                    competition_lookup[str(match_id)] = competition_name

        except (json.JSONDecodeError, OSError) as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Loaded competition information for {len(competition_lookup)} matches.")

    return competition_lookup



def aggregate_all_matches(events_dir, matches_dir, lineups_dir):
    """Parse all matches and aggregate player statistics."""

    events_dir = Path(events_dir)

    # Load external metadata
    competition_lookup = load_competition_lookup(matches_dir)
    nickname_lookup = load_player_nicknames(lineups_dir)

    all_appearances = []

    files = sorted(events_dir.rglob("*.json"))

    print(f"Found {len(files)} event files.")

    for i, file_path in enumerate(files, start=1):
        try:
            appearances = parse_player_appearances(
                file_path,
                competition_lookup
            )

            all_appearances.extend(appearances)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        if i % 100 == 0:
            print(f"Processed {i}/{len(files)} files")

    print(f"Extracted {len(all_appearances)} player appearances.")

    if not all_appearances:
        return pd.DataFrame()

    df = pd.DataFrame(all_appearances)

    # Normalize IDs for reliable nickname matching
    df["player_id"] = df["player_id"].astype(str)

    # --------------------------------------------------
    # Position statistics
    # --------------------------------------------------

    position_stats = (
        df.groupby(
            ["player_id", "player_name", "end_position"],
            as_index=False
        )
        .agg(position_minutes=("minutes_played", "sum"))
    )

    position_stats = position_stats.sort_values(
        ["player_id", "position_minutes", "end_position"],
        ascending=[True, False, True]
    )

    dominant_positions = (
        position_stats
        .drop_duplicates(subset=["player_id"])
        .rename(columns={"end_position": "position"})
        [["player_id", "position", "position_minutes"]]
    )

    # Position breakdown
    position_stats["position_entry"] = position_stats.apply(
        lambda row: (
            f"{row['end_position']}: "
            f"{row['position_minutes']:.1f}"
        ),
        axis=1
    )

    position_breakdown = (
        position_stats
        .groupby("player_id")["position_entry"]
        .agg("; ".join)
        .reset_index(name="positions_breakdown")
    )

    # --------------------------------------------------
    # Most played team
    # --------------------------------------------------

    team_stats = (
        df.groupby(
            ["player_id", "team"],
            as_index=False
        )
        .agg(team_minutes=("minutes_played", "sum"))
        .sort_values(
            ["player_id", "team_minutes", "team"],
            ascending=[True, False, True]
        )
    )

    most_played_team = (
        team_stats
        .drop_duplicates(subset=["player_id"])
        [["player_id", "team", "team_minutes"]]
        .rename(columns={
            "team": "most_played_team",
            "team_minutes": "most_played_team_minutes"
        })
    )

    # --------------------------------------------------
    # Most played competition
    # --------------------------------------------------

    competition_stats = (
        df.groupby(
            ["player_id", "competition_name"],
            as_index=False
        )
        .agg(competition_minutes=("minutes_played", "sum"))
        .sort_values(
            ["player_id", "competition_minutes", "competition_name"],
            ascending=[True, False, True]
        )
    )

    most_played_competition = (
        competition_stats
        .drop_duplicates(subset=["player_id"])
        [["player_id", "competition_name", "competition_minutes"]]
        .rename(columns={
            "competition_name": "most_played_competition",
            "competition_minutes": "most_played_competition_minutes"
        })
    )

    # --------------------------------------------------
    # General player statistics
    # --------------------------------------------------

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

    # Nicknames from lineup JSON files
    player_stats["nick_name"] = (
        player_stats["player_id"].map(nickname_lookup)
    )

    # --------------------------------------------------
    # Combine all statistics
    # --------------------------------------------------

    # IMPORTANT: initialize result before merging
    result = player_stats.copy()

    result = result.merge(
        dominant_positions,
        on="player_id",
        how="left"
    )

    result = result.merge(
        position_breakdown,
        on="player_id",
        how="left"
    )

    result = result.merge(
        most_played_team,
        on="player_id",
        how="left"
    )

    result = result.merge(
        most_played_competition,
        on="player_id",
        how="left"
    )

    # --------------------------------------------------
    # Round minutes
    # --------------------------------------------------

    minute_columns = [
        "total_minutes",
        "position_minutes",
        "most_played_team_minutes",
        "most_played_competition_minutes",
    ]

    for column in minute_columns:
        result[column] = result[column].round(2)

    # --------------------------------------------------
    # Final columns
    # --------------------------------------------------

    result = result[
        [
            "player_id",
            "player_name",
            "nick_name",
            "most_played_team",
            "most_played_team_minutes",
            "most_played_competition",
            "most_played_competition_minutes",
            "total_minutes",
            "matches_played",
            "position",
            "position_minutes",
            "positions_breakdown",
        ]
    ]

    result = result.sort_values(
        "total_minutes",
        ascending=False
    ).reset_index(drop=True)

    # Diagnostics
    matched_nicknames = result["nick_name"].notna().sum()

    print(
        f"Players with nicknames: "
        f"{matched_nicknames}/{len(result)}"
    )

    print(
        f"Players with unknown competition: "
        f"{(result['most_played_competition'] == 'Unknown').sum()}"
    )

    return result



# --------------------------------------------------
# RUN
# --------------------------------------------------

events_directory = "open-data-projects/data/events/"
matches_directory = "open-data-projects/data/matches/"
lineups_directory = "open-data-projects/data/lineups/"

players = aggregate_all_matches(
    events_dir=events_directory,
    matches_dir=matches_directory,
    lineups_dir=lineups_directory
)

# Save aggregated player statistics
output_file = Path(
    "open-data-projects/data/processed/players_positions.csv"
)

output_file.parent.mkdir(parents=True, exist_ok=True)

players.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(players.head(20).to_string(index=False))
print(f"\nTotal unique players: {len(players)}")
