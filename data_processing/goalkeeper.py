
import json
import re
import math
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

EVENTS_DIR = Path("open-data-projects/data/events")

OUTPUT_FILE = Path(
    "open-data-projects/data/processed/goalkeepers.csv"
)

ON_TARGET_OUTCOMES = {
    "goal",
    "saved",
    "saved to post",
}

SAVED_OUTCOMES = {
    "saved",
    "saved to post",
}


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_name(value):
    if isinstance(value, dict):
        return value.get("name")
    return value


def get_id(value):
    if isinstance(value, dict):
        return value.get("id")
    return value


def normalize(value):
    name = get_name(value)

    if name is None:
        return None

    return str(name).strip().lower()


def make_slug(value):
    """Convert event labels into safe column-name suffixes."""
    if value is None:
        return "unknown"

    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)

    return value.strip("_") or "unknown"


def get_xg(shot):
    try:
        return float(shot.get("statsbomb_xg") or 0)
    except (TypeError, ValueError):
        return 0.0


def valid_location(location):
    return (
        isinstance(location, (list, tuple))
        and len(location) >= 2
        and location[0] is not None
        and location[1] is not None
    )


def location_distance(start, end):
    if not valid_location(start) or not valid_location(end):
        return None

    return math.dist(
        (float(start[0]), float(start[1])),
        (float(end[0]), float(end[1]))
    )


def is_on_target(outcome):
    return normalize(outcome) in ON_TARGET_OUTCOMES


def is_saved(outcome):
    return normalize(outcome) in SAVED_OUTCOMES


def get_freeze_frame_goalkeeper(shot):
    """
    Find the opposing goalkeeper in a shot freeze frame.

    The freeze frame is interpreted relative to the shooting team:
    teammate=False identifies an opponent.
    """

    freeze_frame = shot.get("freeze_frame", [])

    if not isinstance(freeze_frame, list):
        return None

    for player_data in freeze_frame:

        if not isinstance(player_data, dict):
            continue

        position = normalize(player_data.get("position"))

        if position != "goalkeeper":
            continue

        if player_data.get("teammate") is False:

            player = player_data.get("player")

            if isinstance(player, dict):
                return {
                    "id": player.get("id"),
                    "name": player.get("name"),
                }

    return None


# ---------------------------------------------------------
# STATISTICS TEMPLATE
# ---------------------------------------------------------

def create_player_stats():

    return {
        # SHOT STOPPING
        "shots_on_target_faced": 0,
        "saves": 0,
        "goals_conceded": 0,
        "xg_against_all_attributed_shots": 0.0,
        "xg_against_on_target_shots": 0.0,

        # PENALTIES
        "penalties_faced": 0,
        "penalties_saved": 0,
        "penalty_goals_conceded": 0,

        # GOALKEEPER ACTIONS
        "gk_actions": 0,
        "gk_actions_with_end_location": 0,
        "gk_actions_with_body_part": 0,
        "gk_actions_with_technique": 0,
        "gk_actions_with_position": 0,

        # DISTRIBUTION / MOVEMENT
        "total_action_distance": 0.0,
        "actions_with_distance": 0,
        "end_location_x_sum": 0.0,
        "end_location_y_sum": 0.0,

        # DYNAMIC COUNTERS
        "action_types": Counter(),
        "action_outcomes": Counter(),
        "action_techniques": Counter(),
        "action_positions": Counter(),
        "action_body_parts": Counter(),
    }


# ---------------------------------------------------------
# PROCESS ONE MATCH
# ---------------------------------------------------------

def process_match(match_file):

    with open(match_file, "r", encoding="utf-8") as f:
        events = json.load(f)

    stats = defaultdict(create_player_stats)
    player_names = {}

    # -----------------------------------------------------
    # 1. INDEX GOALKEEPER EVENTS
    # -----------------------------------------------------

    goalkeeper_event_players = {}

    for event in events:

        if get_name(event.get("type")) != "Goal Keeper":
            continue

        player = event.get("player")

        if not player:
            continue

        player_id = player.get("id")

        if player_id is None:
            continue

        player_names[player_id] = player.get("name")

        event_id = event.get("id")

        if event_id:
            goalkeeper_event_players[event_id] = player_id

    # -----------------------------------------------------
    # 2. PROCESS GOALKEEPER EVENTS
    # -----------------------------------------------------

    for event in events:

        if get_name(event.get("type")) != "Goal Keeper":
            continue

        player = event.get("player")

        if not player:
            continue

        player_id = player.get("id")

        if player_id is None:
            continue

        player_stats = stats[player_id]

        goalkeeper = event.get("goalkeeper", {})

        player_stats["gk_actions"] += 1

        # Action type
        action_type = get_name(goalkeeper.get("type"))

        if action_type:
            player_stats["action_types"][action_type] += 1

        # Outcome
        outcome = get_name(goalkeeper.get("outcome"))

        if outcome:
            player_stats["action_outcomes"][outcome] += 1

        # Technique
        technique = get_name(goalkeeper.get("technique"))

        if technique:
            player_stats["gk_actions_with_technique"] += 1
            player_stats["action_techniques"][technique] += 1

        # Position
        position = get_name(goalkeeper.get("position"))

        if position:
            player_stats["gk_actions_with_position"] += 1
            player_stats["action_positions"][position] += 1

        # Body part
        body_part = get_name(goalkeeper.get("body_part"))

        if body_part:
            player_stats["gk_actions_with_body_part"] += 1
            player_stats["action_body_parts"][body_part] += 1

        # End location
        end_location = goalkeeper.get("end_location")

        if valid_location(end_location):

            player_stats["gk_actions_with_end_location"] += 1

            x = float(end_location[0])
            y = float(end_location[1])

            player_stats["end_location_x_sum"] += x
            player_stats["end_location_y_sum"] += y

            distance = location_distance(
                event.get("location"),
                end_location
            )

            if distance is not None:
                player_stats["total_action_distance"] += distance
                player_stats["actions_with_distance"] += 1

    # -----------------------------------------------------
    # 3. PROCESS SHOTS AND ATTRIBUTE THEM TO GOALKEEPERS
    # -----------------------------------------------------

    unattributed_on_target_shots = 0
    unattributed_penalties = 0

    for event in events:

        if get_name(event.get("type")) != "Shot":
            continue

        shot = event.get("shot", {})

        outcome = shot.get("outcome")
        outcome_name = normalize(outcome)

        shot_type = normalize(shot.get("type"))
        is_penalty = shot_type == "penalty"

        on_target = is_on_target(outcome)
        saved = is_saved(outcome)
        goal = outcome_name == "goal"

        xg = get_xg(shot)

        # Find goalkeeper through related event IDs first.
        goalkeeper_id = None

        related_events = event.get("related_events", [])

        if isinstance(related_events, list):

            for related_id in related_events:

                if related_id in goalkeeper_event_players:
                    goalkeeper_id = goalkeeper_event_players[related_id]
                    break

        # Fallback: goalkeeper in shot freeze frame.
        if goalkeeper_id is None:

            freeze_frame_keeper = get_freeze_frame_goalkeeper(shot)

            if freeze_frame_keeper:
                goalkeeper_id = freeze_frame_keeper.get("id")

                if goalkeeper_id is not None:
                    player_names[goalkeeper_id] = (
                        freeze_frame_keeper.get("name")
                    )

        # If goalkeeper cannot be identified, do not guess.
        if goalkeeper_id is None:

            if on_target:
                unattributed_on_target_shots += 1

            if is_penalty:
                unattributed_penalties += 1

            continue

        player_stats = stats[goalkeeper_id]

        # -------------------------------------------------
        # ALL ATTRIBUTED SHOTS
        # -------------------------------------------------

        player_stats[
            "xg_against_all_attributed_shots"
        ] += xg

        # -------------------------------------------------
        # SHOTS ON TARGET
        # -------------------------------------------------

        if on_target:

            player_stats["shots_on_target_faced"] += 1

            player_stats[
                "xg_against_on_target_shots"
            ] += xg

            if saved:
                player_stats["saves"] += 1

            if goal:
                player_stats["goals_conceded"] += 1

        # -------------------------------------------------
        # PENALTIES
        # -------------------------------------------------

        if is_penalty:

            player_stats["penalties_faced"] += 1

            if saved:
                player_stats["penalties_saved"] += 1

            if goal:
                player_stats["penalty_goals_conceded"] += 1

    # -----------------------------------------------------
    # 4. MATCH RESULTS
    # -----------------------------------------------------

    results = []

    for player_id, player_stats in stats.items():

        row = {
            "player_id": player_id,
            "player_name": player_names.get(player_id),
        }

        # Static metrics
        for key, value in player_stats.items():

            if isinstance(value, Counter):
                continue

            row[key] = value

        # Dynamic event counters
        for action, count in player_stats["action_types"].items():
            row[f"action_type_{make_slug(action)}"] = count

        for outcome, count in player_stats["action_outcomes"].items():
            row[f"action_outcome_{make_slug(outcome)}"] = count

        for technique, count in player_stats["action_techniques"].items():
            row[f"action_technique_{make_slug(technique)}"] = count

        for position, count in player_stats["action_positions"].items():
            row[f"action_position_{make_slug(position)}"] = count

        for body_part, count in player_stats["action_body_parts"].items():
            row[f"action_body_part_{make_slug(body_part)}"] = count

        results.append(row)

    diagnostics = {
        "unattributed_on_target_shots": unattributed_on_target_shots,
        "unattributed_penalties": unattributed_penalties,
    }

    return results, diagnostics


# ---------------------------------------------------------
# PROCESS ALL MATCHES
# ---------------------------------------------------------

def process_all_matches(events_dir):

    match_files = sorted(
        Path(events_dir).rglob("*.json")
    )

    print(f"Found {len(match_files)} event files.")

    all_results = []
    total_unattributed_on_target = 0
    total_unattributed_penalties = 0

    for i, match_file in enumerate(match_files, start=1):

        try:
            match_results, diagnostics = process_match(match_file)

            all_results.extend(match_results)

            total_unattributed_on_target += (
                diagnostics["unattributed_on_target_shots"]
            )

            total_unattributed_penalties += (
                diagnostics["unattributed_penalties"]
            )

        except Exception as e:
            print(f"Error processing {match_file}: {e}")

        if i % 100 == 0:
            print(f"Processed {i}/{len(match_files)} matches")

    print(f"Processed {len(match_files)} event files.")

    print(
        "Unattributed shots on target:",
        total_unattributed_on_target
    )

    print(
        "Unattributed penalties:",
        total_unattributed_penalties
    )

    return pd.DataFrame(all_results)


# ---------------------------------------------------------
# AGGREGATE BY PLAYER
# ---------------------------------------------------------

def aggregate_player_statistics(df):

    if df.empty:
        return df

    # Identify numeric and dynamic counter columns.
    counter_columns = [
        column for column in df.columns
        if column.startswith((
            "action_type_",
            "action_outcome_",
            "action_technique_",
            "action_position_",
            "action_body_part_",
        ))
    ]

    numeric_columns = [
        column for column in df.columns
        if column not in {"player_id", "player_name"}
    ]

    result = (
        df.groupby(
            ["player_id", "player_name"],
            as_index=False
        )[numeric_columns]
        .sum()
    )

    # -----------------------------------------------------
    # SHOT STOPPING METRICS
    # -----------------------------------------------------

    result["save_pct"] = (
        result["saves"]
        .div(
            result["shots_on_target_faced"].replace(
                0, float("nan")
            )
        )
        .mul(100)
        .round(2)
    )

    result["goals_conceded_per_shot_on_target"] = (
        result["goals_conceded"]
        .div(
            result["shots_on_target_faced"].replace(
                0, float("nan")
            )
        )
        .round(3)
    )

    result["xg_per_shot_faced"] = (
        result["xg_against_all_attributed_shots"]
        .div(
            result["shots_on_target_faced"].replace(
                0, float("nan")
            )
        )
        .round(3)
    )

    # -----------------------------------------------------
    # PENALTY METRICS
    # -----------------------------------------------------

    result["penalty_save_pct"] = (
        result["penalties_saved"]
        .div(
            result["penalties_faced"].replace(
                0, float("nan")
            )
        )
        .mul(100)
        .round(2)
    )

    # -----------------------------------------------------
    # ACTION METRICS
    # -----------------------------------------------------

    result["actions_per_shot_on_target"] = (
        result["gk_actions"]
        .div(
            result["shots_on_target_faced"].replace(
                0, float("nan")
            )
        )
        .round(3)
    )

    result["average_action_distance"] = (
        result["total_action_distance"]
        .div(
            result["actions_with_distance"].replace(
                0, float("nan")
            )
        )
        .round(2)
    )

    result["average_end_location_x"] = (
        result["end_location_x_sum"]
        .div(
            result["gk_actions_with_end_location"].replace(
                0, float("nan")
            )
        )
        .round(2)
    )

    result["average_end_location_y"] = (
        result["end_location_y_sum"]
        .div(
            result["gk_actions_with_end_location"].replace(
                0, float("nan")
            )
        )
        .round(2)
    )

    # -----------------------------------------------------
    # ROUND XG
    # -----------------------------------------------------

    xg_columns = [
        "xg_against_all_attributed_shots",
        "xg_against_on_target_shots",
    ]

    result[xg_columns] = result[xg_columns].round(3)

    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    result = result.sort_values(
        ["shots_on_target_faced", "gk_actions"],
        ascending=False
    ).reset_index(drop=True)

    return result


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

match_data = process_all_matches(EVENTS_DIR)

goalkeeper_statistics = aggregate_player_statistics(match_data)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)
# ---------------------------------------------------------
# SELECT FINAL COLUMNS
# ---------------------------------------------------------

FINAL_COLUMNS = [
    "player_id",
    "player_name",

    "shots_on_target_faced",
    "saves",
    "goals_conceded",
    "save_pct",
    "xg_against_all_attributed_shots",
    "xg_against_on_target_shots",

    "penalties_faced",
    "penalties_saved",
    "penalty_goals_conceded",
    "penalty_save_pct",

    "gk_actions",
    "actions_per_shot_on_target",

    "action_type_claim",
    "action_type_punch",
    "action_type_keeper_sweeper",
    "action_type_smother",
    "action_type_collected",

    "action_outcome_fail",
    "action_outcome_punched_out",
]

# Add missing action columns as zero.
for column in FINAL_COLUMNS:
    if column not in goalkeeper_statistics.columns:
        goalkeeper_statistics[column] = 0

goalkeeper_statistics = goalkeeper_statistics[FINAL_COLUMNS]

goalkeeper_statistics.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nTop 20 goalkeepers by shots on target faced:")

display_columns = [
    "player_name",
    "shots_on_target_faced",
    "saves",
    "goals_conceded",
    "save_pct",
    "xg_against_all_attributed_shots",
    "penalties_faced",
    "penalties_saved",
    "penalty_goals_conceded",
    "gk_actions",
    "average_action_distance",
]

display_columns = [
    col for col in display_columns
    if col in goalkeeper_statistics.columns
]

print(
    goalkeeper_statistics[display_columns]
    .head(20)
    .to_string(index=False)
)

print(f"\nTotal goalkeepers: {len(goalkeeper_statistics)}")
print(f"Saved to: {OUTPUT_FILE}")
