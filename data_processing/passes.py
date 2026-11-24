
import json
import math
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

EVENTS_DIR = Path("open-data-projects/data/events")

OUTPUT_FILE = "open-data-projects/data/processed/passes.csv"

# Pass types excluded from regular passes.
# Corners are counted separately.
EXCLUDED_PASS_TYPES = {
    "Corner",
    "Free Kick",
    "Goal Kick",
    "Kick Off",
    "Throw-in",
    "Recovery",
    "Interception",
}

# ---------------------------------------------------------
# LINE-BREAKING PASS CONFIGURATION
# ---------------------------------------------------------

GOAL_CENTER = (120.0, 40.0)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def get_name(value):
    """Extract a name from a StatsBomb object."""
    if isinstance(value, dict):
        return value.get("name")
    return value


def is_completed(event):
    """
    A StatsBomb pass without an outcome is considered completed.
    A pass with an outcome is unsuccessful.
    """
    return event.get("pass", {}).get("outcome") is None


# ---------------------------------------------------------
# LINE-BREAKING PASS CONFIGURATION
# ---------------------------------------------------------

GOAL_CENTER = (120.0, 40.0)

# Adjustable approximation: maximum difference between
# defenders' x-coordinates to form a defensive pair.
DEFENDER_PAIR_X_GAP = 10.0


# ---------------------------------------------------------
# GEOMETRY HELPERS
# ---------------------------------------------------------

def distance_to_goal(location):
    """Euclidean distance from a location to the opponent's goal."""
    return math.dist(location[:2], GOAL_CENTER)

# ---------------------------------------------------------
# LINE-BREAKING PASS DETECTION
# ---------------------------------------------------------

MIN_GOAL_DISTANCE_REDUCTION = 0.10


def is_line_breaking_pass(event):
    """
    Line-breaking pass proxy without 360 data.

    Requirements:
    1. Completed pass.
    2. Open-play pass.
    3. Not high-lofted.
    4. Pass endpoint is at least 10% closer to
       the opponent's goal than the starting point.

    NOTE:
    Without defender positions, this cannot confirm
    whether the pass actually breaks a defensive line.
    """

    pass_data = event.get("pass", {})

    # 1. Completed pass
    if not is_completed(event):
        return False

    # 2. Open-play pass only
    if pass_data.get("type") is not None:
        return False

    # 3. Exclude high passes
    pass_height = get_name(pass_data.get("height"))

    if pass_height == "High":
        return False

    # 4. Check progression towards goal
    start = event.get("location")
    end = pass_data.get("end_location")

    if not start or not end or len(start) < 2 or len(end) < 2:
        return False

    start = (float(start[0]), float(start[1]))
    end = (float(end[0]), float(end[1]))

    start_distance = distance_to_goal(start)
    end_distance = distance_to_goal(end)

    if start_distance == 0:
        return False

    progression = (
        start_distance - end_distance
    ) / start_distance

    return progression >= MIN_GOAL_DISTANCE_REDUCTION

# ---------------------------------------------------------
# PROCESS ONE MATCH
# ---------------------------------------------------------

def process_match(match_file):
    with open(match_file, "r", encoding="utf-8") as f:
        events = json.load(f)

    # Accumulate statistics by player within this match.
    stats = defaultdict(lambda: {
        "player_name": None,
        "regular_passes": 0,
        "completed_passes": 0,
        "corners": 0,
        "completed_corners": 0,
        "forward_passes": 0,
        "line_breaking_passes": 0,
        "key_pass_ids": set(),
        "xag": 0.0,
    })

    # Map pass event ID to the player who made it.
    pass_to_player = {}

    # Store shots for processing after all passes are read.
    shots = []

    for event in events:

        event_type = get_name(event.get("type"))

        # ---------------------------------------------
        # SHOTS: collect xG and key_pass_id
        # ---------------------------------------------

        if event_type == "Shot":
            shot = event.get("shot", {})

            key_pass_id = shot.get("key_pass_id")

            if key_pass_id:
                shots.append({
                    "key_pass_id": key_pass_id,
                    "xg": shot.get("statsbomb_xg", 0) or 0,
                })

            continue

        # ---------------------------------------------
        # PASSES
        # ---------------------------------------------

        if event_type != "Pass":
            continue

        player = event.get("player")

        if not player:
            continue

        player_id = player["id"]
        player_name = player["name"]

        player_stats = stats[player_id]
        player_stats["player_name"] = player_name

        pass_data = event.get("pass", {})

        pass_id = event.get("id")

        if pass_id:
            pass_to_player[pass_id] = player_id

        pass_type = get_name(pass_data.get("type"))

        completed = is_completed(event)

        # ---------------------------------------------
        # CORNERS
        # ---------------------------------------------

        if pass_type == "Corner":

            player_stats["corners"] += 1

            if completed:
                player_stats["completed_corners"] += 1

            continue

        # ---------------------------------------------
        # REGULAR PASSES
        # ---------------------------------------------

        if pass_type in EXCLUDED_PASS_TYPES:
            continue

        player_stats["regular_passes"] += 1

        if completed:
            player_stats["completed_passes"] += 1

        # ---------------------------------------------
        # FORWARD PASSES
        # ---------------------------------------------

        angle = pass_data.get("angle")

        if angle is not None:
            # Angle is in radians.
            # This classifies passes with a positive x component
            # as forward, assuming the angle uses the attacking
            # direction as its positive x-axis.
            if abs(angle) < math.pi / 2:
                player_stats["forward_passes"] += 1

        
        # ---------------------------------------------
        # LINE-BREAKING PASSES
        # ---------------------------------------------

        if is_line_breaking_pass(event):
            player_stats["line_breaking_passes"] += 1


    # ---------------------------------------------
    # KEY PASSES AND xAG
    # ---------------------------------------------

    for shot in shots:

        key_pass_id = shot["key_pass_id"]

        player_id = pass_to_player.get(key_pass_id)

        if player_id is None:
            continue

        player_stats = stats[player_id]

        player_stats["key_pass_ids"].add(key_pass_id)
        player_stats["xag"] += shot["xg"]

    # Convert sets to counts and return match-level records.
    results = []

    for player_id, player_stats in stats.items():

        results.append({
            "player_id": player_id,
            "player_name": player_stats["player_name"],
            "regular_passes": player_stats["regular_passes"],
            "completed_passes": player_stats["completed_passes"],
            "corners": player_stats["corners"],
            "completed_corners": player_stats["completed_corners"],
            "forward_passes": player_stats["forward_passes"],
            "key_passes": len(player_stats["key_pass_ids"]),
            "xag": player_stats["xag"],
            "line_breaking_passes": player_stats["line_breaking_passes"],
        })

    return results


# ---------------------------------------------------------
# PROCESS ALL MATCHES
# ---------------------------------------------------------

def process_all_matches(events_dir):

    events_dir = Path(events_dir)

    match_files = sorted(events_dir.rglob("*.json"))

    print(f"Found {len(match_files)} event files.")

    all_results = []

    for i, match_file in enumerate(match_files, start=1):

        try:
            match_results = process_match(match_file)
            all_results.extend(match_results)

        except Exception as e:
            print(f"Error processing {match_file}: {e}")

        if i % 100 == 0:
            print(f"Processed {i}/{len(match_files)} matches")

    print(f"Processed {len(match_files)} event files.")
    return pd.DataFrame(all_results)


# ---------------------------------------------------------
# AGGREGATE BY PLAYER
# ---------------------------------------------------------

def aggregate_player_statistics(df):

    if df.empty:
        return df

    result = (
        df.groupby("player_id", as_index=False)
        .agg(
            player_name=(
                "player_name",
                lambda s: next(
                    (name for name in s if pd.notna(name)),
                    None
                )
            ),
            regular_passes=("regular_passes", "sum"),
            completed_passes=("completed_passes", "sum"),
            corners=("corners", "sum"),
            completed_corners=("completed_corners", "sum"),
            forward_passes=("forward_passes", "sum"),
            key_passes=("key_passes", "sum"),
            xag=("xag", "sum"),
            line_breaking_passes=("line_breaking_passes", "sum"),
        )
    )

    
    # Completion percentages
    result["pass_completion_pct"] = (
        result["completed_passes"]
        .div(result["regular_passes"].replace(0, float("nan")))
        .mul(100)
    )

    result["corner_completion_pct"] = (
        result["completed_corners"]
        .div(result["corners"].replace(0, float("nan")))
        .mul(100)
    )

    # Round values safely
    result["pass_completion_pct"] = (
        result["pass_completion_pct"].round(2)
    )

    result["corner_completion_pct"] = (
        result["corner_completion_pct"].round(2)
    )

    result["xag"] = result["xag"].round(4)


    result = result.sort_values(
        "regular_passes",
        ascending=False
    ).reset_index(drop=True)

    return result


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

match_data = process_all_matches(EVENTS_DIR)

player_statistics = aggregate_player_statistics(match_data)

player_statistics.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nTop 20 players by regular passes:")
print(player_statistics.head(20).to_string(index=False))

print(f"\nTotal players: {len(player_statistics)}")
print(f"Saved to: {OUTPUT_FILE}")
