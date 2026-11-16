
import json
import math
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

EVENTS_DIR = Path("open-data-projects/data/events")
THREE_SIXTY_DIR = EVENTS_DIR.parent / "three-sixty"

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

# Maximum difference between defenders' x coordinates
# for them to be considered a close defensive pair.
# This is an adjustable approximation, not a published
# StatsBomb threshold.
DEFENDER_PAIR_X_GAP = 10.0


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


def orientation(a, b, c):
    """Cross product used for segment intersection."""
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def segments_intersect(a, b, c, d):
    """Return True if line segments AB and CD intersect."""

    eps = 1e-9

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    if (
        ((o1 > eps and o2 < -eps) or
         (o1 < -eps and o2 > eps))
        and
        ((o3 > eps and o4 < -eps) or
         (o3 < -eps and o4 > eps))
    ):
        return True

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and
            min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
        )

    if abs(o1) <= eps and on_segment(a, c, b):
        return True
    if abs(o2) <= eps and on_segment(a, d, b):
        return True
    if abs(o3) <= eps and on_segment(c, a, d):
        return True
    if abs(o4) <= eps and on_segment(c, b, d):
        return True

    return False


# ---------------------------------------------------------
# LINE-BREAKING PASS DETECTION
# ---------------------------------------------------------

def is_line_breaking_pass(event, freeze_frame):
    """
    Approximation of StatsBomb's published line-breaking rules.

    Requirements:
    1. Completed open-play pass.
    2. Not a high-lofted pass.
    3. At least 10% closer to the opponent's goal.
    4. Intersects a close defensive pair OR finishes
       goal-side of a defensive pair.
    """

    pass_data = event.get("pass", {})

    # 1. Completed pass
    if not is_completed(event):
        return False

    # 2. Open-play pass only
    if pass_data.get("type") is not None:
        return False

    # 3. Exclude high-lofted passes
    pass_height = get_name(pass_data.get("height"))

    if pass_height == "High":
        return False

    # 4. Check minimum progression
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

    progression = (start_distance - end_distance) / start_distance

    if progression < 0.10:
        return False

    # 5. Extract visible opponents from the freeze frame
    if not freeze_frame:
        return False

    defenders = []

    for player in freeze_frame:

        if player.get("teammate", True):
            continue

        if player.get("actor", False):
            continue

        location = player.get("location")

        if not location or len(location) < 2:
            continue

        defenders.append((
            float(location[0]),
            float(location[1])
        ))

    if len(defenders) < 2:
        return False

    # 6. Evaluate all possible defensive pairs
    for i in range(len(defenders)):

        for j in range(i + 1, len(defenders)):

            defender_1 = defenders[i]
            defender_2 = defenders[j]

            x_gap = abs(defender_1[0] - defender_2[0])

            if x_gap > DEFENDER_PAIR_X_GAP:
                continue

            # Condition A: pass intersects the defensive pair
            if segments_intersect(
                start,
                end,
                defender_1,
                defender_2
            ):
                return True

            # Condition B: pass finishes goal-side of the pair
            goal_side = (
                end[0] > max(defender_1[0], defender_2[0])
            )

            between_defenders = (
                min(defender_1[1], defender_2[1])
                <= end[1]
                <= max(defender_1[1], defender_2[1])
            )

            if goal_side and between_defenders:
                return True

    return False

# ---------------------------------------------------------
# THREE-SIXTY FILE LOOKUP
# ---------------------------------------------------------

def build_three_sixty_lookup(match_file):
    """
    Load three-sixty data and map:
        event_uuid -> freeze_frame

    The three-sixty file is expected to have the same
    filename as the corresponding events file.
    """

    three_sixty_file = THREE_SIXTY_DIR / match_file.name

    if not three_sixty_file.exists():
        return {}

    try:
        with open(three_sixty_file, "r", encoding="utf-8") as f:
            records = json.load(f)

    except (json.JSONDecodeError, OSError) as e:
        print(f"Could not read {three_sixty_file}: {e}")
        return {}

    return {
        record.get("event_uuid"): record.get("freeze_frame", [])
        for record in records
        if record.get("event_uuid")
    }


# ---------------------------------------------------------
# PROCESS ONE MATCH
# ---------------------------------------------------------

def process_match(match_file):
    with open(match_file, "r", encoding="utf-8") as f:
        events = json.load(f)

    three_sixty = build_three_sixty_lookup(match_file)

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

        freeze_frame = three_sixty.get(pass_id, [])

        if is_line_breaking_pass(event, freeze_frame):
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
        df.groupby(
            ["player_id", "player_name"],
            as_index=False
        )
        .agg(
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
