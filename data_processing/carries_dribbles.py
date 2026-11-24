
import json
import math
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

EVENTS_DIR = Path("open-data-projects/data/events")
OUTPUT_FILE = Path(
    "open-data-projects/data/processed/carries_dribbles.csv"
)

# Descriptive carry-distance thresholds, in StatsBomb pitch units.
LONG_CARRY_DISTANCE = 10.0
VERY_LONG_CARRY_DISTANCE = 20.0


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_name(value):
    """Extract a name from a StatsBomb object."""
    if isinstance(value, dict):
        return value.get("name")
    return value


def get_id(value):
    """Extract an ID from a StatsBomb object."""
    if isinstance(value, dict):
        return value.get("id")
    return value


def valid_location(location):
    """Check whether a StatsBomb location contains x and y."""
    return (
        isinstance(location, (list, tuple))
        and len(location) >= 2
        and location[0] is not None
        and location[1] is not None
    )


def carry_distance(start, end):
    """Calculate straight-line carry distance."""
    if not valid_location(start) or not valid_location(end):
        return None

    return math.dist(
        (float(start[0]), float(start[1])),
        (float(end[0]), float(end[1]))
    )


def is_carry_success(event, next_event):
    """
    Estimate carry success through possession retention.

    A carry is classified as successful if the next event:
    - belongs to the same possession ID, and
    - belongs to the same possession team.

    Returns:
        True  - possession retained
        False - possession not retained
        None  - cannot classify
    """

    if not next_event:
        return None

    possession_id = event.get("possession")
    next_possession_id = next_event.get("possession")

    team_id = get_id(event.get("possession_team"))
    next_team_id = get_id(next_event.get("possession_team"))

    if (
        possession_id is None
        or next_possession_id is None
        or team_id is None
        or next_team_id is None
    ):
        return None

    return (
        possession_id == next_possession_id
        and team_id == next_team_id
    )


# ---------------------------------------------------------
# PROCESS ONE MATCH
# ---------------------------------------------------------

def process_match(match_file):

    with open(match_file, "r", encoding="utf-8") as f:
        events = json.load(f)

    stats = defaultdict(lambda: {
        "player_name": None,

        # Carries
        "carries": 0,
        "successful_carries": 0,
        "classified_carries": 0,
        "carries_with_distance": 0,
        "total_carry_distance": 0.0,
        "long_carries_10m": 0,
        "very_long_carries_20m": 0,

        # Dribbles
        "dribbles": 0,
        "completed_dribbles": 0,
        "incomplete_dribbles": 0,
        "dribbles_with_known_outcome": 0,
        "nutmegs": 0,
        "successful_nutmegs": 0,
    })

    for i, event in enumerate(events):

        event_type = get_name(event.get("type"))

        if event_type not in {"Carry", "Dribble"}:
            continue

        player = event.get("player")

        if not player:
            continue

        player_id = player.get("id")
        player_name = player.get("name")

        if player_id is None:
            continue

        player_stats = stats[player_id]
        player_stats["player_name"] = player_name

        # ---------------------------------------------
        # CARRIES
        # ---------------------------------------------

        if event_type == "Carry":

            player_stats["carries"] += 1

            next_event = (
                events[i + 1]
                if i + 1 < len(events)
                else None
            )

            success = is_carry_success(event, next_event)

            if success is not None:
                player_stats["classified_carries"] += 1

                if success:
                    player_stats["successful_carries"] += 1

            carry_data = event.get("carry", {})

            start = event.get("location")
            end = carry_data.get("end_location")

            distance = carry_distance(start, end)

            if distance is not None:

                player_stats["carries_with_distance"] += 1
                player_stats["total_carry_distance"] += distance

                if distance >= LONG_CARRY_DISTANCE:
                    player_stats["long_carries_10m"] += 1

                if distance >= VERY_LONG_CARRY_DISTANCE:
                    player_stats["very_long_carries_20m"] += 1

        # ---------------------------------------------
        # DRIBBLES
        # ---------------------------------------------

        elif event_type == "Dribble":

            player_stats["dribbles"] += 1

            dribble_data = event.get("dribble", {})

            outcome = get_name(
                dribble_data.get("outcome")
            )

            if outcome is not None:

                player_stats["dribbles_with_known_outcome"] += 1

                if outcome == "Complete":
                    player_stats["completed_dribbles"] += 1

                elif outcome == "Incomplete":
                    player_stats["incomplete_dribbles"] += 1

            # Nutmeg feature
            if dribble_data.get("nutmeg") is True:

                player_stats["nutmegs"] += 1

                if outcome == "Complete":
                    player_stats["successful_nutmegs"] += 1

    # ---------------------------------------------
    # MATCH-LEVEL RESULTS
    # ---------------------------------------------

    results = []

    for player_id, player_stats in stats.items():

        results.append({
            "player_id": player_id,
            "player_name": player_stats["player_name"],

            "carries": player_stats["carries"],
            "successful_carries": player_stats["successful_carries"],
            "classified_carries": player_stats["classified_carries"],
            "carries_with_distance": player_stats["carries_with_distance"],
            "total_carry_distance": player_stats["total_carry_distance"],
            "long_carries_10m": player_stats["long_carries_10m"],
            "very_long_carries_20m": player_stats["very_long_carries_20m"],

            "dribbles": player_stats["dribbles"],
            "completed_dribbles": player_stats["completed_dribbles"],
            "incomplete_dribbles": player_stats["incomplete_dribbles"],
            "dribbles_with_known_outcome": player_stats["dribbles_with_known_outcome"],
            "nutmegs": player_stats["nutmegs"],
            "successful_nutmegs": player_stats["successful_nutmegs"],
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
            carries=("carries", "sum"),
            successful_carries=("successful_carries", "sum"),
            classified_carries=("classified_carries", "sum"),
            carries_with_distance=("carries_with_distance", "sum"),
            total_carry_distance=("total_carry_distance", "sum"),
            long_carries_10m=("long_carries_10m", "sum"),
            very_long_carries_20m=("very_long_carries_20m", "sum"),

            dribbles=("dribbles", "sum"),
            completed_dribbles=("completed_dribbles", "sum"),
            incomplete_dribbles=("incomplete_dribbles", "sum"),
            dribbles_with_known_outcome=("dribbles_with_known_outcome", "sum"),
            nutmegs=("nutmegs", "sum"),
            successful_nutmegs=("successful_nutmegs", "sum"),
        )
    )

    # ---------------------------------------------
    # SUCCESS RATES
    # ---------------------------------------------

    result["carry_retention_pct"] = (
        result["successful_carries"]
        .div(result["classified_carries"].replace(0, float("nan")))
        .mul(100)
        .round(2)
    )

    result["dribble_success_pct"] = (
        result["completed_dribbles"]
        .div(result["dribbles_with_known_outcome"].replace(0, float("nan")))
        .mul(100)
        .round(2)
    )

    # ---------------------------------------------
    # DISTANCE FEATURES
    # ---------------------------------------------

    result["average_carry_distance"] = (
        result["total_carry_distance"]
        .div(result["carries_with_distance"].replace(0, float("nan")))
        .round(2)
    )

    result["total_carry_distance"] = (
        result["total_carry_distance"].round(2)
    )

    # Share of carries that cover at least 10 or 20 units.
    result["long_carries_10m_pct"] = (
        result["long_carries_10m"]
        .div(result["carries_with_distance"].replace(0, float("nan")))
        .mul(100)
        .round(2)
    )

    result["very_long_carries_20m_pct"] = (
        result["very_long_carries_20m"]
        .div(result["carries_with_distance"].replace(0, float("nan")))
        .mul(100)
        .round(2)
    )

    # ---------------------------------------------
    # NUTMEG FEATURES
    # ---------------------------------------------

    result["successful_nutmeg_pct"] = (
        result["successful_nutmegs"]
        .div(result["nutmegs"].replace(0, float("nan")))
        .mul(100)
        .round(2)
    )

    # ---------------------------------------------
    # SORT
    # ---------------------------------------------

    result = result.sort_values(
        ["carries", "dribbles"],
        ascending=False
    ).reset_index(drop=True)

    return result


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

match_data = process_all_matches(EVENTS_DIR)

player_statistics = aggregate_player_statistics(match_data)

# Create output directory if it does not exist.
Path(OUTPUT_FILE).parent.mkdir(
    parents=True,
    exist_ok=True
)

player_statistics.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nTop 20 players by carries:")
print(
    player_statistics.head(20).to_string(index=False)
)

print(f"\nTotal players: {len(player_statistics)}")
print(f"Saved to: {OUTPUT_FILE}")
