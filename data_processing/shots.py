
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

EVENTS_DIR = Path("open-data-projects/data/events")

OUTPUT_FILE = Path(
    "open-data-projects/data/processed/shots.csv"
)

# StatsBomb outcomes classified as shots on target.
ON_TARGET_OUTCOMES = {
    "goal",
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


def normalize(value):
    """Normalize StatsBomb names for reliable comparisons."""
    name = get_name(value)
    return str(name).strip().lower() if name is not None else None


def get_shot_xg(shot):
    """Return shot xG, defaulting to zero if unavailable."""
    try:
        return float(shot.get("statsbomb_xg") or 0)
    except (TypeError, ValueError):
        return 0.0


def is_on_target(outcome):
    return normalize(outcome) in ON_TARGET_OUTCOMES


# ---------------------------------------------------------
# PLAYER STATISTICS TEMPLATE
# ---------------------------------------------------------

def create_player_stats():
    return {
        # NON-PENALTY SHOTS
        "np_shots": 0,
        "np_xg": 0.0,
        "np_shots_on_target": 0,
        "np_goals": 0,

        # NON-PENALTY HEADED SHOTS
        "headed_shots": 0,
        "headed_xg": 0.0,
        "headed_shots_on_target": 0,
        "headed_goals": 0,

        # ONE-ON-ONE SHOTS (NON-PENALTY)
        "one_on_one_shots": 0,
        "one_on_one_xg": 0.0,
        "one_on_one_shots_on_target": 0,
        "one_on_one_goals": 0,

        # PENALTIES
        "penalties": 0,
        "penalty_goals": 0,
    }


# ---------------------------------------------------------
# PROCESS ONE MATCH
# ---------------------------------------------------------

def process_match(match_file):

    with open(match_file, "r", encoding="utf-8") as f:
        events = json.load(f)

    stats = defaultdict(create_player_stats)
    player_names = {}

    for event in events:

        if get_name(event.get("type")) != "Shot":
            continue

        player = event.get("player")

        if not player:
            continue

        player_id = player.get("id")
        player_name = player.get("name")

        if player_id is None:
            continue

        player_names[player_id] = player_name

        player_stats = stats[player_id]

        shot = event.get("shot", {})

        outcome = shot.get("outcome")
        outcome_name = normalize(outcome)

        shot_type = normalize(shot.get("type"))
        body_part = normalize(shot.get("body_part"))

        xg = get_shot_xg(shot)

        on_target = is_on_target(outcome)
        is_goal = outcome_name == "goal"
        is_penalty = shot_type == "penalty"
        is_headed = body_part == "head"
        is_one_on_one = shot.get("one_on_one") is True

        # -------------------------------------------------
        # PENALTIES
        # -------------------------------------------------

        if is_penalty:

            player_stats["penalties"] += 1

            if is_goal:
                player_stats["penalty_goals"] += 1

            continue

        # -------------------------------------------------
        # NON-PENALTY SHOTS
        # -------------------------------------------------

        player_stats["np_shots"] += 1
        player_stats["np_xg"] += xg

        if on_target:
            player_stats["np_shots_on_target"] += 1

        if is_goal:
            player_stats["np_goals"] += 1

        # -------------------------------------------------
        # HEADED NON-PENALTY SHOTS
        # -------------------------------------------------

        if is_headed:

            player_stats["headed_shots"] += 1
            player_stats["headed_xg"] += xg

            if on_target:
                player_stats["headed_shots_on_target"] += 1

            if is_goal:
                player_stats["headed_goals"] += 1

        # -------------------------------------------------
        # ONE-ON-ONE NON-PENALTY SHOTS
        # -------------------------------------------------

        if is_one_on_one:

            player_stats["one_on_one_shots"] += 1
            player_stats["one_on_one_xg"] += xg

            if on_target:
                player_stats["one_on_one_shots_on_target"] += 1

            if is_goal:
                player_stats["one_on_one_goals"] += 1

    # -----------------------------------------------------
    # MATCH RESULTS
    # -----------------------------------------------------

    results = []

    for player_id, player_stats in stats.items():

        row = {
            "player_id": player_id,
            "player_name": player_names.get(player_id),
        }

        row.update(player_stats)
        results.append(row)

    return results


# ---------------------------------------------------------
# PROCESS ALL MATCHES
# ---------------------------------------------------------

def process_all_matches(events_dir):

    match_files = sorted(
        Path(events_dir).rglob("*.json")
    )

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

    print(f"Finished processing {len(match_files)} matches.")

    return pd.DataFrame(all_results)


# ---------------------------------------------------------
# AGGREGATE BY PLAYER
# ---------------------------------------------------------

def aggregate_player_statistics(df):

    if df.empty:
        return df

    numeric_columns = [
        "np_shots",
        "np_xg",
        "np_shots_on_target",
        "np_goals",

        "headed_shots",
        "headed_xg",
        "headed_shots_on_target",
        "headed_goals",

        "one_on_one_shots",
        "one_on_one_xg",
        "one_on_one_shots_on_target",
        "one_on_one_goals",

        "penalties",
        "penalty_goals",
    ]

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
            **{
                column: (column, "sum")
                for column in numeric_columns
            }
        )
    )

    # -----------------------------------------------------
    # NON-PENALTY SHOT RATES
    # -----------------------------------------------------

    result["np_conversion_pct"] = (
        result["np_goals"]
        .div(result["np_shots"].replace(0, float("nan")))
        .mul(100)
        .round(2)
    )

    result["np_shots_on_target_pct"] = (
        result["np_shots_on_target"]
        .div(result["np_shots"].replace(0, float("nan")))
        .mul(100)
        .round(2)
    )

    result["np_xg_per_shot"] = (
        result["np_xg"]
        .div(result["np_shots"].replace(0, float("nan")))
        .round(3)
    )

    result["np_goals_minus_xg"] = (
        result["np_goals"] - result["np_xg"]
    ).round(3)

    # -----------------------------------------------------
    # HEADED SHOT RATES
    # -----------------------------------------------------

    result["headed_conversion_pct"] = (
        result["headed_goals"]
        .div(result["headed_shots"].replace(0, float("nan")))
        .mul(100)
        .round(2)
    )

    result["headed_shots_on_target_pct"] = (
        result["headed_shots_on_target"]
        .div(result["headed_shots"].replace(0, float("nan")))
        .mul(100)
        .round(2)
    )

    result["headed_xg_per_shot"] = (
        result["headed_xg"]
        .div(result["headed_shots"].replace(0, float("nan")))
        .round(3)
    )

    # ONE-ON-ONE RATES

    result["one_on_one_conversion_pct"] = (
        result["one_on_one_goals"]
        .div(result["one_on_one_shots"].replace(0, float("nan")))
        .mul(100)
        .round(2)
    )

    result["one_on_one_shots_on_target_pct"] = (
        result["one_on_one_shots_on_target"]
        .div(result["one_on_one_shots"].replace(0, float("nan")))
        .mul(100)
        .round(2)
    )

    result["one_on_one_xg_per_shot"] = (
        result["one_on_one_xg"]
        .div(result["one_on_one_shots"].replace(0, float("nan")))
        .round(3)
    )


    # PENALTY CONVERSION

    result["penalty_conversion_pct"] = (
        result["penalty_goals"]
        .div(result["penalties"].replace(0, float("nan")))
        .mul(100)
        .round(2)
    )

    # -----------------------------------------------------
    # ROUND XG
    # -----------------------------------------------------

    xg_columns = [
        "np_xg",
        "headed_xg",
        "one_on_one_xg",
    ]

    result[xg_columns] = result[xg_columns].round(3)

    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    result = result.sort_values(
        ["np_xg", "np_shots"],
        ascending=False
    ).reset_index(drop=True)

    return result


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

match_data = process_all_matches(EVENTS_DIR)

player_statistics = aggregate_player_statistics(match_data)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

player_statistics.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nTop 20 players by non-penalty xG:")

print(
    player_statistics[
        [
            "player_name",
            "np_shots",
            "np_xg",
            "np_goals",
            "np_conversion_pct",
            "headed_shots",
            "headed_goals",
            "one_on_one_shots",
            "one_on_one_goals",
            "penalties",
            "penalty_goals",
        ]
    ].head(20).to_string(index=False)
)

print(f"\nTotal players: {len(player_statistics)}")
print(f"Saved to: {OUTPUT_FILE}")
