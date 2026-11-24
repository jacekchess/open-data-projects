
from pathlib import Path
import pandas as pd
import yaml


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path("open-data-projects/data/processed")
MASTER_DIR = Path("open-data-projects/data/master")

CONFIG_FILE = Path("open-data-projects/master_data/position_roles_config.yaml")

MASTER_OUTPUT = MASTER_DIR / "master_players.csv"
QUANTILES_OUTPUT = MASTER_DIR / "position_group_quantiles.csv"
UPDATED_CONFIG_OUTPUT = MASTER_DIR / "position_roles_config_with_quantiles.yaml"


PLAYER_COLUMNS = [
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
]


STAT_FILES = [
    "defending.csv",
    "passes.csv",
    "goalkeepers.csv",
    "shots.csv",
    "carries_dribbles.csv",
]


QUANTILES = {
    "p10": 0.10,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
}


# These metrics are already percentages, averages or
# per-event rates. Do not divide them by minutes.
RATE_METRICS = {
    "carry_retention_pct",
    "dribble_success_pct",
    "average_carry_distance",
    "long_carries_10m_pct",
    "very_long_carries_20m_pct",
    "successful_nutmeg_pct",

    "pass_completion_pct",
    "corner_completion_pct",

    "save_pct",
    "penalty_save_pct",
    "actions_per_shot_on_target",

    "np_conversion_pct",
    "np_shots_on_target_pct",
    "np_xg_per_shot",

    "headed_conversion_pct",
    "headed_shots_on_target_pct",
    "headed_xg_per_shot",

    "one_on_one_conversion_pct",
    "one_on_one_shots_on_target_pct",
    "one_on_one_xg_per_shot",

    "penalty_conversion_pct",

    # Newly introduced defending metrics
    "tackle_success_pct",
    "dribbled_past_pct",
    "aerial_duel_win_pct",
}

# Higher values indicate worse performance for these metrics.
# Their percentile rankings will therefore be reversed.
LOWER_IS_BETTER = {
    "aerial_lost",
    "tackles_lost",
    "dribbled_past",
    "dribbled_past_pct",
    "dispossessed",
    "fouls",
    "yellow_cards",
    "red_cards",
    "goals_conceded",
    "penalty_goals_conceded",
    "action_outcome_fail",
}


# ============================================================
# LOAD CONFIGURATION
# ============================================================

def load_config():
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ============================================================
# POSITION MAPPING
# ============================================================

def build_position_mapping(config):
    position_mapping = {}

    for group_name, group_data in config["position_groups"].items():

        for position in group_data["positions"]:

            if position in position_mapping:
                raise ValueError(
                    f"Position {position} belongs to multiple groups."
                )

            position_mapping[position] = group_name

    expected_positions = set(config.get("source_positions", []))
    mapped_positions = set(position_mapping.keys())

    missing = expected_positions - mapped_positions

    if missing:
        raise ValueError(
            f"Positions missing from YAML: {sorted(missing)}"
        )

    return position_mapping


# ============================================================
# COLLECT METRICS USED BY ALL ROLES
# ============================================================

def collect_role_metrics(config):
    metrics = set()

    for role_data in config["roles"].values():
        role_metrics = role_data.get("metrics", {})

        if isinstance(role_metrics, dict):
            # YAML format: high / low / support
            for tier in ("high", "low", "support"):
                metrics.update(role_metrics.get(tier, []) or [])

        elif isinstance(role_metrics, list):
            # Backwards compatibility with the earlier flat-list format
            metrics.update(role_metrics)

    # Derived metrics and additional parser metrics
    additional_metrics = {
        # Goalkeepers
        "goals_prevented",
        "xg_against_all_attributed_shots",
        "xg_against_on_target_shots",

        # Defending
        "aerial_won",
        "aerial_lost",
        "aerial_duel_win_pct",
        "tackle_success_pct",
        "dribbled_past_pct",

        # Shooting
        "np_goals_minus_xg",
    }

    metrics.update(additional_metrics)

    return sorted(metrics)

# ============================================================
# LOAD PLAYER POSITION TABLE
# ============================================================

def load_players():

    possible_files = [
        PROCESSED_DIR / "players_positions.csv",
        PROCESSED_DIR / "players_position.csv",
    ]

    players_file = next(
        (path for path in possible_files if path.exists()),
        None
    )

    if players_file is None:
        raise FileNotFoundError(
            "Could not find players_position.csv "
            "or players_positions.csv"
        )

    players = pd.read_csv(
        players_file,
        dtype={"player_id": "string"}
    )

    missing_columns = set(PLAYER_COLUMNS) - set(players.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in player table: {sorted(missing_columns)}"
        )

    players = players[PLAYER_COLUMNS].copy()

    players["player_id"] = players["player_id"].str.strip()

    players["total_minutes"] = pd.to_numeric(
        players["total_minutes"],
        errors="coerce"
    )

    # Remove players with fewer than 90 minutes.
    players = players[
        players["total_minutes"] >= 90
    ].copy()

    if players["player_id"].duplicated().any():
        raise ValueError(
            "Player position table contains duplicate player_id values."
        )

    return players


# ============================================================
# LOAD AND MERGE STATISTICS
# ============================================================

def load_statistics_file(filename):

    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing statistics file: {path}")

    df = pd.read_csv(
        path,
        dtype={"player_id": "string"}
    )

    if "player_id" not in df.columns:
        raise ValueError(f"{filename} has no player_id column.")

    # Drop player_name to avoid merge conflicts.
    if "player_name" in df.columns:
        df = df.drop(columns=["player_name"])

    df["player_id"] = df["player_id"].str.strip()

    if df["player_id"].duplicated().any():
        raise ValueError(
            f"{filename} contains duplicate player_id values. "
            "Each statistics file must have one row per player."
        )

    return df


def merge_all_statistics(players):

    master = players.copy()

    for filename in STAT_FILES:

        stats = load_statistics_file(filename)

        overlapping_columns = (
            set(master.columns) & set(stats.columns)
        ) - {"player_id"}

        if overlapping_columns:
            raise ValueError(
                f"Column conflicts in {filename}: "
                f"{sorted(overlapping_columns)}"
            )

        master = master.merge(
            stats,
            on="player_id",
            how="left",
            validate="one_to_one"
        )

    return master


# ============================================================
# SELECT AND NORMALIZE METRICS
# ============================================================

def add_derived_metrics(master):

    master = master.copy()

    # --------------------------------------------------------
    # GOALKEEPER: GOALS PREVENTED
    # --------------------------------------------------------

    if (
        "xg_against_all_attributed_shots" in master.columns
        and "goals_conceded" in master.columns
    ):
        master["goals_prevented"] = (
            pd.to_numeric(
                master["xg_against_all_attributed_shots"],
                errors="coerce"
            )
            -
            pd.to_numeric(
                master["goals_conceded"],
                errors="coerce"
            )
        )

    # --------------------------------------------------------
    # SHOOTING: FINISHING OVERPERFORMANCE
    # --------------------------------------------------------

    if (
        "np_goals" in master.columns
        and "np_xg" in master.columns
    ):
        master["np_goals_minus_xg"] = (
            pd.to_numeric(master["np_goals"], errors="coerce")
            -
            pd.to_numeric(master["np_xg"], errors="coerce")
        )

    return master

def prepare_metrics(master, role_metrics):

    master = add_derived_metrics(master)

    available_metrics = [
        metric
        for metric in role_metrics
        if metric in master.columns
    ]

    missing_metrics = sorted(
        set(role_metrics) - set(available_metrics)
    )

    if missing_metrics:
        print("\nWARNING: These configured metrics were not found:")
        for metric in missing_metrics:
            print(f"  - {metric}")

    for metric in available_metrics:

        master[metric] = pd.to_numeric(
            master[metric],
            errors="coerce"
        )

        # Preserve missing values rather than automatically
        # treating missing measurements as genuine zeroes.

        if metric not in RATE_METRICS:

            per90_column = f"{metric}_per90"

            master[per90_column] = (
                master[metric] * 90 / master["total_minutes"]
            )

    return master, available_metrics

# ============================================================
# CALCULATE PLAYER PERCENTILES BY POSITION GROUP
# ============================================================

def calculate_player_percentiles(master, metrics):
    master = master.copy()

    for metric in metrics:

        if metric in RATE_METRICS:
            source_column = metric
        else:
            source_column = f"{metric}_per90"

        if source_column not in master.columns:
            continue

        percentile_column = f"{metric}_percentile"

        # Higher values are better for normal metrics.
        # Lower values are better for metrics in LOWER_IS_BETTER.
        ascending = metric not in LOWER_IS_BETTER

        master[percentile_column] = (
            master.groupby("grouped_position")[source_column]
            .rank(
                method="average",
                pct=True,
                ascending=ascending,
            )
            .mul(100)
            .round()
            .clip(1, 100)
            .astype("Int64")
        )

    return master

# ============================================================
# CALCULATE QUANTILES BY POSITION GROUP
# ============================================================

def calculate_position_quantiles(master, metrics):

    quantile_rows = []
    quantile_data = {}

    for group_name, group in master.groupby("grouped_position"):

        quantile_data[group_name] = {}

        for metric in metrics:

            # Count/volume metrics use per-90 values.
            # Percentages and averages retain their original values.
            if metric in RATE_METRICS:
                column = metric
                normalization = "unchanged"
            else:
                column = f"{metric}_per90"
                normalization = "per_90"

            if column not in group.columns:
                continue

            values = group[column].dropna()

            if values.empty:
                continue

            metric_quantiles = {
                name: float(values.quantile(q))
                for name, q in QUANTILES.items()
            }

            quantile_data[group_name][metric] = {
                "source_column": column,
                "normalization": normalization,
                "n_players": int(len(values)),
                **metric_quantiles,
            }

            # Long-format output table.
            for quantile_name, value in metric_quantiles.items():

                quantile_rows.append({
                    "grouped_position": group_name,
                    "metric": metric,
                    "source_column": column,
                    "quantile": quantile_name,
                    "value": value,
                    "n_players": len(values),
                })

    quantiles_df = pd.DataFrame(quantile_rows)

    return quantiles_df, quantile_data


# ============================================================
# ADD ROLE-SPECIFIC QUANTILES TO YAML
# ============================================================

def update_yaml_with_quantiles(config, quantile_data):

    for group_name, group_data in config["position_groups"].items():

        group_metrics = quantile_data.get(group_name, {})

        role_profiles = {}

        for role_name in group_data["roles"]:

            role_metrics = config["roles"][role_name]["metrics"]

            selected_quantiles = {}

            for tier in ("high", "low", "support"):
                for metric in role_metrics.get(tier, []) or []:
                    if metric in group_metrics:
                        selected_quantiles[metric] = group_metrics[metric]

            role_profiles[role_name] = {
                "metrics": selected_quantiles
            }

        # Each group gets its own role profiles.
        # This matters because the same role can appear
        # in different position groups with different distributions.
        group_data["role_profiles"] = role_profiles

        # Also store all group-level metric quantiles.
        group_data["metric_quantiles"] = group_metrics

    config["quantile_metadata"] = {
        "quantiles": QUANTILES,
        "minimum_minutes": 90,
        "normalization": (
            "Count and volume metrics use per-90 values. "
            "Percentages and averages are unchanged."
        ),
        "population": (
            "All players in each grouped position. "
            "Statistics are not filtered by role."
        ),
        "percentile_scoring": {
            "scale": "1-100",
            "higher_is_better": "Higher values receive higher percentiles.",
            "lower_is_better": sorted(LOWER_IS_BETTER),
            "ranking_population": "Players within the same grouped position.",
            "count_metrics": "Ranked using per-90 values.",
            "rate_metrics": "Ranked using original values.",
            "ties": "Players with tied values receive the same percentile."
        },
    }

    return config

def report_metric_coverage(config, master, available_metrics):
    configured_metrics = set()

    for role_data in config["roles"].values():
        role_metrics = role_data.get("metrics", {})

        for tier in ("high", "low", "support"):
            configured_metrics.update(
                role_metrics.get(tier, []) or []
            )

    available_metrics = set(available_metrics)

    missing = sorted(configured_metrics - available_metrics)

    print("\nMETRIC COVERAGE")
    print(f"Configured unique metrics: {len(configured_metrics)}")
    print(f"Available metrics: {len(configured_metrics & available_metrics)}")
    print(f"Missing metrics: {len(missing)}")

    if missing:
        print("\nConfigured metrics not found in the processed statistics:")
        for metric in missing:
            print(f"  - {metric}")

    percentile_metrics = {
        col.removesuffix("_percentile")
        for col in master.columns
        if col.endswith("_percentile")
    }

    missing_percentiles = sorted(
        configured_metrics - percentile_metrics
    )

    print(f"\nMissing percentile columns: {len(missing_percentiles)}")

    if missing_percentiles:
        for metric in missing_percentiles:
            print(f"  - {metric}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading YAML configuration...")
    config = load_config()

    print("Building position mapping...")
    position_mapping = build_position_mapping(config)

    print("Collecting role metrics...")
    role_metrics = collect_role_metrics(config)

    print(f"Metrics used across all roles: {len(role_metrics)}")

    print("Loading player positions...")
    players = load_players()

    print(f"Players with at least 90 minutes: {len(players)}")

    print("Mapping grouped positions...")
    players["grouped_position"] = (
        players["position"].map(position_mapping)
    )

    unknown_positions = players.loc[
        players["grouped_position"].isna(),
        "position"
    ].dropna().unique()

    if len(unknown_positions):
        raise ValueError(
            f"Unmapped player positions: {list(unknown_positions)}"
        )

    print("Merging statistics...")
    master = merge_all_statistics(players)

    print("Preparing metrics and calculating per-90 values...")
    master, available_metrics = prepare_metrics(
        master,
        role_metrics
    )

    print("Calculating player percentiles by position group...")
    master = calculate_player_percentiles(
        master,
        available_metrics
    )

    report_metric_coverage(
        config,
        master,
        available_metrics
    )

    print("Calculating position-group quantiles...")
    quantiles_df, quantile_data = calculate_position_quantiles(
        master,
        available_metrics
    )

    print("Updating YAML configuration...")
    updated_config = update_yaml_with_quantiles(
        config,
        quantile_data
    )

    MASTER_DIR.mkdir(parents=True, exist_ok=True)

    print("Saving master player table...")
    master.to_csv(
        MASTER_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("Saving quantile table...")
    quantiles_df.to_csv(
        QUANTILES_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("Saving updated YAML...")
    with UPDATED_CONFIG_OUTPUT.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            updated_config,
            file,
            allow_unicode=True,
            sort_keys=False,
            width=120
        )

    print("\nFinished successfully.")
    print(f"Master table: {MASTER_OUTPUT}")
    print(f"Quantiles: {QUANTILES_OUTPUT}")
    print(f"Updated YAML: {UPDATED_CONFIG_OUTPUT}")

    print("\nPlayers by grouped position:")
    print(
        master["grouped_position"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()
