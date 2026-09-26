import json
from pathlib import Path
from typing import Any, Optional

import polars as pl


DEFAULT_DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"


def load_tracking_json(json_path: str | Path, match_id: int) -> pl.DataFrame:
    """
    Load a SkillCorner tracking JSON file into a table.
    One row per (frame, entity). Ball rows have player_id = -1.
    Columns: match_id, time, frame, period, player_id, is_detected, is_ball, x, y.

    :param json_path: Tracking JSON file.
    :param match_id: Match identifier attached to each row.
    :returns: Tracking rows for players and the ball.
    """
    with Path(json_path).open() as tracking_file:
        tracking_frames = json.load(tracking_file)

    tracking_rows = []
    for frame_data in tracking_frames:
        frame_info = {
            "match_id": match_id,
            "time": frame_data.get("timestamp"),
            "frame": int(frame_data.get("frame")),
            "period": frame_data.get("period"),
        }
        for player_position in frame_data.get("player_data") or []:
            tracking_rows.append({
                **frame_info,
                "player_id": player_position.get("player_id"),
                "is_detected": bool(player_position.get("is_detected", False)),
                "is_ball": False,
                "x": player_position.get("x"),
                "y": player_position.get("y"),
            })

        ball_position = frame_data.get("ball_data")
        if ball_position is not None:
            tracking_rows.append({
                **frame_info,
                "player_id": -1,  # sentinel for the ball
                "is_detected": bool(ball_position.get("is_detected", False)),
                "is_ball": True,
                "x": ball_position.get("x"),
                "y": ball_position.get("y"),
            })

    return pl.DataFrame(tracking_rows, schema_overrides={
        "time": pl.String,
        "period": pl.Int64,
        "player_id": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
    })


def load_tracking_data(tracking_parquet_path: Path, tracking_json_path: str | Path, match_id: int) -> pl.DataFrame:
    """
    Load tracking data, creating the Parquet cache from JSON if needed.

    :param tracking_parquet_path: Tracking cache to read or create.
    :param tracking_json_path: Source file used when the cache is missing.
    :param match_id: Match identifier attached when reading JSON.
    :returns: Tracking rows with non-negative period numbers.
    """
    if not tracking_parquet_path.exists():
        tracking = load_tracking_json(tracking_json_path, match_id)
        tracking.write_parquet(tracking_parquet_path)
    return pl.read_parquet(tracking_parquet_path).filter(pl.col("period") >= 0)


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    with Path(metadata_path).open() as metadata_file:
        return json.load(metadata_file)


def build_player_lookup(metadata: dict[str, Any]) -> pl.DataFrame:
    home_team_id = metadata["home_team"]["id"]
    away_team_id = metadata["away_team"]["id"]
    team_names = {
        home_team_id: metadata["home_team"]["short_name"],
        away_team_id: metadata["away_team"]["short_name"],
    }
    player_rows = []
    for player in metadata["players"]:
        player_rows.append({
            "player_id": player["id"],
            "team_id": player["team_id"],
            "team_short": team_names.get(player["team_id"], "Unknown"),
            "short_name": player["short_name"],
            "role": player["player_role"]["acronym"],  # GK, LCB, LW, ..., SUB
            "jersey_number": player["number"],
        })
    return pl.DataFrame(player_rows)


def enrich_tracking_with_player_info(tracking: pl.DataFrame, player_lookup: pl.DataFrame) -> pl.DataFrame:
    tracking_with_players = tracking.join(player_lookup, on="player_id", how="left")
    return tracking_with_players.with_columns(
        pl.when(pl.col("is_ball")).then(pl.lit("ball")).otherwise(pl.col(column)).alias(column)
        for column in ("team_short", "role", "short_name")
    )


def read_attack_directions_from_metadata(metadata: dict[str, Any]) -> dict[int, dict[int, str]]:
    """
    Read each team's attack direction by period.

    :param metadata: Match metadata with team IDs and home_team_side.
    :returns: Directions by team and period, with R for right and L for left.
    """
    home_team_id, away_team_id = metadata["home_team"]["id"], metadata["away_team"]["id"]
    attack_directions = {home_team_id: {}, away_team_id: {}}
    for period_number, home_team_side in enumerate(metadata["home_team_side"], start=1):
        if home_team_side not in ("left_to_right", "right_to_left"):
            raise ValueError(f"Unknown home_team_side: {home_team_side!r}")
        home_direction = "R" if home_team_side == "left_to_right" else "L"
        attack_directions[home_team_id][period_number] = home_direction
        attack_directions[away_team_id][period_number] = "L" if home_direction == "R" else "R"
    return attack_directions


def estimate_attack_directions_from_goalkeepers(tracking: pl.DataFrame,
                                                player_lookup: pl.DataFrame,
                                                match_periods: list[dict[str, Any]]) -> dict[int, dict[int, str]]:
    """
    Estimate attack directions from the mean goalkeeper position.
    If the goalkeeper is on -x, the team attacks +x (R).

    :param tracking: Tracking positions with player_id, frame and x.
    :param player_lookup: Player metadata with player_id, team_id, team_short and role.
    :param match_periods: Period boundaries with period, start_frame and end_frame.
    :returns: Directions by team and period, or ? when no goalkeeper position is available.
    """
    attack_directions = {}
    for team_id, team_name in player_lookup.select("team_id", "team_short").unique().iter_rows():
        if team_name in ("Ball", "Unknown"):
            continue
        goalkeeper_ids = player_lookup.filter(
            (pl.col("role") == "GK") & (pl.col("team_id") == team_id)
        )["player_id"].to_list()
        team_directions = {}
        for period_bounds in match_periods:
            goalkeeper_x_positions = tracking.filter(
                pl.col("player_id").is_in(goalkeeper_ids)
                & pl.col("frame").is_between(period_bounds["start_frame"], period_bounds["end_frame"])
                & pl.col("x").is_not_null()
            )["x"]
            if goalkeeper_x_positions.is_empty():
                team_directions[period_bounds["period"]] = "?"
            else:
                team_directions[period_bounds["period"]] = "R" if goalkeeper_x_positions.mean() < 0 else "L"
        attack_directions[team_id] = team_directions
    return attack_directions


def validate_attack_directions(tracking: pl.DataFrame, metadata: dict[str, Any], player_lookup: pl.DataFrame) -> None:
    metadata_directions = read_attack_directions_from_metadata(metadata)
    goalkeeper_directions = estimate_attack_directions_from_goalkeepers(tracking,
                                                                        player_lookup,
                                                                        metadata["match_periods"])

    for team_id, team_directions in metadata_directions.items():
        for period_number in (1, 2):
            if goalkeeper_directions[team_id][period_number] != team_directions[period_number]:
                raise ValueError(f"Disagree on period {period_number} for team {team_id}: "
                                 f"meta={team_directions[period_number]} "
                                 f"vs gk={goalkeeper_directions[team_id][period_number]}")


def build_attack_direction_lookup(attack_directions: dict[int, dict[int, str]],
                                  team_names: dict[int, str] | None = None) -> pl.DataFrame:
    """
    Put attack directions into a table by team and period.

    :param attack_directions: Directions indexed by team ID and period.
    :param team_names: Optional team labels for name-based lookup.
    :returns: One direction per team and period.
    """
    team_column = "team_short" if team_names is not None else "team_id"
    return pl.DataFrame(
        [(team_names[team_id] if team_names is not None else team_id, period_number, direction)
         for team_id, period_directions in attack_directions.items()
         for period_number, direction in period_directions.items()],
        schema={team_column: pl.String if team_names is not None else pl.Int64,
                "period": pl.Int64, "attack_dir": pl.String},
        orient="row",
    )


def add_attack_directions(tracking: pl.DataFrame, attack_direction_lookup: pl.DataFrame) -> pl.DataFrame:
    """
    Attach attack directions to tracking rows.

    :param tracking: Player and ball tracking rows.
    :param attack_direction_lookup: Directions indexed by team ID or name and period.
    :returns: Tracking with Ball on ball rows and ? for unknown team/period pairs.
    """
    team_column = "team_id" if "team_id" in attack_direction_lookup.columns else "team_short"
    return (
        tracking.join(attack_direction_lookup, on=[team_column, "period"], how="left", validate="m:1")
        .with_columns(
            pl.when(pl.col("is_ball")).then(pl.lit("Ball"))
            .otherwise(pl.col("attack_dir").fill_null("?"))
            .alias("attack_dir")
        )
    )


def enrich_tracking_with_attack_direction(tracking: pl.DataFrame, metadata: dict[str, Any]) -> pl.DataFrame:
    attack_directions = read_attack_directions_from_metadata(metadata)
    if "match_id" in tracking.columns and tracking["match_id"].n_unique() > 1:
        raise ValueError("Enrich attack direction one match at a time")
    if "match_id" in tracking.columns and "id" in metadata:
        match_ids = tracking["match_id"].drop_nulls().unique().to_list()
        if match_ids and match_ids != [metadata["id"]]:
            raise ValueError("Tracking match_id does not match metadata")
    return add_attack_directions(tracking, build_attack_direction_lookup(attack_directions))


# if __name__ == "__main__":
#     MATCH_ID = 2004437
#     tracking_parquet_path = Path(f"data/tracking/{MATCH_ID}.parquet")
#     tracking_json_path = Path(f"data/tracking/{MATCH_ID}.json")
#     tracking = load_tracking_data(tracking_parquet_path, tracking_json_path, MATCH_ID)


def validate_binary_outcomes(observations: pl.DataFrame, group_cols: str | list[str], success_col: str) -> None:
    """
    Check required columns and reject missing or non-binary outcomes.

    :param observations: Table to validate, with one row per attempt.
    :param group_cols: Group columns that must exist; null group keys are permitted.
    :param success_col: Outcome column containing only Boolean or 0/1 values.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    required_columns = set(group_cols + [success_col])
    missing_columns = required_columns - set(observations.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if observations[success_col].null_count() > 0:
        raise ValueError(f"{success_col!r} contains null values")

    outcomes = observations[success_col]
    if outcomes.dtype != pl.Boolean and (not outcomes.dtype.is_numeric() or not outcomes.is_in([0, 1]).all()):
        raise ValueError(f"{success_col!r} must contain only Boolean or 0/1 values")


def aggregate_binary_outcomes(observations: pl.DataFrame, group_cols: str | list[str], success_col: str) -> pl.DataFrame:
    """
    Count successes and attempts for each group.
    Example: "How many successes and attempts does each player have?"

    :param observations: Table where each row represents one attempt.
    :param group_cols: Columns defining the entities to group together.
    :param success_col: Boolean or 0/1 column indicating whether each attempt succeeded.
    :returns: One row per group with its number of successes and attempts.
    """
    validate_binary_outcomes(observations, group_cols, success_col)
    return observations.group_by(group_cols).agg(pl.col(success_col).cast(pl.Int64).sum().alias("successes"),
                                                pl.len().alias("attempts"))
