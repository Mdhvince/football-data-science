import json
from pathlib import Path
from typing import Any, Optional

import polars as pl


def json_tracking_to_dataframe(json_path: str | Path, match_id: int) -> pl.DataFrame:
    """
    Load a SkillCorner tracking JSON into a flat DataFrame.
    One row per (frame, entity). Ball rows have player_id = -1.
    Columns: match_id, time, frame, period, player_id, is_detected, is_ball, x, y.

    :param json_path: Tracking JSON file.
    :param match_id: Match identifier attached to each row.
    :returns: Tracking rows for players and the ball.
    """
    with Path(json_path).open() as tracking_file:
        frames = json.load(tracking_file)

    rows = []
    for frame in frames:
        frame_info = {
            "match_id": match_id,
            "time": frame.get("timestamp"),
            "frame": int(frame.get("frame")),
            "period": frame.get("period"),
        }
        for player in frame.get("player_data") or []:
            rows.append({
                **frame_info,
                "player_id": player.get("player_id"),
                "is_detected": bool(player.get("is_detected", False)),
                "is_ball": False,
                "x": player.get("x"),
                "y": player.get("y"),
            })

        ball = frame.get("ball_data")
        if ball is not None:
            rows.append({
                **frame_info,
                "player_id": -1,  # sentinel for the ball
                "is_detected": bool(ball.get("is_detected", False)),
                "is_ball": True,
                "x": ball.get("x"),
                "y": ball.get("y"),
            })

    return pl.DataFrame(rows, schema_overrides={
        "time": pl.String,
        "period": pl.Int64,
        "player_id": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
    })


def generate_tracking_dataframe(tracking_parquet_path: Path,
                                tracking_json_path: str | Path,
                                match_id: int) -> pl.DataFrame:
    if not tracking_parquet_path.exists():
        df = json_tracking_to_dataframe(tracking_json_path, match_id)
        df.write_parquet(tracking_parquet_path)
    return pl.read_parquet(tracking_parquet_path).filter(pl.col("period") >= 0)


def load_metadata(meta_path: str | Path) -> dict[str, Any]:
    with Path(meta_path).open() as metadata_file:
        return json.load(metadata_file)


def build_player_lookup(meta: dict[str, Any]) -> pl.DataFrame:
    home_id = meta["home_team"]["id"]
    away_id = meta["away_team"]["id"]
    team_names = {home_id: meta["home_team"]["short_name"], away_id: meta["away_team"]["short_name"]}
    rows = []
    for player in meta["players"]:
        rows.append({
            "player_id": player["id"],
            "team_id": player["team_id"],
            "team_short": team_names.get(player["team_id"], "Unknown"),
            "short_name": player["short_name"],
            "role": player["player_role"]["acronym"],  # GK, LCB, LW, ..., SUB
            "jersey_number": player["number"],
        })
    return pl.DataFrame(rows)


def enrich_tracking_with_player_info(tracking: pl.DataFrame, player_lookup: pl.DataFrame) -> pl.DataFrame:
    tracking_enriched = tracking.join(player_lookup, on="player_id", how="left")
    return tracking_enriched.with_columns(
        pl.when(pl.col("is_ball")).then(pl.lit("ball")).otherwise(pl.col(column)).alias(column)
        for column in ("team_short", "role", "short_name")
    )


def attack_direction_from_meta(meta: dict[str, Any]) -> dict[int, dict[int, str]]:
    """
    Read each team's attack direction by period.

    :param meta: Match metadata with team IDs and home_team_side.
    :returns: Directions by team and period, with R for right and L for left.
    """
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    directions = {home_id: {}, away_id: {}}
    for period, side in enumerate(meta["home_team_side"], start=1):
        if side not in ("left_to_right", "right_to_left"):
            raise ValueError(f"Unknown home_team_side: {side!r}")
        home_direction = "R" if side == "left_to_right" else "L"
        directions[home_id][period] = home_direction
        directions[away_id][period] = "L" if home_direction == "R" else "R"
    return directions


def attack_direction_from_goalkeeper(tracking: pl.DataFrame,
                                     player_lookup_table: pl.DataFrame,
                                     match_periods: list[dict[str, Any]]) -> dict[int, dict[int, str]]:
    """
    Infer attack direction from the mean goalkeeper position.
    If the goalkeeper is on -x, the team attacks +x (R).

    :param tracking: Tracking positions with player_id, frame and x.
    :param player_lookup_table: Player metadata with player_id, team_id, team_short and role.
    :param match_periods: Period boundaries with period, start_frame and end_frame.
    :returns: Directions by team and period, or ? when no goalkeeper position is available.
    """
    directions = {}
    for team_id, team_short in player_lookup_table.select("team_id", "team_short").unique().iter_rows():
        if team_short in ("Ball", "Unknown"):
            continue
        goalkeeper_ids = player_lookup_table.filter(
            (pl.col("role") == "GK") & (pl.col("team_id") == team_id)
        )["player_id"].to_list()
        team_directions = {}
        for period in match_periods:
            goalkeeper_x = tracking.filter(
                pl.col("player_id").is_in(goalkeeper_ids)
                & pl.col("frame").is_between(period["start_frame"], period["end_frame"])
                & pl.col("x").is_not_null()
            )["x"]
            if goalkeeper_x.is_empty():
                team_directions[period["period"]] = "?"
            else:
                team_directions[period["period"]] = "R" if goalkeeper_x.mean() < 0 else "L"
        directions[team_id] = team_directions
    return directions


def sanity_check_attack_direction(tracking: pl.DataFrame, meta: dict[str, Any], player_lookup: pl.DataFrame) -> None:
    dir_meta = attack_direction_from_meta(meta)
    dir_gk = attack_direction_from_goalkeeper(tracking, player_lookup, meta["match_periods"])

    for team_id, directions in dir_meta.items():
        for period in (1, 2):
            if dir_gk[team_id][period] != directions[period]:
                raise ValueError(f"Disagree on period {period} for team {team_id}: "
                                 f"meta={directions[period]} vs gk={dir_gk[team_id][period]}")


def attack_dir_lookup(dir_meta: dict[int, dict[int, str]], team_name_map: dict[int, str] | None = None) -> pl.DataFrame:
    """
    Flatten attack directions by team and period.

    :param dir_meta: Directions indexed by team ID and period.
    :param team_name_map: Optional team labels for name-based lookup.
    :returns: One direction per team and period.
    """
    key = "team_short" if team_name_map is not None else "team_id"
    return pl.DataFrame(
        [(team_name_map[team_id] if team_name_map is not None else team_id, period, direction)
         for team_id, periods in dir_meta.items()
         for period, direction in periods.items()],
        schema={key: pl.String if team_name_map is not None else pl.Int64,
                "period": pl.Int64, "attack_dir": pl.String},
        orient="row",
    )


def add_attack_dir(tracking: pl.DataFrame, attack_lookup: pl.DataFrame) -> pl.DataFrame:
    """
    Attach attack directions to tracking rows.

    :param tracking: Player and ball tracking rows.
    :param attack_lookup: Directions indexed by team ID or name and period.
    :returns: Tracking with Ball on ball rows and ? for unknown team/period pairs.
    """
    key = "team_id" if "team_id" in attack_lookup.columns else "team_short"
    return (
        tracking.join(attack_lookup, on=[key, "period"], how="left", validate="m:1")
        .with_columns(
            pl.when(pl.col("is_ball")).then(pl.lit("Ball"))
            .otherwise(pl.col("attack_dir").fill_null("?"))
            .alias("attack_dir")
        )
    )


def enrich_tracking_with_attack_direction(tracking: pl.DataFrame, meta: dict[str, Any]) -> pl.DataFrame:
    dir_meta = attack_direction_from_meta(meta)
    if "match_id" in tracking.columns and tracking["match_id"].n_unique() > 1:
        raise ValueError("Enrich attack direction one match at a time")
    if "match_id" in tracking.columns and "id" in meta:
        match_ids = tracking["match_id"].drop_nulls().unique().to_list()
        if match_ids and match_ids != [meta["id"]]:
            raise ValueError("Tracking match_id does not match metadata")
    return add_attack_dir(tracking, attack_dir_lookup(dir_meta))


# if __name__ == "__main__":
#     MATCH_ID = 2004437
#     tracking_parquet_path = Path(f"data/tracking/{MATCH_ID}.parquet")
#     tracking_json_path = Path(f"data/tracking/{MATCH_ID}.json")
#     tracking = generate_tracking_dataframe(tracking_parquet_path, tracking_json_path, MATCH_ID)


def validate_binary_outcomes(df: pl.DataFrame, group_cols: str | list[str], success_col: str) -> None:
    """
    Check required columns and reject missing or non-binary outcomes.

    :param df: Observation-level table to validate.
    :param group_cols: Group columns that must exist; null group keys are permitted.
    :param success_col: Outcome column containing only Boolean or 0/1 values.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    required = set(group_cols + [success_col])
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[success_col].null_count() > 0:
        raise ValueError(f"{success_col!r} contains null values")

    values = df[success_col]
    if values.dtype != pl.Boolean and (not values.dtype.is_numeric() or not values.is_in([0, 1]).all()):
        raise ValueError(f"{success_col!r} must contain only Boolean or 0/1 values")



def aggregate_binary_outcomes(df: pl.DataFrame, group_cols: str | list[str], success_col: str) -> pl.DataFrame:
    """
    Aggregate binary observations into successes and attempts for each group.
    Example: "How many successes and attempts does each player have?"

    :param df: Observation-level table where each row represents one attempt.
    :param group_cols: Columns defining the entities to aggregate.
    :param success_col: Boolean or 0/1 column indicating whether each attempt succeeded.
    :returns: One row per group with its number of successes and attempts.
    """
    validate_binary_outcomes(df, group_cols, success_col)
    return df.group_by(group_cols).agg(pl.col(success_col).cast(pl.Int64).sum().alias("successes"),
                                      pl.len().alias("attempts"))
