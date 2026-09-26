from typing import Any

import numpy as np
import polars as pl

from src.utils import validate_binary_outcomes


SHOT_FEATURES = ["distance_to_goal", "shot_cone_defenders", "nearest_defender_distance"]


def prepare_shots(events: pl.DataFrame, frame_col: str = "frame_end") -> pl.DataFrame:
    """
    Select shot possessions and keep their outcome and chosen tracking frame.
    Example: "Which possessions ended in a shot?"

    :param events: SkillCorner dynamic events with non-null binary lead_to_goal.
    :param frame_col: Event frame column used to select the tracking snapshot; defaults to frame_end.
    :returns: Shot rows with is_goal and shot_frame, without dropping missing group keys.
    """
    required_columns = {"event_type", "end_type", "lead_to_goal", "match_id", "period",
                        "player_id", "team_id", "player_position", frame_col}
    missing_columns = required_columns - set(events.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    shots = events.filter((pl.col("event_type") == "player_possession") & (pl.col("end_type") == "shot"))
    validate_binary_outcomes(shots, "match_id", "lead_to_goal")
    return shots.with_columns(pl.col("lead_to_goal").cast(pl.Boolean).alias("is_goal"),
                              pl.col(frame_col).alias("shot_frame"))


def _measure_defensive_geometry(shooter_x: float,
                                shooter_y: float,
                                goal_x: float,
                                opponents: pl.DataFrame,
                                goal_width: float) -> dict[str, float | int]:
    """
    Count opponents in the shot cone and measure the nearest opponent distance.

    :param shooter_x: Shooter position along the pitch.
    :param shooter_y: Shooter position across the pitch.
    :param goal_x: Goal-line position, which must differ from shooter_x.
    :param opponents: Non-empty opponent positions with finite x and y.
    :param goal_width: Distance between the goalposts in metres.
    :returns: Shot-cone defender count and nearest opponent distance.
    """
    opponent_x, opponent_y = opponents["x"].to_numpy(), opponents["y"].to_numpy()
    # Find the two edges of the shooter-to-goalposts triangle at each opponent's x.
    fraction_to_goal = (opponent_x - shooter_x) / (goal_x - shooter_x)
    cone_lower_y = shooter_y + fraction_to_goal * (-goal_width / 2 - shooter_y)
    cone_upper_y = shooter_y + fraction_to_goal * (goal_width / 2 - shooter_y)
    inside_cone = ((fraction_to_goal >= 0) & (fraction_to_goal <= 1)
                   & (opponent_y >= cone_lower_y) & (opponent_y <= cone_upper_y))
    return {
        "shot_cone_defenders": int(inside_cone.sum()),
        "nearest_defender_distance": float(np.hypot(opponent_x - shooter_x, opponent_y - shooter_y).min()),
    }


def _compute_shot_features(shot: dict[str, Any],
                           frame_tracking: pl.DataFrame | None,
                           pitch_length: float,
                           goal_width: float,
                           detected_only: bool,
                           include_goalkeeper: bool) -> dict[str, float | int | str | None]:
    """
    Check one shot's tracking data before measuring its geometry.

    :param shot: Prepared shot with shooter and team IDs.
    :param frame_tracking: Matching player positions, or None if the frame is missing.
    :param pitch_length: Pitch length in metres.
    :param goal_width: Distance between the goalposts in metres.
    :param detected_only: Exclude estimated positions when True.
    :param include_goalkeeper: Include opposing goalkeepers in defensive features.
    :returns: Feature values, opponent count and tracking status for the shot.
    """
    shot_features: dict[str, float | int | str | None] = dict.fromkeys(SHOT_FEATURES)
    shot_features.update(n_opponents=0, tracking_status="missing_frame")
    if frame_tracking is None:
        return shot_features

    players_with_positions = frame_tracking.filter(pl.col("x").is_finite() & pl.col("y").is_finite())
    if detected_only:
        players_with_positions = players_with_positions.filter(pl.col("is_detected"))
    shooter_tracking = players_with_positions.filter(pl.col("player_id") == shot["player_id"])
    if shooter_tracking.is_empty():
        shot_features["tracking_status"] = "missing_shooter"
        return shot_features

    shooter_position = shooter_tracking.row(0, named=True)
    if shot["team_id"] is None or shooter_position["team_id"] != shot["team_id"]:
        shot_features["tracking_status"] = "team_mismatch"
        return shot_features
    if shooter_position["attack_dir"] not in ("R", "L"):
        shot_features["tracking_status"] = "unknown_direction"
        return shot_features

    shooter_x, shooter_y = shooter_position["x"], shooter_position["y"]
    goal_x = pitch_length / 2 * (1 if shooter_position["attack_dir"] == "R" else -1)
    shot_features["distance_to_goal"] = float(np.hypot(goal_x - shooter_x, shooter_y))
    opponents = players_with_positions.filter(pl.col("team_id") != shot["team_id"])
    if not include_goalkeeper:
        opponents = opponents.filter(pl.col("role").is_not_null() & (pl.col("role") != "GK"))
    shot_features["n_opponents"] = opponents.height
    if opponents.is_empty():
        shot_features["tracking_status"] = "no_opponents"
        return shot_features
    if np.isclose(goal_x, shooter_x):
        shot_features["tracking_status"] = "degenerate_cone"
        return shot_features

    shot_features.update(_measure_defensive_geometry(shooter_x, shooter_y, goal_x, opponents, goal_width))
    shot_features["tracking_status"] = "ok"
    return shot_features


def compute_shot_tracking_features(shots: pl.DataFrame,
                                   tracking: pl.DataFrame,
                                   pitch_length: float,
                                   goal_width: float = 7.32,
                                   detected_only: bool = False,
                                   include_goalkeeper: bool = True) -> pl.DataFrame:
    """
    Measure shot geometry using shooter and opponent positions from enriched tracking.
    Example: "How far was the shooter from goal and the nearest opponent?"

    :param shots: Prepared shots with match_id, period, shot_frame, player_id and team_id.
    :param tracking: Enriched tracking in centred metres; R attacks +x, L attacks -x.
    :param pitch_length: Pitch length from metadata; process different pitches separately.
    :param goal_width: Distance between the goalposts in metres.
    :param detected_only: Exclude estimated positions when True, including the shooter.
    :param include_goalkeeper: Count the opposing goalkeeper in both defensive features.
    :returns: All shot rows with features, opponent count and an explicit tracking_status.
    """
    if not np.isfinite([pitch_length, goal_width]).all() or min(pitch_length, goal_width) <= 0:
        raise ValueError("Pitch length and goal width must be finite and positive")
    required_shot_columns = {"match_id", "period", "shot_frame", "player_id", "team_id"}
    required_tracking_columns = {"match_id", "period", "frame", "player_id", "team_id",
                                 "is_ball", "is_detected", "x", "y", "role", "attack_dir"}
    for table, required_columns in [(shots, required_shot_columns), (tracking, required_tracking_columns)]:
        missing_columns = required_columns - set(table.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    feature_columns = SHOT_FEATURES + ["n_opponents", "tracking_status"]
    if set(feature_columns) & set(shots.columns):
        raise ValueError("Shots already contain tracking feature columns")

    frame_keys = ["match_id", "period", "frame"]
    shot_frames = shots.select("match_id", "period", pl.col("shot_frame").alias("frame")).unique()
    player_tracking = tracking.join(shot_frames, on=frame_keys, how="semi").filter(~pl.col("is_ball"))
    if player_tracking.select(frame_keys + ["player_id"]).is_duplicated().any():
        raise ValueError("Tracking must contain one row per match/period/frame/player")
    tracking_by_frame = player_tracking.partition_by(frame_keys, as_dict=True)
    feature_rows = []
    for shot in shots.iter_rows(named=True):
        frame_tracking = tracking_by_frame.get((shot["match_id"], shot["period"], shot["shot_frame"]))
        feature_rows.append(
            _compute_shot_features(shot, frame_tracking, pitch_length, goal_width, detected_only, include_goalkeeper))
    tracking_features = pl.DataFrame(feature_rows, schema={
        "distance_to_goal": pl.Float64, "shot_cone_defenders": pl.Int64,
        "nearest_defender_distance": pl.Float64, "n_opponents": pl.Int64,
        "tracking_status": pl.String,
    })
    return shots.hstack(tracking_features)
