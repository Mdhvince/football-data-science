from typing import Any

import numpy as np
import polars as pl

from .utils import validate_binary_outcomes


SHOT_FEATURES = ["distance_to_goal", "shot_cone_defenders", "nearest_defender_distance"]


def prepare_shots(events: pl.DataFrame, frame_col: str = "frame_end") -> pl.DataFrame:
    """
    Select shot possessions and keep their outcome and chosen tracking frame.
    Example: "Which possessions ended in a shot?"

    :param events: SkillCorner dynamic events with non-null binary lead_to_goal.
    :param frame_col: Shot moment; frame_end by default, frame_start for notebook parity.
    :returns: Shot rows with is_goal and shot_frame, without dropping missing group keys.
    """
    required = {"event_type", "end_type", "lead_to_goal", "match_id", "period",
                "player_id", "team_id", "player_position", frame_col}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    shots = events.filter((pl.col("event_type") == "player_possession") & (pl.col("end_type") == "shot"))
    validate_binary_outcomes(shots, "match_id", "lead_to_goal")
    return shots.with_columns(pl.col("lead_to_goal").cast(pl.Boolean).alias("is_goal"),
                              pl.col(frame_col).alias("shot_frame"))


def _measure_defensive_geometry(x: float,
                                y: float,
                                goal_x: float,
                                opponents: pl.DataFrame,
                                goal_width: float) -> dict[str, float | int]:
    """
    Count opponents in the shot cone and measure the nearest opponent distance.

    :param x: Shooter position along the pitch.
    :param y: Shooter position across the pitch.
    :param goal_x: Goal-line position, which must differ from x.
    :param opponents: Non-empty opponent positions with finite x and y.
    :param goal_width: Distance between the goalposts in metres.
    :returns: Shot-cone defender count and nearest opponent distance.
    """
    opponent_x, opponent_y = opponents["x"].to_numpy(), opponents["y"].to_numpy()
    # Interpolate the two edges of the shooter-to-goalposts triangle at each opponent's x.
    fraction_to_goal = (opponent_x - x) / (goal_x - x)
    cone_lower_y = y + fraction_to_goal * (-goal_width / 2 - y)
    cone_upper_y = y + fraction_to_goal * (goal_width / 2 - y)
    inside_cone = ((fraction_to_goal >= 0) & (fraction_to_goal <= 1)
                   & (opponent_y >= cone_lower_y) & (opponent_y <= cone_upper_y))
    return {
        "shot_cone_defenders": int(inside_cone.sum()),
        "nearest_defender_distance": float(np.hypot(opponent_x - x, opponent_y - y).min()),
    }


def _compute_shot_features(shot: dict[str, Any],
                           frame: pl.DataFrame | None,
                           pitch_length: float,
                           goal_width: float,
                           detected_only: bool,
                           include_goalkeeper: bool) -> dict[str, float | int | str | None]:
    """
    Resolve one shot's tracking quality before measuring its geometry.

    :param shot: Prepared shot with shooter and team IDs.
    :param frame: Matching player positions, or None if the frame is missing.
    :param pitch_length: Pitch length in metres.
    :param goal_width: Distance between the goalposts in metres.
    :param detected_only: Exclude inferred positions when True.
    :param include_goalkeeper: Include opposing goalkeepers in defensive features.
    :returns: Feature values, opponent count and tracking status for the shot.
    """
    features: dict[str, float | int | str | None] = dict.fromkeys(SHOT_FEATURES)
    features.update(n_opponents=0, tracking_status="missing_frame")
    if frame is None:
        return features

    valid_players = frame.filter(pl.col("x").is_finite() & pl.col("y").is_finite())
    if detected_only:
        valid_players = valid_players.filter(pl.col("is_detected"))
    shooter_rows = valid_players.filter(pl.col("player_id") == shot["player_id"])
    if shooter_rows.is_empty():
        features["tracking_status"] = "missing_shooter"
        return features

    shooter = shooter_rows.row(0, named=True)
    if shot["team_id"] is None or shooter["team_id"] != shot["team_id"]:
        features["tracking_status"] = "team_mismatch"
        return features
    if shooter["attack_dir"] not in ("R", "L"):
        features["tracking_status"] = "unknown_direction"
        return features

    x, y = shooter["x"], shooter["y"]
    goal_x = pitch_length / 2 * (1 if shooter["attack_dir"] == "R" else -1)
    features["distance_to_goal"] = float(np.hypot(goal_x - x, y))
    opponents = valid_players.filter(pl.col("team_id") != shot["team_id"])
    if not include_goalkeeper:
        opponents = opponents.filter(pl.col("role").is_not_null() & (pl.col("role") != "GK"))
    features["n_opponents"] = opponents.height
    if opponents.is_empty():
        features["tracking_status"] = "no_opponents"
        return features
    if np.isclose(goal_x, x):
        features["tracking_status"] = "degenerate_cone"
        return features

    features.update(_measure_defensive_geometry(x, y, goal_x, opponents, goal_width))
    features["tracking_status"] = "ok"
    return features


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
    :param goal_width: Goal width in metres.
    :param detected_only: Exclude inferred positions when True, including the shooter.
    :param include_goalkeeper: Count the opposing goalkeeper in both defensive features.
    :returns: All shot rows with features, opponent count and an explicit tracking_status.
    """
    if not np.isfinite([pitch_length, goal_width]).all() or min(pitch_length, goal_width) <= 0:
        raise ValueError("Pitch length and goal width must be finite and positive")
    required_shots = {"match_id", "period", "shot_frame", "player_id", "team_id"}
    required_tracking = {"match_id", "period", "frame", "player_id", "team_id",
                         "is_ball", "is_detected", "x", "y", "role", "attack_dir"}
    for df, required in [(shots, required_shots), (tracking, required_tracking)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
    result_cols = SHOT_FEATURES + ["n_opponents", "tracking_status"]
    if set(result_cols) & set(shots.columns):
        raise ValueError("Shots already contain tracking feature columns")

    frame_keys = ["match_id", "period", "frame"]
    needed_frames = shots.select("match_id", "period", pl.col("shot_frame").alias("frame")).unique()
    players = tracking.join(needed_frames, on=frame_keys, how="semi").filter(~pl.col("is_ball"))
    if players.select(frame_keys + ["player_id"]).is_duplicated().any():
        raise ValueError("Tracking must contain one row per match/period/frame/player")
    frames = players.partition_by(frame_keys, as_dict=True)
    rows = []
    for shot in shots.iter_rows(named=True):
        frame = frames.get((shot["match_id"], shot["period"], shot["shot_frame"]))
        rows.append(_compute_shot_features(shot, frame, pitch_length, goal_width, detected_only, include_goalkeeper))
    features = pl.DataFrame(rows, schema={
        "distance_to_goal": pl.Float64, "shot_cone_defenders": pl.Int64,
        "nearest_defender_distance": pl.Float64, "n_opponents": pl.Int64,
        "tracking_status": pl.String,
    })
    return shots.hstack(features)
