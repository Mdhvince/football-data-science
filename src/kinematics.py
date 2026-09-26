from typing import Any, TypedDict

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.signal import savgol_filter


class KinematicsResult(TypedDict):
    speed_smooth: NDArray[np.float64]
    ax: NDArray[np.float64]
    ay: NDArray[np.float64]
    n_jumps_rejected: int


def summarize_player_kinematics(tracking: pl.DataFrame,
                                player_lookup: pl.DataFrame,
                                match_id: int,
                                player_id: int,
                                fps: float,
                                sprint_threshold_mps: float = 6.0,
                                high_intensity_threshold_mps: float = 5.5) -> dict[str, Any]:
    """
    Compute one player's speed and distance summary for one match.

    :param tracking: Full-match positions with player_id, frame, x and y.
    :param player_lookup: Player metadata with player_id, short_name, team_short and role.
    :param match_id: Match identifier copied into the result.
    :param player_id: Player to summarize.
    :param fps: Frames per second of the tracking data.
    :param sprint_threshold_mps: Speed above which a frame counts as a sprint.
    :param high_intensity_threshold_mps: Speed above which a frame counts as high-intensity running.
    :return: Summary dict, or a dict with an "error" key if the player has fewer than 10 frames.
    """
    seconds_per_frame = 1 / fps

    player_info = player_lookup.filter(pl.col("player_id") == player_id).row(0, named=True)
    player_tracking = tracking.filter(pl.col("player_id") == player_id).sort("frame")
    if player_tracking.height < 10:
        return {
            "match_id": match_id,
            "player": {"player_id": player_id, "name": str(player_info["short_name"])},
            "error": "insufficient frames",
        }

    player_kinematics = compute_kinematics(player_tracking, is_ball=False, fps=fps, smoothing_method="savgol")
    speeds_mps = player_kinematics["speed_smooth"]
    valid_speeds_mps = speeds_mps[np.isfinite(speeds_mps) & (speeds_mps >= 0)]
    high_intensity_speeds_mps = valid_speeds_mps[valid_speeds_mps > high_intensity_threshold_mps]
    sprint_speeds_mps = valid_speeds_mps[valid_speeds_mps > sprint_threshold_mps]

    return {
        "match_id": match_id,
        "player": {
            "player_id": player_id,
            "name": str(player_info["short_name"]),
            "team": str(player_info["team_short"]),
            "role": str(player_info["role"]),
        },
        "n_frames_detected": len(valid_speeds_mps),
        "distance_total_km": round(float(valid_speeds_mps.sum() * seconds_per_frame / 1000), 3),
        "distance_high_intensity_km": round(float(high_intensity_speeds_mps.sum() * seconds_per_frame / 1000), 3),
        "distance_sprint_km": round(float(sprint_speeds_mps.sum() * seconds_per_frame / 1000), 3),
        "smoothed": summarize_speeds(speeds_mps),
        "fraction_frames_sprinting": round(len(sprint_speeds_mps) / max(1, len(valid_speeds_mps)), 4),
    }


def compute_kinematics(positions: pl.DataFrame,
                       is_ball: bool,
                       fps: float,
                       smooth_window_s: float = 0.70,
                       smoothing_method: str = "rolling_median",
                       jump_speed_limit_mps: float = 40.0,
                       speed_limit_mps: float | None = None) -> KinematicsResult:
    """
    Compute smoothed speed and acceleration for one tracked entity.

    Steps:
        1. Reject positions whose per-frame jump speed exceeds jump_speed_limit_mps.
        2. Fill all gaps by linear interpolation and fill the edges.
        3. Smooth with the chosen method.
        4. Use central differences to compute velocity and acceleration.
        5. Mask speeds and accelerations whose window spans a frame gap.
        6. Replace smoothed speeds >= speed_limit_mps with NaN.

    :param positions: Positions sorted by frame, belonging to a single entity (ball or one player).
    :param is_ball: True for ball positions, False for a single player's positions.
    :param fps: Frames per second of the tracking data.
    :param smooth_window_s: Smoothing window in seconds; rounded to an odd number of frames.
    :param smoothing_method: One of "rolling_median", "rolling_mean", "savgol", "ewm".
    :param jump_speed_limit_mps: Per-frame jump speed above which a position is rejected.
    :param speed_limit_mps: Speeds >= this value become NaN; defaults to 40 for the ball, 12 for a player.
    :return: Dict with speed_smooth, ax, ay (arrays, NaN where invalid) and n_jumps_rejected.
    """
    seconds_per_frame = 1.0 / fps
    window_frames = _get_smoothing_window_frames(smooth_window_s, fps)
    if speed_limit_mps is None:
        speed_limit_mps = 40.0 if is_ball else 12.0

    positions = _clean_and_interpolate_positions(positions, seconds_per_frame, jump_speed_limit_mps)
    positions = _add_smoothed_kinematics(positions, smoothing_method, window_frames, seconds_per_frame)
    positions = _mask_kinematics_near_frame_gaps(positions, window_frames)

    smoothed_speeds_mps = positions["speed_smooth"].to_numpy()
    filtered_speeds_mps = np.where(smoothed_speeds_mps >= speed_limit_mps, np.nan, smoothed_speeds_mps)

    return {
        "speed_smooth": filtered_speeds_mps,
        "ax": positions["ax"].to_numpy(),
        "ay": positions["ay"].to_numpy(),
        "n_jumps_rejected": int(positions["is_teleport"].sum()),
    }


def _get_smoothing_window_frames(window_seconds: float, fps: float) -> int:
    """
    Convert a window in seconds to an odd number of frames, minimum three.

    A centred smoothing window needs the same number of frames on each side.
    Three frames give at least one neighbour on each side.
    Example: 0.70 s at 25 fps gives 17.5, rounds to 18, then becomes 19 frames.

    :param window_seconds: Smoothing window in seconds.
    :param fps: Frames per second.
    :return: Odd frame count of at least three.
    """
    window_frames = max(3, int(round(window_seconds * fps)))
    return window_frames + 1 if window_frames % 2 == 0 else window_frames


def _calculate_centred_derivative(column: str, seconds_per_frame: float) -> pl.Expr:
    """
    Build a Polars expression for the centred derivative of a column.
    Uses the difference between the next and previous values over two time steps.

    :param column: Name of the column to differentiate.
    :param seconds_per_frame: Time step between two frames (1 / fps).
    :return: Polars expression for the derivative.
    """
    return (pl.col(column).shift(-1) - pl.col(column).shift(1)) / (2 * seconds_per_frame)


def _clean_and_interpolate_positions(positions: pl.DataFrame,
                                     seconds_per_frame: float,
                                     jump_speed_limit_mps: float) -> pl.DataFrame:
    """
    Remove positions with large jumps, then fill the gaps.

    :param positions: Positions with x and y columns for one entity.
    :param seconds_per_frame: Time step between two frames (1 / fps).
    :param jump_speed_limit_mps: Jump speed above which a position is rejected.
    :return: Positions plus jump_speed, is_teleport, x_clean, y_clean, x_filled, y_filled.
    """
    jump_speed_mps = (pl.col("x").diff() ** 2 + pl.col("y").diff() ** 2).sqrt() / seconds_per_frame
    has_excessive_jump = (jump_speed_mps > jump_speed_limit_mps).fill_null(False)

    return (
        positions
        .with_columns(jump_speed_mps.alias("jump_speed"))
        .with_columns(has_excessive_jump.alias("is_teleport"),
                      pl.when(has_excessive_jump).then(None).otherwise(pl.col("x")).alias("x_clean"),
                      pl.when(has_excessive_jump).then(None).otherwise(pl.col("y")).alias("y_clean"))
        .with_columns(pl.col("x_clean").interpolate().forward_fill().backward_fill().alias("x_filled"),
                      pl.col("y_clean").interpolate().forward_fill().backward_fill().alias("y_filled"))
    )


def smooth_series(series: pl.Series, smoothing_method: str, window_frames: int) -> pl.Series:
    """
    Smooth an interpolated Polars Series with the chosen method.

    :param series: Position series with no nulls (already interpolated).
    :param smoothing_method: One of "rolling_median", "rolling_mean", "savgol", "ewm".
    :param window_frames: Odd smoothing window in frames (see _get_smoothing_window_frames).
    :return: Smoothed Polars Series, same length as series.
    """
    if smoothing_method == "rolling_median":
        return series.rolling_median(window_size=window_frames, center=True, min_samples=1)
    if smoothing_method == "rolling_mean":
        return series.rolling_mean(window_size=window_frames, center=True, min_samples=1)
    if smoothing_method == "savgol":
        return pl.Series(savgol_filter(series.to_numpy(), window_length=window_frames, polyorder=2))
    if smoothing_method == "ewm":
        return series.ewm_mean(span=window_frames, adjust=False)
    raise ValueError(f"Unknown smoothing method: {smoothing_method!r}")


def _add_smoothed_kinematics(positions: pl.DataFrame,
                             smoothing_method: str,
                             window_frames: int,
                             seconds_per_frame: float) -> pl.DataFrame:
    """
    Smooth positions, then compute velocity, acceleration and speed.

    Smoothing comes first because differentiation makes position noise stronger.
    The speed_smooth column is the length of the velocity vector (vx, vy).

    :param positions: Positions with x_filled and y_filled from _clean_and_interpolate_positions.
    :param smoothing_method: Method passed to smooth_series.
    :param window_frames: Odd smoothing window in frames.
    :param seconds_per_frame: Time step between two frames (1 / fps).
    :return: Positions plus x_smooth, y_smooth, vx, vy, ax, ay, speed_smooth.
    """
    return (
        positions
        .with_columns(pl.Series("x_smooth", smooth_series(positions["x_filled"], smoothing_method, window_frames)),
                      pl.Series("y_smooth", smooth_series(positions["y_filled"], smoothing_method, window_frames)))
        .with_columns(_calculate_centred_derivative("x_smooth", seconds_per_frame).alias("vx"),
                      _calculate_centred_derivative("y_smooth", seconds_per_frame).alias("vy"))
        .with_columns(_calculate_centred_derivative("vx", seconds_per_frame).alias("ax"),
                      _calculate_centred_derivative("vy", seconds_per_frame).alias("ay"),
                      (pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt().alias("speed_smooth"))
    )


def _mask_kinematics_near_frame_gaps(positions: pl.DataFrame, window_frames: int) -> pl.DataFrame:
    """
    Remove motion values whose window crosses a gap in frame numbers.

    :param positions: Table with a frame column and motion columns.
    :param window_frames: Odd smoothing window in frames.
    :return: Positions with speed_smooth, ax and ay set to null around gaps.
    """
    if "frame" not in positions.columns:
        return positions

    frame_numbers = positions["frame"].to_numpy()
    gap_indices = np.where(np.diff(frame_numbers) != 1)[0]
    near_frame_gap = np.zeros(positions.height, dtype=bool)
    # +2 also masks the two central-difference neighbours around each gap.
    mask_half_width = window_frames // 2 + 2
    for gap_index in gap_indices:
        mask_start = max(0, gap_index - mask_half_width + 1)
        mask_stop = min(positions.height, gap_index + mask_half_width + 1)
        near_frame_gap[mask_start:mask_stop] = True

    if not near_frame_gap.any():
        return positions

    return (
        positions
        .with_columns(pl.Series("is_invalid", near_frame_gap))
        .with_columns(pl.when(pl.col("is_invalid"))
                      .then(None)
                      .otherwise(pl.col(column)).alias(column) for column in ("speed_smooth", "ax", "ay"))
    )


def summarize_speeds(speeds_mps: NDArray[np.float64]) -> dict[str, float | int]:
    """
    Summarize finite, non-negative speeds with JSON-compatible values.

    :param speeds_mps: Speed samples in metres per second.
    :return: Sample count and speed statistics, or only the count when empty.
    """
    valid_speeds_mps = speeds_mps[np.isfinite(speeds_mps) & (speeds_mps >= 0)]
    if valid_speeds_mps.size == 0:
        return {"n": 0}
    return {
        "n": int(valid_speeds_mps.size),
        "mean": round(float(np.mean(valid_speeds_mps)), 3),
        "p50": round(float(np.percentile(valid_speeds_mps, 50)), 3),
        "p90": round(float(np.percentile(valid_speeds_mps, 90)), 3),
        "p99": round(float(np.percentile(valid_speeds_mps, 99)), 3),
        "max": round(float(np.max(valid_speeds_mps)), 3),
    }


if __name__ == "__main__":
    pass
