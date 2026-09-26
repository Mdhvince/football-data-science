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


def compute_player_kinematics(tracking: pl.DataFrame,
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
    player_track = tracking.filter(pl.col("player_id") == player_id).sort("frame")
    if player_track.height < 10:
        return {
            "match_id": match_id,
            "player": {"player_id": player_id, "name": str(player_info["short_name"])},
            "error": "insufficient frames",
        }

    motion = compute_kinematics(player_track, is_ball_df=False, fps=fps, method="savgol")
    speeds_mps = motion["speed_smooth"]
    valid_speeds_mps = speeds_mps[np.isfinite(speeds_mps) & (speeds_mps >= 0)]
    high_intensity_speeds = valid_speeds_mps[valid_speeds_mps > high_intensity_threshold_mps]
    sprint_speeds = valid_speeds_mps[valid_speeds_mps > sprint_threshold_mps]

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
        "distance_high_intensity_km": round(float(high_intensity_speeds.sum() * seconds_per_frame / 1000), 3),
        "distance_sprint_km": round(float(sprint_speeds.sum() * seconds_per_frame / 1000), 3),
        "smoothed": stats_from_speeds(speeds_mps),
        "fraction_frames_sprinting": round(len(sprint_speeds) / max(1, len(valid_speeds_mps)), 4),
    }


def compute_kinematics(df: pl.DataFrame,
                       is_ball_df: bool,
                       fps: float,
                       smooth_window_s: float = 0.70,
                       method: str = "rolling_median",
                       teleportation_threshold: float = 40.0,
                       speed_threshold: float | None = None) -> KinematicsResult:
    """
    Run the full kinematics pipeline for one tracked entity.

    Pipeline:
        1. Teleport reject (per-frame jump > teleportation_threshold -> null).
        2. Interpolate ALL gaps linearly.
        3. Smooth with the chosen method.
        4. Central-difference for velocity + acceleration.
        5. Mask any speed/accel whose smoothing window spans a frame
           discontinuity (half-time, substitution).
        6. Null (do NOT clip) any smoothed speed >= speed_threshold.

    :param df: Positions sorted by frame, belonging to a single entity (ball or one player).
    :param is_ball_df: True if df is the ball, False if df is a single player.
    :param fps: Frames per second of the tracking data.
    :param smooth_window_s: Smoothing window length in seconds; rounded to an odd number of frames.
    :param method: Smoothing method, one of "rolling_median", "rolling_mean", "savgol", "ewm".
    :param teleportation_threshold: Per-frame jump speed above which a position is treated as a teleport and rejected.
    :param speed_threshold: Speeds >= this value become NaN; defaults to 40 for the ball, 12 for a player.
    :return: Dict with speed_smooth, ax, ay (numpy arrays, NaN where invalid) and n_jumps_rejected (int).
    """
    seconds_per_frame = 1.0 / fps
    window_frames = _odd_window_frames(smooth_window_s, fps)
    if speed_threshold is None:
        speed_threshold = 40.0 if is_ball_df else 12.0

    tracks = _clean_and_interpolate_positions(df, seconds_per_frame, teleportation_threshold)
    tracks = _add_smoothed_kinematics(tracks, method, window_frames, seconds_per_frame)
    tracks = _mask_frame_gaps(tracks, window_frames)

    speeds = tracks["speed_smooth"].to_numpy()
    speed_smooth = np.where(speeds >= speed_threshold, np.nan, speeds)

    return {
        "speed_smooth": speed_smooth,
        "ax": tracks["ax"].to_numpy(),
        "ay": tracks["ay"].to_numpy(),
        "n_jumps_rejected": int(tracks["is_teleport"].sum()),
    }


def _odd_window_frames(window_s: float, fps: float) -> int:
    """
    Convert a window in seconds to an odd number of frames, minimum 3.

    Why: rolling median/mean and Savitzky-Golay filters are centred on the
    current frame, so the window must be symmetric. Symmetry requires an odd
    frame count. The minimum of 3 gives at least one neighbor on each side.

    Example: 0.70 s at 25 fps -> 17.5 -> rounds to 18 -> becomes 19 frames.

    :param window_s: Smoothing window in seconds.
    :param fps: Frames per second.
    :return: Odd frame count of at least three.
    """
    frames = max(3, int(round(window_s * fps)))
    return frames + 1 if frames % 2 == 0 else frames


def _calculate_centred_derivative(column: str, seconds_per_frame: float) -> pl.Expr:
    """
    Build a Polars expression for the centred derivative of a column.
    Uses (x[i+1] - x[i-1]) / (2 * dt).

    Example: positions [0.00, 0.04, 0.08] m at 25 fps give a velocity of
    (0.08 - 0.00) / (2 * 0.04) = 1 m/s at the middle frame.

    :param column: Name of the column to differentiate.
    :param seconds_per_frame: Time step between two frames (1 / fps).
    :return: Polars expression for the derivative.
    """
    return (pl.col(column).shift(-1) - pl.col(column).shift(1)) / (2 * seconds_per_frame)


def _clean_and_interpolate_positions(df: pl.DataFrame,
                                     seconds_per_frame: float,
                                     teleportation_threshold: float) -> pl.DataFrame:
    """
    Nullify impossible position jumps, then interpolate the holes.

    :param df: Positions with x and y columns for one entity.
    :param seconds_per_frame: Time step between two frames (1 / fps).
    :param teleportation_threshold: Jump speed above which a position is rejected.
    :return: df plus jump_speed, is_teleport, x_clean, y_clean, x_filled, y_filled.
    """
    jump_speed = (pl.col("x").diff() ** 2 + pl.col("y").diff() ** 2).sqrt() / seconds_per_frame
    is_teleport = (jump_speed > teleportation_threshold).fill_null(False)

    return (
        df
        .with_columns(jump_speed.alias("jump_speed"))
        .with_columns(is_teleport.alias("is_teleport"),
                      pl.when(is_teleport).then(None).otherwise(pl.col("x")).alias("x_clean"),
                      pl.when(is_teleport).then(None).otherwise(pl.col("y")).alias("y_clean"))
        .with_columns(pl.col("x_clean").interpolate().forward_fill().backward_fill().alias("x_filled"),
                      pl.col("y_clean").interpolate().forward_fill().backward_fill().alias("y_filled"))
    )


def smooth_series(series: pl.Series, method: str, window_frames: int) -> pl.Series:
    """
    Smooth a fully interpolated Polars Series with the chosen method.

    :param series: Position series with no nulls (already interpolated).
    :param method: One of "rolling_median", "rolling_mean", "savgol", "ewm".
    :param window_frames: Odd smoothing window in frames (see _odd_window_frames).
    :return: Smoothed Polars Series, same length as series.
    """
    if method == "rolling_median":
        return series.rolling_median(window_size=window_frames, center=True, min_samples=1)
    if method == "rolling_mean":
        return series.rolling_mean(window_size=window_frames, center=True, min_samples=1)
    if method == "savgol":
        return pl.Series(savgol_filter(series.to_numpy(), window_length=window_frames, polyorder=2))
    if method == "ewm":
        return series.ewm_mean(span=window_frames, adjust=False)
    raise ValueError(f"Unknown smoothing method: {method!r}")


def _add_smoothed_kinematics(tracks: pl.DataFrame,
                             method: str,
                             window_frames: int,
                             seconds_per_frame: float) -> pl.DataFrame:
    """
    Smoothen the positions, then derive velocity, acceleration, speed.

    Order matters: smooth first, differentiate after. Raw tracking positions carry pixel-level noise, and each
    derivation multiplies that noise. The smoothed coordinates keep the real motion and drop the noise before the
    central difference is applied. speed_smooth is the norm of (vx, vy).

    :param tracks: DataFrame with x_filled, y_filled from _clean_and_interpolate_positions.
    :param method: Smoothing method passed to smooth_series.
    :param window_frames: Odd smoothing window in frames.
    :param seconds_per_frame: Time step between two frames (1 / fps).
    :return: tracks plus x_smooth, y_smooth, vx, vy, ax, ay, speed_smooth.
    """
    return (
        tracks
        .with_columns(pl.Series("x_smooth", smooth_series(tracks["x_filled"], method, window_frames)),
                      pl.Series("y_smooth", smooth_series(tracks["y_filled"], method, window_frames)))
        .with_columns(_calculate_centred_derivative("x_smooth", seconds_per_frame).alias("vx"),
                      _calculate_centred_derivative("y_smooth", seconds_per_frame).alias("vy"))
        .with_columns(_calculate_centred_derivative("vx", seconds_per_frame).alias("ax"),
                      _calculate_centred_derivative("vy", seconds_per_frame).alias("ay"),
                      (pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt().alias("speed_smooth"))
    )


def _mask_frame_gaps(tracks: pl.DataFrame, window_frames: int) -> pl.DataFrame:
    """
    Nullify kinematics whose window spans a frame-number discontinuity.

    Example: frames [..., 4521, 4522, 6001, 6002] have a gap between index
    of frame 4522 and frame 6001 (second half starts); everything within
    mask_half_width around that boundary is nulled.

    :param tracks: DataFrame with a frame column and kinematics columns.
    :param window_frames: Odd smoothing window in frames.
    :return: tracks with speed_smooth, ax, ay nulled around gaps.
    """
    if "frame" not in tracks.columns:
        return tracks

    frames = tracks["frame"].to_numpy()
    gap_positions = np.where(np.diff(frames) != 1)[0]
    is_invalid = np.zeros(tracks.height, dtype=bool)
    # +2 also masks the two central-difference neighbors around each gap
    mask_half_width = window_frames // 2 + 2
    for gap in gap_positions:
        start = max(0, gap - mask_half_width + 1)
        stop = min(tracks.height, gap + mask_half_width + 1)
        is_invalid[start:stop] = True

    if not is_invalid.any():
        return tracks

    return (
        tracks
        .with_columns(pl.Series("is_invalid", is_invalid))
        .with_columns(pl.when(pl.col("is_invalid"))
                      .then(None)
                      .otherwise(pl.col(column)).alias(column) for column in ("speed_smooth", "ax", "ay"))
    )


def stats_from_speeds(s: NDArray[np.float64]) -> dict[str, float | int]:
    """
    Summarize finite, non-negative speeds with JSON-compatible values.

    :param s: Speed samples in metres per second.
    :return: Sample count and speed statistics, or only the count when empty.
    """
    speeds = s[np.isfinite(s) & (s >= 0)]
    if speeds.size == 0:
        return {"n": 0}
    return {
        "n": int(speeds.size),
        "mean": round(float(np.mean(speeds)), 3),
        "p50": round(float(np.percentile(speeds, 50)), 3),
        "p90": round(float(np.percentile(speeds, 90)), 3),
        "p99": round(float(np.percentile(speeds, 99)), 3),
        "max": round(float(np.max(speeds)), 3),
    }


if __name__ == "__main__":
    pass

