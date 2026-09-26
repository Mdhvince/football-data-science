import numpy as np
import polars as pl
import pytest

from src.shot_features import compute_shot_tracking_features, prepare_shots
from src.utils import (add_attack_directions,
                       aggregate_binary_outcomes,
                       build_attack_direction_lookup,
                       read_attack_directions_from_metadata,
                       enrich_tracking_with_attack_direction)


def sample_tracking() -> pl.DataFrame:
    return pl.DataFrame({
        "match_id": [1] * 6, "period": [1] * 6, "frame": [10] * 6,
        "player_id": [1, 2, 3, 4, 5, -1], "team_id": [40, 2, 2, 2, 40, None],
        "is_ball": [False] * 5 + [True], "is_detected": [True, True, True, False, True, True],
        "x": [40., 45., 50., 39., 41., 40.1], "y": [0., 0., 10., 0., 0., 0.],
        "role": ["CF", "GK", "CB", "CB", "CF", "ball"], "attack_dir": ["R"] * 6,
    })


def sample_shots() -> pl.DataFrame:
    return pl.DataFrame({"match_id": [1], "period": [1], "shot_frame": [10], "player_id": [1], "team_id": [40]})


def test_geometry_policies_and_direction() -> None:
    shots, tracking = sample_shots(), sample_tracking()
    shot_features = compute_shot_tracking_features(shots, tracking, 105).row(0, named=True)
    assert shot_features["distance_to_goal"] == 12.5
    assert shot_features["shot_cone_defenders"] == 1
    assert shot_features["nearest_defender_distance"] == 1
    assert shot_features["n_opponents"] == 3
    detected_only_features = compute_shot_tracking_features(shots, tracking, 105, detected_only=True)
    assert detected_only_features["nearest_defender_distance"][0] == 5
    features_without_goalkeeper = compute_shot_tracking_features(shots, tracking, 105, include_goalkeeper=False)
    assert features_without_goalkeeper["shot_cone_defenders"][0] == 0
    mirrored_tracking = tracking.with_columns(-pl.col("x"), -pl.col("y"), pl.lit("L").alias("attack_dir"))
    assert compute_shot_tracking_features(shots, mirrored_tracking, 105)["distance_to_goal"][0] == 12.5
    assert compute_shot_tracking_features(shots, mirrored_tracking, 105)["shot_cone_defenders"][0] == 1
    # Direction comes from metadata even when the shooter is in their own half.
    shooter_in_own_half = tracking.with_columns(
        pl.when(pl.col("player_id") == 1).then(-10.).otherwise(pl.col("x")).alias("x"))
    assert compute_shot_tracking_features(shots, shooter_in_own_half, 105)["distance_to_goal"][0] == 62.5


@pytest.mark.parametrize("shot_changes,expected_status", [
    ({"period": 2}, "missing_frame"), ({"match_id": 2}, "missing_frame"),
    ({"player_id": 99}, "missing_shooter"), ({"team_id": 99}, "team_mismatch"),
])
def test_exact_keys_and_missing_data(shot_changes: dict[str, int], expected_status: str) -> None:
    shots = sample_shots().with_columns([pl.lit(value).alias(column) for column, value in shot_changes.items()])
    shot_features = compute_shot_tracking_features(shots, sample_tracking(), 105)
    assert shot_features.height == 1
    assert shot_features["tracking_status"][0] == expected_status
    assert shot_features["distance_to_goal"][0] is None


def test_invalid_tracking_and_empty_inputs() -> None:
    shots, tracking = sample_shots(), sample_tracking()
    with pytest.raises(ValueError, match="one row"):
        compute_shot_tracking_features(shots, pl.concat([tracking, tracking.head(1)]), 105)
    tracking_without_direction = tracking.with_columns(pl.lit("?").alias("attack_dir"))
    shot_features = compute_shot_tracking_features(shots, tracking_without_direction, 105)
    assert shot_features["tracking_status"][0] == "unknown_direction"
    missing_opponent_positions = tracking.with_columns(
        pl.when(pl.col("team_id") == 2).then(float("nan")).otherwise(pl.col("x")).alias("x"))
    shot_features = compute_shot_tracking_features(shots, missing_opponent_positions, 105)
    assert shot_features["tracking_status"][0] == "no_opponents"
    assert shot_features["shot_cone_defenders"][0] is None
    assert compute_shot_tracking_features(shots.head(0), tracking, 105).height == 0


def test_prepare_shots_frame_and_outcome_validation() -> None:
    events = sample_shots().with_columns(
        pl.lit("player_possession").alias("event_type"), pl.lit("shot").alias("end_type"),
        pl.lit(True).alias("lead_to_goal"), pl.lit("CF").alias("player_position"),
        pl.lit(9).alias("frame_start"), pl.lit(10).alias("frame_end"),
    )
    assert prepare_shots(events)["shot_frame"][0] == 10
    assert prepare_shots(events, "frame_start")["shot_frame"][0] == 9
    with pytest.raises(ValueError, match="lead_to_goal"):
        prepare_shots(events.with_columns(pl.lit(None).alias("lead_to_goal")))


@pytest.mark.parametrize("invalid_outcomes", [[0, 2], [0., .2], [True, None], [0., np.nan]])
def test_aggregate_rejects_non_binary(invalid_outcomes: list[bool | int | float | None]) -> None:
    with pytest.raises(ValueError):
        aggregate_binary_outcomes(pl.DataFrame({"group": [1, 1], "success": invalid_outcomes}), "group", "success")


def test_attack_direction_lookup_accepts_team_ids_and_names() -> None:
    attack_directions = {40: {1: "R", 2: "L"}, 2: {1: "L", 2: "R"}}
    tracking = sample_tracking().with_columns(pl.lit("same name").alias("team_short")).drop("attack_dir")
    tracking_with_directions = add_attack_directions(tracking, build_attack_direction_lookup(attack_directions))
    assert tracking_with_directions["attack_dir"].to_list() == ["R", "L", "L", "L", "R", "Ball"]
    name_based_lookup = build_attack_direction_lookup({40: {1: "R"}}, {40: "same name"})
    assert add_attack_directions(tracking, name_based_lookup)["attack_dir"][0] == "R"
    with pytest.raises(ValueError, match="Unknown"):
        read_attack_directions_from_metadata(
            {"home_team": {"id": 40}, "away_team": {"id": 2}, "home_team_side": ["bad"]})


def test_boolean_aggregation_and_multiple_shots_keep_order() -> None:
    grouped_outcomes = aggregate_binary_outcomes(pl.DataFrame({"group": [1, 1, 1], "success": [True, False, True]}),
                                                 "group",
                                                 "success")
    assert grouped_outcomes["successes"][0] == 2
    assert grouped_outcomes["attempts"][0] == 3
    shots = pl.concat([sample_shots(), sample_shots()]).with_columns(pl.Series("event_id", ["b", "a"]))
    shot_features = compute_shot_tracking_features(shots, sample_tracking(), 105)
    assert shot_features["event_id"].to_list() == ["b", "a"]
    assert shot_features["distance_to_goal"].to_list() == [12.5, 12.5]


def test_enrichment_rejects_mixed_or_wrong_matches() -> None:
    metadata = {"id": 1, "home_team": {"id": 40}, "away_team": {"id": 2}, "home_team_side": ["left_to_right"]}
    tracking = sample_tracking().drop("attack_dir")
    with pytest.raises(ValueError, match="one match"):
        enrich_tracking_with_attack_direction(
            pl.concat([tracking, tracking.with_columns(pl.lit(2, dtype=pl.Int64).alias("match_id"))]), metadata)
    with pytest.raises(ValueError, match="metadata"):
        enrich_tracking_with_attack_direction(tracking.with_columns(pl.lit(2, dtype=pl.Int64).alias("match_id")),
                                               metadata)
