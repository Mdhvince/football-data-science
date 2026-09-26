import numpy as np
import polars as pl
import pytest

from shot_features import compute_shot_tracking_features, prepare_shots
from utils import (add_attack_dir,
                   aggregate_binary_outcomes,
                   attack_dir_lookup,
                   attack_direction_from_meta,
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
    result = compute_shot_tracking_features(shots, tracking, 105).row(0, named=True)
    assert result["distance_to_goal"] == 12.5
    assert result["shot_cone_defenders"] == 1
    assert result["nearest_defender_distance"] == 1
    assert result["n_opponents"] == 3
    detected = compute_shot_tracking_features(shots, tracking, 105, detected_only=True)
    assert detected["nearest_defender_distance"][0] == 5
    no_gk = compute_shot_tracking_features(shots, tracking, 105, include_goalkeeper=False)
    assert no_gk["shot_cone_defenders"][0] == 0
    mirrored = tracking.with_columns(-pl.col("x"), -pl.col("y"), pl.lit("L").alias("attack_dir"))
    assert compute_shot_tracking_features(shots, mirrored, 105)["distance_to_goal"][0] == 12.5
    assert compute_shot_tracking_features(shots, mirrored, 105)["shot_cone_defenders"][0] == 1
    # Direction comes from metadata even when shooter is in their own half.
    own_half = tracking.with_columns(pl.when(pl.col("player_id") == 1).then(-10.).otherwise(pl.col("x")).alias("x"))
    assert compute_shot_tracking_features(shots, own_half, 105)["distance_to_goal"][0] == 62.5


@pytest.mark.parametrize("change,status", [
    ({"period": 2}, "missing_frame"), ({"match_id": 2}, "missing_frame"),
    ({"player_id": 99}, "missing_shooter"), ({"team_id": 99}, "team_mismatch"),
])
def test_exact_keys_and_missing_data(change: dict[str, int], status: str) -> None:
    shots = sample_shots().with_columns([pl.lit(value).alias(key) for key, value in change.items()])
    result = compute_shot_tracking_features(shots, sample_tracking(), 105)
    assert result.height == 1
    assert result["tracking_status"][0] == status
    assert result["distance_to_goal"][0] is None


def test_invalid_tracking_and_empty_inputs() -> None:
    shots, tracking = sample_shots(), sample_tracking()
    with pytest.raises(ValueError, match="one row"):
        compute_shot_tracking_features(shots, pl.concat([tracking, tracking.head(1)]), 105)
    unknown = tracking.with_columns(pl.lit("?").alias("attack_dir"))
    assert compute_shot_tracking_features(shots, unknown, 105)["tracking_status"][0] == "unknown_direction"
    invalid = tracking.with_columns(pl.when(pl.col("team_id") == 2).then(float("nan")).otherwise(pl.col("x")).alias("x"))
    result = compute_shot_tracking_features(shots, invalid, 105)
    assert result["tracking_status"][0] == "no_opponents"
    assert result["shot_cone_defenders"][0] is None
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


@pytest.mark.parametrize("values", [[0, 2], [0., .2], [True, None], [0., np.nan]])
def test_aggregate_rejects_non_binary(values: list[bool | int | float | None]) -> None:
    with pytest.raises(ValueError):
        aggregate_binary_outcomes(pl.DataFrame({"group": [1, 1], "success": values}), "group", "success")


def test_attack_direction_id_and_legacy_compatibility() -> None:
    directions = {40: {1: "R", 2: "L"}, 2: {1: "L", 2: "R"}}
    tracking = sample_tracking().with_columns(pl.lit("same name").alias("team_short")).drop("attack_dir")
    enriched = add_attack_dir(tracking, attack_dir_lookup(directions))
    assert enriched["attack_dir"].to_list() == ["R", "L", "L", "L", "R", "Ball"]
    legacy = attack_dir_lookup({40: {1: "R"}}, {40: "same name"})
    assert add_attack_dir(tracking, legacy)["attack_dir"][0] == "R"
    with pytest.raises(ValueError, match="Unknown"):
        attack_direction_from_meta({"home_team": {"id": 40}, "away_team": {"id": 2}, "home_team_side": ["bad"]})


def test_boolean_aggregation_and_multiple_shots_keep_order() -> None:
    result = aggregate_binary_outcomes(pl.DataFrame({"group": [1, 1, 1], "success": [True, False, True]}),
                                       "group",
                                       "success")
    assert result["successes"][0] == 2
    assert result["attempts"][0] == 3
    shots = pl.concat([sample_shots(), sample_shots()]).with_columns(pl.Series("event_id", ["b", "a"]))
    result = compute_shot_tracking_features(shots, sample_tracking(), 105)
    assert result["event_id"].to_list() == ["b", "a"]
    assert result["distance_to_goal"].to_list() == [12.5, 12.5]


def test_enrichment_rejects_mixed_or_wrong_matches() -> None:
    meta = {"id": 1, "home_team": {"id": 40}, "away_team": {"id": 2}, "home_team_side": ["left_to_right"]}
    tracking = sample_tracking().drop("attack_dir")
    with pytest.raises(ValueError, match="one match"):
        enrich_tracking_with_attack_direction(
            pl.concat([tracking, tracking.with_columns(pl.lit(2, dtype=pl.Int64).alias("match_id"))]), meta)
    with pytest.raises(ValueError, match="metadata"):
        enrich_tracking_with_attack_direction(tracking.with_columns(pl.lit(2, dtype=pl.Int64).alias("match_id")), meta)
