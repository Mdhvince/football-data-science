from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.hierarchical_soccer_factor_model import prepare_match_shots
from src.utils import build_player_lookup, load_metadata, sanity_check_attack_direction


@pytest.mark.skipif(not Path("data/meta/2004437.json").exists(), reason="Local sample match unavailable")
def test_local_match_coordinates_and_pipeline() -> None:
    meta = load_metadata("data/meta/2004437.json")
    tracking = pl.read_parquet("data/tracking/2004437.parquet")
    sanity_check_attack_direction(tracking, meta, build_player_lookup(meta))
    shots = prepare_match_shots(2004437)
    assert shots.height == 21
    assert shots["is_goal"].sum() == 2
    assert shots["tracking_status"].to_list() == ["ok"] * 21
    aligned = shots.join(tracking.select("match_id", "period", "frame", "player_id", "x", "y"),
                         left_on=["match_id", "period", "shot_frame", "player_id"],
                         right_on=["match_id", "period", "frame", "player_id"])
    assert aligned.height == shots.height
    # Event coordinates are attack-normalized; tracking coordinates keep pitch orientation.
    np.testing.assert_allclose(aligned["x_end"].abs(), aligned["x"].abs(), atol=0.02)
    expected = np.hypot(meta["pitch_length"] / 2 - aligned["x_end"].to_numpy(), aligned["y_end"].to_numpy())
    np.testing.assert_allclose(aligned["distance_to_goal"], expected, atol=0.02)
