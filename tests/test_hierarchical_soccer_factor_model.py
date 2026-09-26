from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymc as pm
import pytest
import pytensor

from src.hierarchical_soccer_factor_model import (build_hierarchical_bernoulli_model,
                                                 build_hierarchical_binomial_model,
                                                 compute_empirical_prior_anchors,
                                                 fit_hierarchical_model,
                                                 summarize_group_effects,
                                                 summarize_sampling_diagnostics)
from src.plots import plot_group_effects, plot_posterior_predictive, plot_prior_predictive
from src.utils import aggregate_binary_outcomes


matplotlib.use("Agg")


@pytest.fixture
def observations() -> pl.DataFrame:
    return pl.DataFrame({
        "player_id": [20, 10, 20, 10] * 6,
        "team_id": [2, 1, 2, 1] * 6,
        "player_position": ["CF", "CF", "LW", "LW"] * 6,
        "is_goal": [0, 1, 0, 0] * 6,
        "distance": [10., 15., 20., 25.] * 6,
        "constant": [1.] * 24,
    })


def test_anchors_fallback_and_extreme_rates(observations: pl.DataFrame) -> None:
    for outcome in [0, 1]:
        anchors = compute_empirical_prior_anchors(observations.with_columns(pl.lit(outcome).alias("is_goal")))
        assert np.isfinite(anchors["mu"])
        assert set(anchors["group_scales"].values()) == {0.5}
        assert set(anchors["eligible_groups"].values()) == {0}
    anchors = compute_empirical_prior_anchors(observations, min_attempts=1)
    expected = np.std([np.log(.5 / 12.5), np.log(6.5 / 6.5)], ddof=1)
    assert anchors["group_scales"]["player"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="empty"):
        compute_empirical_prior_anchors(observations.head(0))


def test_models_validate_counts_and_features(observations: pl.DataFrame) -> None:
    anchors = compute_empirical_prior_anchors(observations)
    groups = aggregate_binary_outcomes(observations, ["player_id", "team_id", "player_position"], "is_goal")
    for bad in [-1, 0.5, 100, float("nan"), None]:
        with pytest.raises(ValueError):
            build_hierarchical_binomial_model(groups.with_columns(pl.lit(bad).alias("successes")), anchors)
    with pytest.raises(ValueError, match="finite"):
        build_hierarchical_bernoulli_model(observations.with_columns(pl.lit(None).alias("distance")),
                                           anchors,
                                           ["distance"])
    with pytest.raises(ValueError, match="null group"):
        build_hierarchical_binomial_model(groups.with_columns(pl.lit(None).alias("player_id")), anchors)


def test_encoding_standardization_and_likelihood(observations: pl.DataFrame) -> None:
    anchors = compute_empirical_prior_anchors(observations)
    groups = aggregate_binary_outcomes(observations, ["player_id", "team_id", "player_position"], "is_goal")
    baseline = build_hierarchical_binomial_model(groups, anchors)
    adjusted = build_hierarchical_bernoulli_model(observations, anchors, ["distance", "constant"])
    assert tuple(adjusted.coords["player"]) == (10, 20)
    np.testing.assert_allclose(adjusted["X"].get_value().mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(adjusted["X"].get_value()[:, 1], 0)
    np.testing.assert_allclose(adjusted["feature_scale"].get_value(), [np.std([10, 15, 20, 25]), 1])
    with pytensor.config.change_flags(cxx=""):
        for model in [baseline, adjusted]:
            assert np.isfinite(model.compile_logp()(model.initial_point()))
        # Prior draws must produce valid probabilities for all aggregate rows.
        with baseline:
            base_p = pm.draw(baseline["p"], draws=1, random_seed=22)
    assert ((base_p > 0) & (base_p < 1)).all()


@pytest.mark.parametrize("adjusted", [False, True])
def test_fit_diagnostics_effects_and_plots(observations: pl.DataFrame, adjusted: bool, tmp_path: Path) -> None:
    anchors = compute_empirical_prior_anchors(observations)
    if adjusted:
        model = build_hierarchical_bernoulli_model(observations, anchors, ["distance"])
    else:
        groups = aggregate_binary_outcomes(observations, ["player_id", "team_id", "player_position"], "is_goal")
        model = build_hierarchical_binomial_model(groups, anchors)
    # Short integration runs verify plumbing, not convergence or substantive inference.
    original_cxx = pytensor.config.cxx
    trace = fit_hierarchical_model(model, draws=30, tune=30, chains=2, cores=1)
    assert pytensor.config.cxx == original_cxx
    diagnostics = summarize_sampling_diagnostics(trace)
    assert diagnostics["divergences"] >= 0
    assert {"r_hat", "ess_bulk", "ess_tail"} <= set(diagnostics["parameters"].columns)
    assert len(diagnostics["bfmi"]) == 2
    for group in ["player", "team", "position"]:
        effects = summarize_group_effects(trace, group)
        assert effects.height == 2
        assert (effects["hdi_low"] <= effects["hdi_high"]).all()
    effects = summarize_group_effects(trace)
    assert set(effects["player"]) == {10, 20}
    for i, (fig, _) in enumerate([
        plot_group_effects(effects, baseline=effects.reverse()),
        plot_prior_predictive(trace), plot_posterior_predictive(trace),
    ]):
        fig.savefig(tmp_path / f"plot_{i}.png")
        assert (tmp_path / f"plot_{i}.png").stat().st_size > 0
        plt.close(fig)
