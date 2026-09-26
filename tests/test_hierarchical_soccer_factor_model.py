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
                                                 estimate_prior_parameters,
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


def test_prior_parameters_handle_small_groups_and_extreme_rates(observations: pl.DataFrame) -> None:
    for outcome in [0, 1]:
        prior_parameters = estimate_prior_parameters(observations.with_columns(pl.lit(outcome).alias("is_goal")))
        assert np.isfinite(prior_parameters["mu"])
        assert set(prior_parameters["group_scales"].values()) == {0.5}
        assert set(prior_parameters["eligible_groups"].values()) == {0}
    prior_parameters = estimate_prior_parameters(observations, min_attempts=1)
    expected_player_scale = np.std([np.log(.5 / 12.5), np.log(6.5 / 6.5)], ddof=1)
    assert prior_parameters["group_scales"]["player"] == pytest.approx(expected_player_scale)
    with pytest.raises(ValueError, match="empty"):
        estimate_prior_parameters(observations.head(0))


def test_models_validate_counts_and_features(observations: pl.DataFrame) -> None:
    prior_parameters = estimate_prior_parameters(observations)
    grouped_outcomes = aggregate_binary_outcomes(observations, ["player_id", "team_id", "player_position"], "is_goal")
    for invalid_success_count in [-1, 0.5, 100, float("nan"), None]:
        with pytest.raises(ValueError):
            build_hierarchical_binomial_model(
                grouped_outcomes.with_columns(pl.lit(invalid_success_count).alias("successes")), prior_parameters)
    with pytest.raises(ValueError, match="finite"):
        build_hierarchical_bernoulli_model(observations.with_columns(pl.lit(None).alias("distance")),
                                           prior_parameters,
                                           ["distance"])
    with pytest.raises(ValueError, match="null group"):
        build_hierarchical_binomial_model(grouped_outcomes.with_columns(pl.lit(None).alias("player_id")),
                                          prior_parameters)


def test_encoding_standardization_and_likelihood(observations: pl.DataFrame) -> None:
    prior_parameters = estimate_prior_parameters(observations)
    grouped_outcomes = aggregate_binary_outcomes(observations, ["player_id", "team_id", "player_position"], "is_goal")
    baseline_model = build_hierarchical_binomial_model(grouped_outcomes, prior_parameters)
    adjusted_model = build_hierarchical_bernoulli_model(observations, prior_parameters, ["distance", "constant"])
    assert tuple(adjusted_model.coords["player"]) == (10, 20)
    np.testing.assert_allclose(adjusted_model["X"].get_value().mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(adjusted_model["X"].get_value()[:, 1], 0)
    np.testing.assert_allclose(adjusted_model["feature_scale"].get_value(), [np.std([10, 15, 20, 25]), 1])
    with pytensor.config.change_flags(cxx=""):
        for model in [baseline_model, adjusted_model]:
            assert np.isfinite(model.compile_logp()(model.initial_point()))
        # Prior draws must produce valid probabilities for all grouped rows.
        with baseline_model:
            baseline_probabilities = pm.draw(baseline_model["p"], draws=1, random_seed=22)
    assert ((baseline_probabilities > 0) & (baseline_probabilities < 1)).all()


@pytest.mark.parametrize("use_tracking_features", [False, True])
def test_fit_diagnostics_effects_and_plots(observations: pl.DataFrame,
                                           use_tracking_features: bool,
                                           tmp_path: Path) -> None:
    prior_parameters = estimate_prior_parameters(observations)
    if use_tracking_features:
        model = build_hierarchical_bernoulli_model(observations, prior_parameters, ["distance"])
    else:
        grouped_outcomes = aggregate_binary_outcomes(observations,
                                                     ["player_id", "team_id", "player_position"],
                                                     "is_goal")
        model = build_hierarchical_binomial_model(grouped_outcomes, prior_parameters)
    # Short runs check that fitting works, not that the estimates are reliable.
    original_compiler_path = pytensor.config.cxx
    inference_data = fit_hierarchical_model(model, draws=30, tune=30, chains=2, cores=1)
    assert pytensor.config.cxx == original_compiler_path
    sampling_diagnostics = summarize_sampling_diagnostics(inference_data)
    assert sampling_diagnostics["divergences"] >= 0
    assert {"r_hat", "ess_bulk", "ess_tail"} <= set(sampling_diagnostics["parameters"].columns)
    assert len(sampling_diagnostics["bfmi"]) == 2
    for group_name in ["player", "team", "position"]:
        group_effects = summarize_group_effects(inference_data, group_name)
        assert group_effects.height == 2
        assert (group_effects["hdi_low"] <= group_effects["hdi_high"]).all()
    player_effects = summarize_group_effects(inference_data)
    assert set(player_effects["player"]) == {10, 20}
    for plot_index, (fig, _) in enumerate([
        plot_group_effects(player_effects, baseline_effects=player_effects.reverse()),
        plot_prior_predictive(inference_data), plot_posterior_predictive(inference_data),
    ]):
        fig.savefig(tmp_path / f"plot_{plot_index}.png")
        assert (tmp_path / f"plot_{plot_index}.png").stat().st_size > 0
        plt.close(fig)
