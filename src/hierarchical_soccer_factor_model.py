"""
Estimate player, team and position effects on conversion rates, with optional shot context.
"""
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymc as pm
import pytensor
from numpy.typing import NDArray

from src.plots import plot_group_effects, plot_posterior_predictive, plot_prior_predictive
from src.shot_features import SHOT_FEATURES, compute_shot_tracking_features, prepare_shots
from src.utils import (DEFAULT_DATA_DIR,
                       aggregate_binary_outcomes,
                       build_player_lookup,
                       enrich_tracking_with_attack_direction,
                       enrich_tracking_with_player_info,
                       load_tracking_data,
                       load_metadata,
                       validate_binary_outcomes)


DEFAULT_GROUP_COLS = {"player": "player_id", "team": "team_id", "position": "player_position"}


def estimate_prior_parameters(observations: pl.DataFrame,
                              success_col: str = "is_goal",
                              group_cols: dict[str, str] | None = None,
                              min_attempts: int = 40,
                              correction: float = 0.5,
                              fallback_scale: float = 0.5,
                              population_log_odds_sd: float = 0.5) -> dict[str, Any]:
    """
    Estimate prior settings from observed rates, with defaults for small groups.
    Example: "How much do conversion rates vary between players with enough shots?"

    :param observations: Reference data; using fitting data is empirical Bayes.
    :param success_col: Binary outcome column.
    :param group_cols: Mapping from effect name to categorical column.
    :param min_attempts: Minimum group size for estimating a scale.
    :param correction: Positive count added to successes and failures to smooth rates.
    :param fallback_scale: Positive scale when fewer than two groups or no variation remain.
    :param population_log_odds_sd: Prior standard deviation of the population log-odds.
    :returns: Population log-odds, group scales and counts of groups with enough attempts.
    """
    group_cols = DEFAULT_GROUP_COLS if group_cols is None else group_cols
    if observations.is_empty() or not group_cols:
        raise ValueError("Reference data and group_cols must not be empty")
    if (min_attempts < 1
            or not np.isfinite([correction, fallback_scale, population_log_odds_sd]).all()
            or min(correction, fallback_scale, population_log_odds_sd) <= 0):
        raise ValueError("Prior settings must be finite and positive")
    group_scales, groups_with_enough_attempts = {}, {}
    for group_name, group_column in group_cols.items():
        if observations[group_column].null_count():
            raise ValueError(f"{group_column!r} contains null group keys")
        group_counts = aggregate_binary_outcomes(observations, group_column, success_col)
        group_counts = group_counts.filter(pl.col("attempts") >= min_attempts)
        successes = group_counts["successes"].to_numpy()
        failures = group_counts["attempts"].to_numpy() - successes
        group_log_odds = np.log((successes + correction) / (failures + correction))
        group_scale = float(np.std(group_log_odds, ddof=1)) if len(group_log_odds) >= 2 else 0.0
        group_scales[group_name] = group_scale if np.isfinite(group_scale) and group_scale > 0 else fallback_scale
        groups_with_enough_attempts[group_name] = group_counts.height

    total_successes = observations[success_col].sum()
    return {
        "mu": float(np.log((total_successes + correction) / (observations.height - total_successes + correction))),
        "sigma_mu": population_log_odds_sd,
        "group_scales": group_scales,
        "eligible_groups": groups_with_enough_attempts,
    }


def _encode_group_labels(observations: pl.DataFrame,
                         group_cols: dict[str, str]) -> tuple[dict[str, Any], dict[str, NDArray[np.int64]]]:
    """
    Assign an index to each sorted group label.

    :param observations: Non-empty model observations with non-null group keys.
    :param group_cols: Mapping from effect name to categorical column.
    :returns: Model coordinates and group indices for each observation.
    """
    if observations.is_empty() or not group_cols:
        raise ValueError("Model data and group_cols must not be empty")
    coordinates: dict[str, Any] = {"observation": np.arange(observations.height)}
    group_indices = {}
    for group_name, group_column in group_cols.items():
        if observations[group_column].null_count():
            raise ValueError(f"{group_column!r} contains null group keys")
        group_labels = observations[group_column].unique().sort().to_list()
        coordinates[group_name] = group_labels
        label_indices = {label: index for index, label in enumerate(group_labels)}
        group_indices[group_name] = np.array([label_indices[label] for label in observations[group_column]],
                                             dtype="int64")
    return coordinates, group_indices


def _standardize_features(observations: pl.DataFrame,
                          columns: list[str]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Centre and scale finite features, using a scale of one for constant columns.

    :param observations: Model observations containing the features.
    :param columns: Unique numeric feature columns.
    :returns: Scaled feature values, column means and column scales.
    """
    if len(set(columns)) != len(columns):
        raise ValueError("Feature columns must be unique")
    feature_values = observations.select(columns).to_numpy().astype(float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Features must be finite; select tracking rows with complete features")
    feature_means, feature_scales = feature_values.mean(axis=0), feature_values.std(axis=0)
    feature_scales = np.where(feature_scales > 0, feature_scales, 1.0)
    return (feature_values - feature_means) / feature_scales, feature_means, feature_scales


def _build_hierarchical_model(observations: pl.DataFrame,
                              group_cols: dict[str, str],
                              prior_parameters: dict[str, Any],
                              successes: NDArray[np.integer],
                              attempts: NDArray[np.integer] | None = None,
                              feature_cols: list[str] | None = None) -> pm.Model:
    """
    Build shared group effects and a Bernoulli or Binomial likelihood.

    :param observations: Model observations.
    :param group_cols: Mapping from effect name to categorical column.
    :param prior_parameters: Population location and positive prior scales.
    :param successes: Observed binary outcomes or success counts.
    :param attempts: Trial counts for a Binomial likelihood; None selects Bernoulli.
    :param feature_cols: Optional numeric context columns to centre and scale.
    :returns: Unfitted model with labelled effects and probabilities.
    """
    coordinates, group_indices = _encode_group_labels(observations, group_cols)
    prior_values = [prior_parameters["mu"], prior_parameters["sigma_mu"],
                    *[prior_parameters["group_scales"][group_name] for group_name in group_cols]]
    if not np.isfinite(prior_values).all() or min(prior_values[1:]) <= 0:
        raise ValueError("Prior locations must be finite and scales positive")
    if feature_cols:
        standardized_features, feature_means, feature_scales = _standardize_features(observations, feature_cols)
        coordinates["feature"] = list(feature_cols)

    with pm.Model(coords=coordinates) as model:
        log_odds = pm.Normal("mu", mu=prior_parameters["mu"], sigma=prior_parameters["sigma_mu"])
        for group_name in group_cols:
            group_scale = pm.HalfNormal(f"sigma_{group_name}", sigma=prior_parameters["group_scales"][group_name])
            raw_group_effect = pm.Normal(f"alpha_{group_name}_raw", mu=0, sigma=1, dims=group_name)
            group_effect = pm.Deterministic(f"alpha_{group_name}", raw_group_effect * group_scale, dims=group_name)
            observation_group_indices = pm.Data(f"{group_name}_idx", group_indices[group_name], dims="observation")
            log_odds = log_odds + group_effect[observation_group_indices]
        if feature_cols:
            pm.Data("feature_mean", feature_means, dims="feature")
            pm.Data("feature_scale", feature_scales, dims="feature")
            feature_data = pm.Data("X", standardized_features, dims=("observation", "feature"))
            feature_coefficients = pm.Normal("beta", mu=0, sigma=1, dims="feature")
            log_odds = log_odds + pm.math.dot(feature_data, feature_coefficients)
        success_probability = pm.Deterministic("p", pm.math.sigmoid(log_odds), dims="observation")
        if attempts is None:
            pm.Bernoulli("outcome", p=success_probability, observed=successes, dims="observation")
        else:
            pm.Binomial("outcome", n=attempts, p=success_probability, observed=successes, dims="observation")
    return model


def build_hierarchical_binomial_model(grouped_outcomes: pl.DataFrame,
                                      prior_parameters: dict[str, Any],
                                      group_cols: dict[str, str] | None = None,
                                      successes_col: str = "successes",
                                      attempts_col: str = "attempts") -> pm.Model:
    """
    Build a non-centred additive hierarchical model for grouped binary outcomes.
    Example: "Estimate conversion rates with player, team and position effects."

    :param grouped_outcomes: Grouped counts from aggregate_binary_outcomes.
    :param prior_parameters: Settings from estimate_prior_parameters or external knowledge.
    :param group_cols: Mapping from effect name to categorical column.
    :param successes_col: Integer success counts.
    :param attempts_col: Positive integer opportunity counts.
    :returns: Unfitted PyMC model with labelled effects and probabilities.
    """
    group_cols = DEFAULT_GROUP_COLS if group_cols is None else group_cols
    outcome_counts = grouped_outcomes.select(successes_col, attempts_col).to_numpy().astype(float)
    if not np.isfinite(outcome_counts).all() or (outcome_counts != np.floor(outcome_counts)).any():
        raise ValueError("Counts must be finite integers")
    successes, attempts = outcome_counts[:, 0], outcome_counts[:, 1]
    if ((successes < 0) | (attempts <= 0) | (successes > attempts)).any():
        raise ValueError("Counts must satisfy 0 <= successes <= attempts and attempts > 0")
    return _build_hierarchical_model(grouped_outcomes,
                                     group_cols,
                                     prior_parameters,
                                     successes.astype(int),
                                     attempts.astype(int))


def build_hierarchical_bernoulli_model(observations: pl.DataFrame,
                                       prior_parameters: dict[str, Any],
                                       feature_cols: list[str],
                                       group_cols: dict[str, str] | None = None,
                                       success_col: str = "is_goal") -> pm.Model:
    """
    Build a hierarchical model adjusted for the context of each observation.
    Example: "Compare finishing after accounting for distance and defensive pressure."

    :param observations: One row per attempt; missing features are rejected.
    :param prior_parameters: Population and group prior settings.
    :param feature_cols: Numeric context columns; constant columns become zero after scaling.
    :param group_cols: Mapping from effect name to categorical column.
    :param success_col: Non-null binary outcome column.
    :returns: Unfitted PyMC model retaining feature means and scales as constant data.
    """
    group_cols = DEFAULT_GROUP_COLS if group_cols is None else group_cols
    validate_binary_outcomes(observations, list(group_cols.values()), success_col)
    return _build_hierarchical_model(observations,
                                     group_cols,
                                     prior_parameters,
                                     observations[success_col].cast(pl.Int64).to_numpy(),
                                     feature_cols=feature_cols)


def fit_hierarchical_model(model: pm.Model,
                           draws: int = 1000,
                           tune: int = 1000,
                           chains: int = 4,
                           target_accept: float = 0.9,
                           random_seed: int = 42,
                           cores: int = 1,
                           sample_predictions: bool = True,
                           use_cpp_compiler: bool = False) -> az.InferenceData:
    """
    Sample the posterior and optional prior/posterior predictive distributions.
    Example: "Fit the model and check whether it reproduces observed outcomes."

    :param model: Model returned by either hierarchical builder.
    :param draws: Posterior draws per chain.
    :param tune: Adaptation steps per chain.
    :param chains: Number of independent chains.
    :param target_accept: NUTS target acceptance probability.
    :param random_seed: Seed for repeatable sampling.
    :param cores: Number of sampling workers.
    :param sample_predictions: Include 500 prior draws and posterior predictive outcomes.
    :param use_cpp_compiler: Enable C++ compilation when a working local compiler is available.
    :returns: ArviZ inference data including sampler statistics.
    """
    # Keep the compiler setting local to this fit, including predictive sampling.
    compiler_path = pytensor.config.cxx if use_cpp_compiler else ""
    with pytensor.config.change_flags(cxx=compiler_path), model:
        inference_data = pm.sample(draws=draws,
                                   tune=tune,
                                   chains=chains,
                                   cores=cores,
                                   target_accept=target_accept,
                                   random_seed=random_seed,
                                   progressbar=False,
                                   return_inferencedata=True)
        if sample_predictions:
            inference_data.extend(pm.sample_prior_predictive(draws=500, random_seed=random_seed))
            pm.sample_posterior_predictive(inference_data,
                                           random_seed=random_seed,
                                           progressbar=False,
                                           extend_inferencedata=True)
    return inference_data


def summarize_sampling_diagnostics(inference_data: az.InferenceData) -> dict[str, Any]:
    """
    Report divergences, R-hat, effective sample sizes and chain energy checks.
    Example: "Are these posterior samples reliable enough to interpret?"

    :param inference_data: Fitted inference data with sample_stats.
    :returns: Polars parameter summary, divergence count and per-chain BFMI.
    """
    parameter_names = [name for name in inference_data.posterior.data_vars if name != "p" and not name.endswith("_raw")]
    parameter_diagnostics = az.summary(inference_data, var_names=parameter_names, kind="diagnostics")
    return {
        "parameters": pl.DataFrame({"parameter": parameter_diagnostics.index.to_list(),
                                    **{column: parameter_diagnostics[column].to_numpy()
                                       for column in parameter_diagnostics.columns}}),
        "divergences": int(inference_data.sample_stats["diverging"].sum()),
        "bfmi": np.asarray(az.bfmi(inference_data)).tolist(),
    }


def summarize_group_effects(inference_data: az.InferenceData,
                            group_name: str = "player",
                            credibility: float = 0.95) -> pl.DataFrame:
    """
    Summarize group effects on the log-odds scale with posterior uncertainty.
    Example: "Which player effects remain uncertain despite a high observed rate?"

    :param inference_data: Fitted inference data.
    :param group_name: Effect name supplied in group_cols.
    :param credibility: Probability mass of each highest-density interval.
    :returns: Group labels, mean effect, HDI bounds, credibility and probability above zero.
    """
    if not 0 < credibility < 1:
        raise ValueError("credibility must be between 0 and 1")
    parameter_name = f"alpha_{group_name}"
    effect_samples = inference_data.posterior[parameter_name]
    credible_interval = az.hdi(effect_samples, hdi_prob=credibility)[parameter_name].transpose(group_name, "hdi").values
    return pl.DataFrame({
        group_name: effect_samples.coords[group_name].values,
        "effect_mean": effect_samples.mean(("chain", "draw")).values,
        "hdi_low": credible_interval[:, 0], "hdi_high": credible_interval[:, 1],
        "credibility": credibility,
        "prob_above_zero": (effect_samples > 0).mean(("chain", "draw")).values,
    }).sort("effect_mean", descending=True)


def prepare_match_shots(match_id: int, data_dir: Path = DEFAULT_DATA_DIR, frame_col: str = "frame_end") -> pl.DataFrame:
    """
    Prepare one match using the existing tracking and metadata helpers.
    Example: "Build the shot context table for Manchester City against Liverpool."

    :param match_id: Match identifier used in local filenames.
    :param data_dir: Directory containing dynamic, tracking and meta folders; defaults to the project's data directory.
    :param frame_col: Event frame used for tracking; defaults to the end of possession.
    :returns: All shots with identifiers, display names, outcome, end coordinates, tracking features and diagnostics.
    """
    data_dir = Path(data_dir)
    events = pl.read_parquet(data_dir / "dynamic" / f"{match_id}.parquet")
    shots = (
        prepare_shots(events, frame_col=frame_col)
        .select("event_id", "match_id", "period", "shot_frame",
                "player_id", "player_name", "team_id", "team_shortname",
                "player_position", "is_goal", "x_end", "y_end")
    )
    tracking = load_tracking_data(data_dir / "tracking" / f"{match_id}.parquet",
                                 data_dir / "tracking" / f"{match_id}.json",
                                 match_id)
    metadata = load_metadata(data_dir / "meta" / f"{match_id}.json")
    player_lookup = build_player_lookup(metadata)
    tracking = enrich_tracking_with_player_info(tracking, player_lookup)
    tracking = enrich_tracking_with_attack_direction(tracking, metadata)
    return compute_shot_tracking_features(shots, tracking, pitch_length=metadata["pitch_length"])


def main() -> None:
    """
    Compare baseline and tracking-adjusted models for the sample match.
    """
    match_id = 2004437
    shots = prepare_match_shots(match_id)
    print(shots.group_by("tracking_status").len())

    # Use the same shots and priors for a meaningful baseline/adjusted comparison.
    group_cols = DEFAULT_GROUP_COLS
    shots_for_model_comparison = shots.filter(pl.col("tracking_status") == "ok").drop_nulls(list(group_cols.values()))
    grouped_shot_outcomes = aggregate_binary_outcomes(shots_for_model_comparison, list(group_cols.values()), "is_goal")
    prior_parameters = estimate_prior_parameters(shots_for_model_comparison)

    baseline_model = build_hierarchical_binomial_model(grouped_shot_outcomes, prior_parameters)
    adjusted_model = build_hierarchical_bernoulli_model(shots_for_model_comparison, prior_parameters, SHOT_FEATURES)
    baseline_inference = fit_hierarchical_model(baseline_model)
    adjusted_inference = fit_hierarchical_model(adjusted_model)

    for model_name, inference_data in [("baseline", baseline_inference), ("adjusted", adjusted_inference)]:
        print(model_name, summarize_sampling_diagnostics(inference_data))
        for group_name in group_cols:
            print(group_name, summarize_group_effects(inference_data, group_name))
        plot_prior_predictive(inference_data,
                               title=f"{model_name.title()} model: prior predictive check, match {match_id}")
        plot_posterior_predictive(inference_data,
                                  title=f"{model_name.title()} model: posterior predictive check, match {match_id}")

    plot_group_effects(summarize_group_effects(adjusted_inference),
                       baseline_effects=summarize_group_effects(baseline_inference),
                       title=f"Player effects: tracking-adjusted vs baseline, match {match_id}")
    plt.show()


if __name__ == "__main__":
    main()
