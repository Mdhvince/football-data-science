"""
Isolate player's true skill from the team's influence. Useful for player evaluation.
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
from src.utils import (aggregate_binary_outcomes,
                       build_player_lookup,
                       enrich_tracking_with_attack_direction,
                       enrich_tracking_with_player_info,
                       generate_tracking_dataframe,
                       load_metadata,
                       validate_binary_outcomes)


DEFAULT_GROUP_COLS = {"player": "player_id", "team": "team_id", "position": "player_position"}


def compute_empirical_prior_anchors(df: pl.DataFrame,
                                    success_col: str = "is_goal",
                                    group_cols: dict[str, str] | None = None,
                                    min_attempts: int = 40,
                                    correction: float = 0.5,
                                    fallback_scale: float = 0.5,
                                    sigma_mu: float = 0.5) -> dict[str, Any]:
    """
    Anchor logit priors on observed rates with smoothing and sparse-group fallbacks.
    Example: "How much do conversion rates vary between well-observed players?"

    :param df: Observation-level reference data; using fitting data is empirical Bayes.
    :param success_col: Binary outcome column.
    :param group_cols: Mapping from effect name to categorical column.
    :param min_attempts: Minimum group volume for estimating a scale.
    :param correction: Positive pseudo-count added to successes and failures.
    :param fallback_scale: Positive scale when fewer than two groups or no variation remain.
    :param sigma_mu: Prior standard deviation of the population logit.
    :returns: Baseline logit, group scales and eligible group counts.
    """
    group_cols = DEFAULT_GROUP_COLS if group_cols is None else group_cols
    if df.is_empty() or not group_cols:
        raise ValueError("Reference data and group_cols must not be empty")
    if (min_attempts < 1
            or not np.isfinite([correction, fallback_scale, sigma_mu]).all()
            or min(correction, fallback_scale, sigma_mu) <= 0):
        raise ValueError("Prior settings must be finite and positive")
    group_scales, eligible_groups = {}, {}
    for group, column in group_cols.items():
        if df[column].null_count():
            raise ValueError(f"{column!r} contains null group keys")
        rates = aggregate_binary_outcomes(df, column, success_col).filter(pl.col("attempts") >= min_attempts)
        successes = rates["successes"].to_numpy()
        failures = rates["attempts"].to_numpy() - successes
        logits = np.log((successes + correction) / (failures + correction))
        scale = float(np.std(logits, ddof=1)) if len(logits) >= 2 else 0.0
        group_scales[group] = scale if np.isfinite(scale) and scale > 0 else fallback_scale
        eligible_groups[group] = rates.height

    total_successes = df[success_col].sum()
    return {
        "mu": float(np.log((total_successes + correction) / (df.height - total_successes + correction))),
        "sigma_mu": sigma_mu,
        "group_scales": group_scales,
        "eligible_groups": eligible_groups,
    }


def _encode_groups(df: pl.DataFrame, group_cols: dict[str, str]) -> tuple[dict[str, Any], dict[str, NDArray[np.int64]]]:
    """
    Encode sorted group labels as observation indices.

    :param df: Non-empty model observations with non-null group keys.
    :param group_cols: Mapping from effect name to categorical column.
    :returns: Model coordinates and per-observation indices for each group.
    """
    if df.is_empty() or not group_cols:
        raise ValueError("Model data and group_cols must not be empty")
    coordinates: dict[str, Any] = {"observation": np.arange(df.height)}
    group_indices = {}
    for group, column in group_cols.items():
        if df[column].null_count():
            raise ValueError(f"{column!r} contains null group keys")
        labels = df[column].unique().sort().to_list()
        coordinates[group] = labels
        label_indices = {label: index for index, label in enumerate(labels)}
        group_indices[group] = np.array([label_indices[value] for value in df[column]], dtype="int64")
    return coordinates, group_indices


def _standardize_features(df: pl.DataFrame,
                          columns: list[str]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Standardize finite features, using unit scale for constant columns.

    :param df: Model observations containing the features.
    :param columns: Unique numeric feature columns.
    :returns: Standardized values, column means and column scales.
    """
    if len(set(columns)) != len(columns):
        raise ValueError("Feature columns must be unique")
    values = df.select(columns).to_numpy().astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite; select usable tracking rows explicitly")
    means, scales = values.mean(axis=0), values.std(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    return (values - means) / scales, means, scales


def _build_hierarchical_model(df: pl.DataFrame,
                              group_cols: dict[str, str],
                              anchors: dict[str, Any],
                              successes: NDArray[np.integer],
                              attempts: NDArray[np.integer] | None = None,
                              feature_cols: list[str] | None = None) -> pm.Model:
    """
    Build shared group effects and a Bernoulli or Binomial likelihood.

    :param df: Model observations.
    :param group_cols: Mapping from effect name to categorical column.
    :param anchors: Population location and positive prior scales.
    :param successes: Observed binary outcomes or success counts.
    :param attempts: Trial counts for a Binomial likelihood; None selects Bernoulli.
    :param feature_cols: Optional numeric context columns to standardize.
    :returns: Unfitted model with labelled effects and probabilities.
    """
    coordinates, group_indices = _encode_groups(df, group_cols)
    prior_values = [anchors["mu"], anchors["sigma_mu"], *[anchors["group_scales"][group] for group in group_cols]]
    if not np.isfinite(prior_values).all() or min(prior_values[1:]) <= 0:
        raise ValueError("Prior locations must be finite and scales positive")
    if feature_cols:
        features, feature_means, feature_scales = _standardize_features(df, feature_cols)
        coordinates["feature"] = list(feature_cols)

    with pm.Model(coords=coordinates) as model:
        log_odds = pm.Normal("mu", mu=anchors["mu"], sigma=anchors["sigma_mu"])
        for group in group_cols:
            group_scale = pm.HalfNormal(f"sigma_{group}", sigma=anchors["group_scales"][group])
            raw_effect = pm.Normal(f"alpha_{group}_raw", mu=0, sigma=1, dims=group)
            effect = pm.Deterministic(f"alpha_{group}", raw_effect * group_scale, dims=group)
            index = pm.Data(f"{group}_idx", group_indices[group], dims="observation")
            log_odds = log_odds + effect[index]
        if feature_cols:
            pm.Data("feature_mean", feature_means, dims="feature")
            pm.Data("feature_scale", feature_scales, dims="feature")
            feature_data = pm.Data("X", features, dims=("observation", "feature"))
            coefficients = pm.Normal("beta", mu=0, sigma=1, dims="feature")
            log_odds = log_odds + pm.math.dot(feature_data, coefficients)
        probability = pm.Deterministic("p", pm.math.sigmoid(log_odds), dims="observation")
        if attempts is None:
            pm.Bernoulli("outcome", p=probability, observed=successes, dims="observation")
        else:
            pm.Binomial("outcome", n=attempts, p=probability, observed=successes, dims="observation")
    return model


def build_hierarchical_binomial_model(df: pl.DataFrame,
                                      anchors: dict[str, Any],
                                      group_cols: dict[str, str] | None = None,
                                      successes_col: str = "successes",
                                      attempts_col: str = "attempts") -> pm.Model:
    """
    Build a non-centred additive hierarchical model for aggregated binary outcomes.
    Example: "Estimate conversion rates with player, team and position effects."

    :param df: Aggregated observations from aggregate_binary_outcomes.
    :param anchors: Prior settings from compute_empirical_prior_anchors or external knowledge.
    :param group_cols: Mapping from effect name to categorical column.
    :param successes_col: Integer success counts.
    :param attempts_col: Positive integer opportunity counts.
    :returns: Unfitted PyMC model with labelled effects and probabilities.
    """
    group_cols = DEFAULT_GROUP_COLS if group_cols is None else group_cols
    counts = df.select(successes_col, attempts_col).to_numpy().astype(float)
    if not np.isfinite(counts).all() or (counts != np.floor(counts)).any():
        raise ValueError("Counts must be finite integers")
    successes, attempts = counts[:, 0], counts[:, 1]
    if ((successes < 0) | (attempts <= 0) | (successes > attempts)).any():
        raise ValueError("Counts must satisfy 0 <= successes <= attempts and attempts > 0")
    return _build_hierarchical_model(df, group_cols, anchors, successes.astype(int), attempts.astype(int))


def build_hierarchical_bernoulli_model(df: pl.DataFrame,
                                       anchors: dict[str, Any],
                                       feature_cols: list[str],
                                       group_cols: dict[str, str] | None = None,
                                       success_col: str = "is_goal") -> pm.Model:
    """
    Build a hierarchical model adjusted for standardized observation-level context.
    Example: "Compare finishing after accounting for distance and defensive pressure."

    :param df: Complete observation-level data; missing features are rejected.
    :param anchors: Explicit population and group prior settings.
    :param feature_cols: Numeric context columns; constant columns become zero.
    :param group_cols: Mapping from effect name to categorical column.
    :param success_col: Non-null binary outcome column.
    :returns: Unfitted PyMC model retaining feature means and scales as constant data.
    """
    group_cols = DEFAULT_GROUP_COLS if group_cols is None else group_cols
    validate_binary_outcomes(df, list(group_cols.values()), success_col)
    return _build_hierarchical_model(df,
                                     group_cols,
                                     anchors,
                                     df[success_col].cast(pl.Int64).to_numpy(),
                                     feature_cols=feature_cols)


def fit_hierarchical_model(model: pm.Model,
                           draws: int = 1000,
                           tune: int = 1000,
                           chains: int = 4,
                           target_accept: float = 0.9,
                           random_seed: int = 42,
                           cores: int = 1,
                           predictive: bool = True,
                           use_cxx: bool = False) -> az.InferenceData:
    """
    Sample the posterior and optional prior/posterior predictive distributions.
    Example: "Fit the model and check whether it reproduces observed outcomes."

    :param model: Model returned by either hierarchical builder.
    :param draws: Posterior draws per chain.
    :param tune: Adaptation steps per chain.
    :param chains: Number of independent chains.
    :param target_accept: NUTS target acceptance probability.
    :param random_seed: Seed for reproducible sampling.
    :param cores: Number of sampling workers.
    :param predictive: Include 500 prior draws and posterior predictive outcomes.
    :param use_cxx: Enable C++ compilation when a working local toolchain is available.
    :returns: ArviZ inference data including sampler statistics.
    """
    # Keep the compiler setting local to this fit, including predictive sampling.
    cxx = pytensor.config.cxx if use_cxx else ""
    with pytensor.config.change_flags(cxx=cxx), model:
        trace = pm.sample(draws=draws,
                          tune=tune,
                          chains=chains,
                          cores=cores,
                          target_accept=target_accept,
                          random_seed=random_seed,
                          progressbar=False,
                          return_inferencedata=True)
        if predictive:
            trace.extend(pm.sample_prior_predictive(draws=500, random_seed=random_seed))
            pm.sample_posterior_predictive(trace, random_seed=random_seed, progressbar=False, extend_inferencedata=True)
    return trace


def summarize_sampling_diagnostics(trace: az.InferenceData) -> dict[str, Any]:
    """
    Report divergences, R-hat, effective sample sizes and chain energy diagnostics.
    Example: "Are these posterior samples reliable enough to interpret?"

    :param trace: Fitted inference data with sample_stats.
    :returns: Polars parameter summary, divergence count and per-chain BFMI.
    """
    names = [name for name in trace.posterior.data_vars if name != "p" and not name.endswith("_raw")]
    summary = az.summary(trace, var_names=names, kind="diagnostics")
    return {
        "parameters": pl.DataFrame({"parameter": summary.index.to_list(),
                                    **{col: summary[col].to_numpy() for col in summary.columns}}),
        "divergences": int(trace.sample_stats["diverging"].sum()),
        "bfmi": np.asarray(az.bfmi(trace)).tolist(),
    }


def summarize_group_effects(trace: az.InferenceData, group: str = "player", credibility: float = 0.95) -> pl.DataFrame:
    """
    Summarize group deviations on the log-odds scale with posterior uncertainty.
    Example: "Which player effects remain uncertain despite a high observed rate?"

    :param trace: Fitted inference data.
    :param group: Effect name supplied in group_cols.
    :param credibility: Probability mass of each highest-density interval.
    :returns: Group labels, mean effect, HDI bounds, credibility and probability above zero.
    """
    if not 0 < credibility < 1:
        raise ValueError("credibility must be between 0 and 1")
    name = f"alpha_{group}"
    values = trace.posterior[name]
    interval = az.hdi(values, hdi_prob=credibility)[name].transpose(group, "hdi").values
    return pl.DataFrame({
        group: values.coords[group].values,
        "effect_mean": values.mean(("chain", "draw")).values,
        "hdi_low": interval[:, 0], "hdi_high": interval[:, 1],
        "credibility": credibility,
        "prob_above_zero": (values > 0).mean(("chain", "draw")).values,
    }).sort("effect_mean", descending=True)


def prepare_match_shots(match_id: int, data_dir: Path = Path("data"), frame_col: str = "frame_end") -> pl.DataFrame:
    """
    Prepare one match using the existing tracking and metadata helpers.
    Example: "Build the shot context table for Manchester City against Liverpool."

    :param match_id: Match identifier used in local filenames.
    :param data_dir: Directory containing dynamic, tracking and meta folders.
    :param frame_col: Event frame used for tracking; defaults to the end of possession.
    :returns: All shots with tracking features and quality status.
    """
    data_dir = Path(data_dir)
    events = pl.read_parquet(data_dir / "dynamic" / f"{match_id}.parquet")
    shots = prepare_shots(events, frame_col=frame_col)
    tracking = generate_tracking_dataframe(data_dir / "tracking" / f"{match_id}.parquet",
                                           data_dir / "tracking" / f"{match_id}.json",
                                           match_id)
    meta = load_metadata(data_dir / "meta" / f"{match_id}.json")
    player_lookup = build_player_lookup(meta)
    tracking = enrich_tracking_with_player_info(tracking, player_lookup)
    tracking = enrich_tracking_with_attack_direction(tracking, meta)
    return compute_shot_tracking_features(shots, tracking, pitch_length=meta["pitch_length"])


def main() -> None:
    """
    Compare baseline and tracking-adjusted models for the sample match.
    """
    match_id = 2004437
    shots = prepare_match_shots(match_id)
    print(shots.group_by("tracking_status").len())

    # Use the same shots and priors for a meaningful baseline/adjusted comparison.
    group_cols = DEFAULT_GROUP_COLS
    usable = shots.filter(pl.col("tracking_status") == "ok").drop_nulls(list(group_cols.values()))
    print(f"Shots used by both models: {usable.height} / {shots.height}")
    groups = aggregate_binary_outcomes(usable, list(group_cols.values()), "is_goal")
    anchors = compute_empirical_prior_anchors(usable)
    print("Prior anchors (empirical Bayes):", anchors)

    baseline_model = build_hierarchical_binomial_model(groups, anchors)
    adjusted_model = build_hierarchical_bernoulli_model(usable, anchors, SHOT_FEATURES)
    baseline_trace = fit_hierarchical_model(baseline_model)
    adjusted_trace = fit_hierarchical_model(adjusted_model)

    for name, trace in [("baseline", baseline_trace), ("adjusted", adjusted_trace)]:
        print(name, summarize_sampling_diagnostics(trace))
        for group in group_cols:
            print(group, summarize_group_effects(trace, group))
        plot_prior_predictive(trace, title=f"{name.title()} model: prior predictive check, match {match_id}")
        plot_posterior_predictive(trace, title=f"{name.title()} model: posterior predictive check, match {match_id}")

    plot_group_effects(summarize_group_effects(adjusted_trace),
                       baseline=summarize_group_effects(baseline_trace),
                       title=f"Player effects: tracking-adjusted vs baseline, match {match_id}")
    plt.show()


if __name__ == "__main__":
    main()
