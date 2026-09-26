from typing import Any

import arviz as az
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from scipy.stats import beta as beta_dist


def _interval_label(credibility: float | pl.Series | None, kind: str) -> str:
    """
    Label an interval without assuming an unavailable probability level.

    :param credibility: Probability mass, optionally supplied for each plotted row.
    :param kind: Interval description, such as HDI or credible interval.
    :returns: Label with the probability level, or an explicit missing-level note.
    """
    if isinstance(credibility, pl.Series):
        levels = credibility.unique().to_list()
        if len(levels) > 1:
            raise ValueError("Plotted intervals must have one consistent credibility level")
        credibility = levels[0] if levels else None
    if credibility is None:
        return f"{kind} (level not supplied)"
    if not np.isfinite(credibility) or not 0 < credibility < 1:
        raise ValueError("credibility must be finite and between 0 and 1")
    return f"{100 * credibility:g}% {kind}"


def plot_group_effects(effects: pl.DataFrame,
                       group: str = "player",
                       baseline: pl.DataFrame | None = None,
                       title: str | None = None) -> tuple[Figure, Axes]:
    """
    Plot posterior group effects and optional baseline means on the same logit scale.
    Example: "How do player estimates change after adjusting for tracking context?"

    :param effects: Group summary with optional credibility; an absent interval level is labelled as unknown.
    :param group: Label column identifying each group.
    :param baseline: Optional baseline means aligned by label; unavailable means are marked on the group labels.
    :param title: Optional model or population context; defaults to the group name and count.
    :returns: Matplotlib figure and axis, without calling show.
    """
    effects = effects.sort("effect_mean")
    interval_label = _interval_label(effects.get_column("credibility", default=None), "HDI")
    labels = [str(label) for label in effects[group]]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * effects.height)))
    y = np.arange(effects.height)
    ax.hlines(y, effects["hdi_low"], effects["hdi_high"], color="steelblue", label=interval_label)
    ax.scatter(effects["effect_mean"], y, label="Posterior mean", color="steelblue")
    if baseline is not None:
        aligned = effects.select(group).join(
            baseline.select(group, "effect_mean"), on=group, how="left", validate="1:1",
            maintain_order="left",
        )
        baseline_means = aligned["effect_mean"].cast(pl.Float64).to_numpy()
        available = np.isfinite(baseline_means)
        if available.any():
            ax.scatter(baseline_means[available], y[available], marker="D", color="orange", label="Baseline mean")
        labels = [label if available[index] else f"{label} (baseline unavailable)" for index, label in enumerate(labels)]
    ax.axvline(0, linestyle="--", color="grey")
    ax.set_yticks(y, labels)
    ax.set(title=title or f"Posterior {group} effects ({effects.height} groups)", xlabel="Group effect (log-odds)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig, ax


def plot_prior_predictive(trace: az.InferenceData, title: str = "Prior predictive check") -> tuple[Figure, Axes]:
    """
    Display conversion probabilities implied by the prior.
    Example: "Do the priors allow plausible finishing rates?"

    :param trace: Inference data containing prior probabilities p for each observation.
    :param title: Chart title identifying the model or population.
    :returns: Matplotlib figure and axis.
    """
    probabilities = trace.prior["p"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probabilities.values.ravel(), bins=40, color="orange", alpha=0.7)
    ax.set_title(f"{title}\n{probabilities.sizes['observation']} observations; chains and draws pooled")
    ax.set(xlabel="Prior conversion probability", ylabel="Probability sample count", xlim=(0, 1))
    fig.tight_layout()
    return fig, ax


def plot_posterior_predictive(trace: az.InferenceData,
                              credibility: float = 0.95,
                              title: str = "Posterior predictive check") -> tuple[Figure, NDArray[np.object_]]:
    """
    Compare observed outcomes with replicated outcomes and total successes.
    Example: "Does the model reproduce the observed number of goals?"

    :param trace: Inference data containing observed and posterior predictive outcome.
    :param credibility: Probability mass for equal-tail predictive intervals.
    :param title: Chart title identifying the model or population.
    :returns: Per-observation and total-success panels; these show in-sample replication, not held-out calibration.
    """
    if not 0 < credibility < 1:
        raise ValueError("credibility must be between 0 and 1")
    samples = trace.posterior_predictive["outcome"].transpose("chain", "draw", "observation").values
    observed = trace.observed_data["outcome"].values
    tail = (1 - credibility) / 2
    low, high = np.quantile(samples, [tail, 1 - tail], axis=(0, 1))
    mean = samples.mean(axis=(0, 1))
    observation_index = np.arange(1, observed.size + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"{title}\n{observed.size} observations; in-sample replication")
    axes[0].vlines(observation_index,
                   low,
                   high,
                   color="steelblue",
                   alpha=0.5,
                   label=f"{100 * credibility:g}% equal-tail predictive interval")
    axes[0].scatter(observation_index, mean, color="steelblue", label="Predictive mean")
    axes[0].scatter(observation_index, observed, color="black", marker="x", label="Observed outcome")
    axes[0].set(title="Outcomes by observation", xlabel="Observation (input order)", ylabel="Successes")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.2))
    axes[1].hist(samples.sum(axis=-1).ravel(), bins=30, color="steelblue", alpha=0.7)
    axes[1].axvline(observed.sum(), color="black", linestyle="--", label="Observed total")
    axes[1].set(title="Total successes", xlabel="Replicated total successes", ylabel="Draw count")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].legend()
    fig.tight_layout()
    return fig, axes


def plot_rate_estimates(df: pl.DataFrame,
                        group_col: str,
                        top_n: int | None = 20,
                        sort_by: str = "post_mean",
                        title: str = "Observed vs Bayesian success rates") -> Axes:
    """
    Compare observed and Bayesian success rates across groups with uncertainty.
    Example: "Which players still look strong after accounting for sample size?"

    :param df: Rate estimates with credible intervals and optional credibility metadata.
    :param group_col: Column identifying each player, team, or group.
    :param top_n: Maximum number of groups displayed.
    :param sort_by: Column used to order the groups.
    :param title: Chart title identifying the outcome or population; selection details are added below it.
    :returns: Matplotlib axes comparing raw and Bayesian estimates, with interval levels and selection details.
    """
    required = {group_col, "naive_rate", "post_mean", "ci_low", "ci_high"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.sort(sort_by, descending=True)

    if top_n is not None:
        data = data.head(top_n)

    # Reverse so the highest value appears at the top.
    data = data.reverse()

    labels = data[group_col].to_list()
    naive = np.array(data["naive_rate"].to_list())
    posterior = np.array(data["post_mean"].to_list())
    ci_low = np.array(data["ci_low"].to_list())
    ci_high = np.array(data["ci_high"].to_list())

    y = np.arange(len(labels))

    interval_label = _interval_label(data.get_column("credibility", default=None), "credible interval")
    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.35)))
    ax.scatter(naive, y, color="black", marker="x", label="Observed rate")
    ax.errorbar(posterior,
                y,
                xerr=[posterior - ci_low, ci_high - posterior],
                fmt="o",
                color="steelblue",
                capsize=3,
                label=f"Posterior mean + {interval_label}")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    selection = f"Top {data.height} of {df.height}" if data.height < df.height else f"All {df.height}"
    ax.set_title(f"{title}\n{selection} {group_col} groups; sorted by {sort_by} (descending)")
    ax.set(xlabel="Success rate", ylabel="")

    ax.legend()
    fig.tight_layout()

    return ax


def plot_rate_posterior(row: dict[str, Any], label: str | None = None) -> Axes:
    """
    Compare the prior and posterior distributions for one binary success rate.
    Example: "How did this player's observations update the population baseline?"

    :param row: Rate estimate with prior, posterior, observed rate, interval bounds and optional credibility metadata.
    :param label: Name displayed in the plot title.
    :returns: Matplotlib axes containing the prior and posterior distributions.
    """
    prior = beta_dist(row["alpha_prior"], row["beta_prior"])
    posterior = beta_dist(row["alpha_post"], row["beta_post"])
    prior_mean = row["prior_mean"]
    naive_rate = row["naive_rate"]
    post_mean = row["post_mean"]
    interval_label = _interval_label(row.get("credibility"), "credible interval")
    observed_available = naive_rate is not None and np.isfinite(naive_rate)

    # Include both distributions and the observed rate in the plotted range.
    lower = min(prior.ppf(0.001), posterior.ppf(0.001))
    upper = max(prior.ppf(0.999), posterior.ppf(0.999))
    if observed_available:
        lower, upper = min(lower, naive_rate), max(upper, naive_rate)
    padding = (upper - lower) * 0.10
    x = np.linspace(max(0, lower - padding), min(1, upper + padding), 500)
    prior_pdf = prior.pdf(x)
    posterior_pdf = posterior.pdf(x)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, prior_pdf, color="orange", linestyle="--", linewidth=2, label="Prior")
    ax.plot(x, posterior_pdf, color="steelblue", linewidth=2.5, label="Posterior")
    mask = (x >= row["ci_low"]) & (x <= row["ci_high"])
    ax.fill_between(x[mask], posterior_pdf[mask], color="steelblue", alpha=0.2, label=interval_label)
    ax.axvline(prior_mean, color="orange", linestyle=":", label=f"Prior mean ({prior_mean:.1%})")
    ax.axvline(post_mean, color="steelblue", linestyle="--", label=f"Posterior mean ({post_mean:.1%})")
    if observed_available:
        ax.axvline(naive_rate, color="black", linestyle="-.", label=f"Observed rate ({naive_rate:.1%})")

    title = "Prior → posterior"

    if label:
        title = f"{label} — {title}"
    if not observed_available:
        title += "\nObserved rate unavailable"

    ax.set(title=title, xlabel="Success rate", ylabel="Density")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))

    ax.legend()
    fig.tight_layout()

    return ax
