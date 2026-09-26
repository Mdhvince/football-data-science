"""
Render football charts on a navy background at 300 DPI.

Plot functions apply PLOT_STYLE locally. Use plt.style.use(PLOT_STYLE) in notebooks
so additional annotations and plots use the same defaults. Save with dpi="figure"
to retain the figure resolution; PNG exports keep the figure background by default.
"""
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


PLOT_COLORS = {
    "background": "#141d2b",
    "foreground": "#c9d4e5",
    "muted": "#8fa3bf",
    "grid": "#233450",
    "blue": "#5b8ff9",
    "orange": "#f6a35c",
    "green": "#61ddaa",
    "purple": "#b6a2e0",
    "red": "#ee6666",
}

PLOT_STYLE = {
    "figure.facecolor": PLOT_COLORS["background"],
    "figure.edgecolor": PLOT_COLORS["background"],
    "figure.dpi": 300,
    "savefig.dpi": "figure",
    "savefig.facecolor": "auto",
    "savefig.edgecolor": "auto",
    "savefig.transparent": False,
    "axes.facecolor": PLOT_COLORS["background"],
    "axes.edgecolor": PLOT_COLORS["muted"],
    "axes.labelcolor": PLOT_COLORS["foreground"],
    "axes.titlecolor": PLOT_COLORS["foreground"],
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.prop_cycle": plt.cycler(color=[PLOT_COLORS[name] for name in ("blue", "orange", "green", "purple", "red")]),
    "text.color": PLOT_COLORS["foreground"],
    "xtick.color": PLOT_COLORS["muted"],
    "ytick.color": PLOT_COLORS["muted"],
    "grid.color": PLOT_COLORS["grid"],
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
    "legend.facecolor": PLOT_COLORS["background"],
    "legend.edgecolor": PLOT_COLORS["grid"],
    "legend.labelcolor": PLOT_COLORS["foreground"],
    "legend.frameon": False,
}


def _format_interval_label(credibility: float | pl.Series | None, interval_kind: str) -> str:
    """
    Label an interval without assuming a missing probability level.

    :param credibility: Probability mass, optionally supplied for each plotted row.
    :param interval_kind: Interval description, such as HDI or credible interval.
    :returns: Label with the probability level, or a note when the level is missing.
    """
    if isinstance(credibility, pl.Series):
        credibility_levels = credibility.unique().to_list()
        if len(credibility_levels) > 1:
            raise ValueError("Plotted intervals must have one consistent credibility level")
        credibility = credibility_levels[0] if credibility_levels else None
    if credibility is None:
        return f"{interval_kind} (level not supplied)"
    if not np.isfinite(credibility) or not 0 < credibility < 1:
        raise ValueError("credibility must be finite and between 0 and 1")
    return f"{100 * credibility:g}% {interval_kind}"


@plt.rc_context(PLOT_STYLE)
def plot_group_effects(group_effects: pl.DataFrame,
                       group_name: str = "player",
                       baseline_effects: pl.DataFrame | None = None,
                       title: str | None = None) -> tuple[Figure, Axes]:
    """
    Plot posterior group effects and optional baseline means on the same log-odds scale.
    Example: "How do player estimates change after adjusting for tracking context?"

    :param group_effects: Group summary with optional credibility; a missing interval level is labelled as unknown.
    :param group_name: Label column identifying each group.
    :param baseline_effects: Baseline means matched by label; missing means are marked on the group labels.
    :param title: Optional model or population context; defaults to the group name and count.
    :returns: Matplotlib figure and axis, without calling show.
    """
    group_effects = group_effects.sort("effect_mean")
    interval_label = _format_interval_label(group_effects.get_column("credibility", default=None), "HDI")
    group_labels = [str(label) for label in group_effects[group_name]]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * group_effects.height)))
    group_positions = np.arange(group_effects.height)
    ax.hlines(group_positions,
              group_effects["hdi_low"],
              group_effects["hdi_high"],
              color=PLOT_COLORS["blue"],
              label=interval_label)
    ax.scatter(group_effects["effect_mean"], group_positions, label="Posterior mean", color=PLOT_COLORS["blue"])
    if baseline_effects is not None:
        aligned_baseline = group_effects.select(group_name).join(
            baseline_effects.select(group_name, "effect_mean"), on=group_name, how="left", validate="1:1",
            maintain_order="left",
        )
        baseline_means = aligned_baseline["effect_mean"].cast(pl.Float64).to_numpy()
        baseline_available = np.isfinite(baseline_means)
        if baseline_available.any():
            ax.scatter(baseline_means[baseline_available],
                       group_positions[baseline_available],
                       marker="D",
                       color=PLOT_COLORS["orange"],
                       label="Baseline mean")
        group_labels = [label if baseline_available[index] else f"{label} (baseline unavailable)"
                        for index, label in enumerate(group_labels)]
    ax.axvline(0, linestyle="--", color=PLOT_COLORS["muted"])
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")
    ax.set_yticks(group_positions, group_labels)
    ax.set(title=title or f"Posterior {group_name} effects ({group_effects.height} groups)",
           xlabel="Group effect (log-odds)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig, ax


@plt.rc_context(PLOT_STYLE)
def plot_prior_predictive(inference_data: az.InferenceData,
                          title: str = "Prior predictive check") -> tuple[Figure, Axes]:
    """
    Display conversion probabilities implied by the prior.
    Example: "Do the priors allow plausible finishing rates?"

    :param inference_data: Inference data containing prior probabilities p for each observation.
    :param title: Chart title identifying the model or population.
    :returns: Matplotlib figure and axis.
    """
    prior_probabilities = inference_data.prior["p"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(prior_probabilities.values.ravel(), bins=40, color=PLOT_COLORS["orange"], alpha=0.7)
    ax.set_title(f"{title}\n{prior_probabilities.sizes['observation']} observations; chains and draws pooled")
    ax.set(xlabel="Prior conversion probability", ylabel="Probability sample count", xlim=(0, 1))
    fig.tight_layout()
    return fig, ax


@plt.rc_context(PLOT_STYLE)
def plot_posterior_predictive(inference_data: az.InferenceData,
                              credibility: float = 0.95,
                              title: str = "Posterior predictive check") -> tuple[Figure, NDArray[np.object_]]:
    """
    Compare observed outcomes with simulated outcomes and total successes.
    Example: "Does the model reproduce the observed number of goals?"

    :param inference_data: Inference data containing observed and posterior predictive outcomes.
    :param credibility: Probability mass for equal-tail predictive intervals.
    :param title: Chart title identifying the model or population.
    :returns: Per-observation and total-success panels; these are checks on training data, not unseen data.
    """
    if not 0 < credibility < 1:
        raise ValueError("credibility must be between 0 and 1")
    predicted_outcomes = inference_data.posterior_predictive["outcome"].transpose("chain", "draw", "observation").values
    observed_outcomes = inference_data.observed_data["outcome"].values
    tail_probability = (1 - credibility) / 2
    interval_low, interval_high = np.quantile(predicted_outcomes, [tail_probability, 1 - tail_probability], axis=(0, 1))
    predicted_means = predicted_outcomes.mean(axis=(0, 1))
    observation_positions = np.arange(1, observed_outcomes.size + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"{title}\n{observed_outcomes.size} observations; in-sample replication")
    axes[0].vlines(observation_positions,
                   interval_low,
                   interval_high,
                   color=PLOT_COLORS["blue"],
                   alpha=0.5,
                   label=f"{100 * credibility:g}% equal-tail predictive interval")
    axes[0].scatter(observation_positions, predicted_means, color=PLOT_COLORS["blue"], label="Predictive mean")
    axes[0].scatter(observation_positions,
                    observed_outcomes,
                    color=PLOT_COLORS["foreground"],
                    marker="x",
                    label="Observed outcome")
    axes[0].set(title="Outcomes by observation", xlabel="Observation (input order)", ylabel="Successes")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.2))
    axes[1].hist(predicted_outcomes.sum(axis=-1).ravel(), bins=30, color=PLOT_COLORS["blue"], alpha=0.7)
    axes[1].axvline(observed_outcomes.sum(), color=PLOT_COLORS["foreground"], linestyle="--", label="Observed total")
    axes[1].set(title="Total successes", xlabel="Replicated total successes", ylabel="Draw count")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].legend()
    fig.tight_layout()
    return fig, axes


@plt.rc_context(PLOT_STYLE)
def plot_rate_estimates(rate_estimates: pl.DataFrame,
                        group_col: str,
                        top_n: int | None = 20,
                        sort_by: str = "post_mean",
                        title: str = "Observed vs Bayesian success rates") -> Axes:
    """
    Compare observed and Bayesian success rates across groups with uncertainty.
    Example: "Which players still look strong after accounting for sample size?"

    :param rate_estimates: Rate estimates with credible intervals and optional credibility metadata.
    :param group_col: Column identifying each player, team, or group.
    :param top_n: Maximum number of groups displayed.
    :param sort_by: Column used to order the groups.
    :param title: Chart title identifying the outcome or population; selection details are added below it.
    :returns: Matplotlib axes comparing raw and Bayesian estimates, with interval levels and selection details.
    """
    required_columns = {group_col, "naive_rate", "post_mean", "ci_low", "ci_high"}

    missing_columns = required_columns - set(rate_estimates.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    displayed_rates = rate_estimates.sort(sort_by, descending=True)

    if top_n is not None:
        displayed_rates = displayed_rates.head(top_n)

    # Reverse so the highest value appears at the top.
    displayed_rates = displayed_rates.reverse()

    group_labels = displayed_rates[group_col].to_list()
    observed_rates = np.array(displayed_rates["naive_rate"].to_list())
    posterior_means = np.array(displayed_rates["post_mean"].to_list())
    interval_low = np.array(displayed_rates["ci_low"].to_list())
    interval_high = np.array(displayed_rates["ci_high"].to_list())

    group_positions = np.arange(len(group_labels))

    interval_label = _format_interval_label(displayed_rates.get_column("credibility", default=None), "credible interval")
    fig, ax = plt.subplots(figsize=(9, max(4, len(group_labels) * 0.35)))
    ax.scatter(observed_rates, group_positions, color=PLOT_COLORS["foreground"], marker="x", label="Observed rate")
    ax.errorbar(posterior_means,
                group_positions,
                xerr=[posterior_means - interval_low, interval_high - posterior_means],
                fmt="o",
                color=PLOT_COLORS["blue"],
                capsize=3,
                label=f"Posterior mean + {interval_label}")

    ax.grid(False, axis="y")
    ax.grid(True, axis="x")
    ax.set_yticks(group_positions)
    ax.set_yticklabels(group_labels)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    selection_label = (f"Top {displayed_rates.height} of {rate_estimates.height}"
                       if displayed_rates.height < rate_estimates.height else f"All {rate_estimates.height}")
    ax.set_title(f"{title}\n{selection_label} {group_col} groups; sorted by {sort_by} (descending)")
    ax.set(xlabel="Success rate", ylabel="")

    ax.legend()
    fig.tight_layout()

    return ax


@plt.rc_context(PLOT_STYLE)
def plot_rate_posterior(rate_estimate: dict[str, Any], label: str | None = None) -> Axes:
    """
    Compare the prior and posterior distributions for one binary success rate.
    Example: "How did this player's observations update the population baseline?"

    :param rate_estimate: Prior, posterior, observed rate, interval bounds and optional credibility metadata.
    :param label: Name displayed in the plot title.
    :returns: Matplotlib axes containing the prior and posterior distributions.
    """
    prior_distribution = beta_dist(rate_estimate["alpha_prior"], rate_estimate["beta_prior"])
    posterior_distribution = beta_dist(rate_estimate["alpha_post"], rate_estimate["beta_post"])
    prior_mean = rate_estimate["prior_mean"]
    observed_rate = rate_estimate["naive_rate"]
    posterior_mean = rate_estimate["post_mean"]
    interval_label = _format_interval_label(rate_estimate.get("credibility"), "credible interval")
    observed_rate_available = observed_rate is not None and np.isfinite(observed_rate)

    # Include both distributions and the observed rate in the plotted range.
    minimum_rate = min(prior_distribution.ppf(0.001), posterior_distribution.ppf(0.001))
    maximum_rate = max(prior_distribution.ppf(0.999), posterior_distribution.ppf(0.999))
    if observed_rate_available:
        minimum_rate, maximum_rate = min(minimum_rate, observed_rate), max(maximum_rate, observed_rate)
    range_padding = (maximum_rate - minimum_rate) * 0.10
    success_rates = np.linspace(max(0, minimum_rate - range_padding), min(1, maximum_rate + range_padding), 500)
    prior_density = prior_distribution.pdf(success_rates)
    posterior_density = posterior_distribution.pdf(success_rates)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(success_rates, prior_density, color=PLOT_COLORS["orange"], linestyle="--", linewidth=2, label="Prior")
    ax.plot(success_rates, posterior_density, color=PLOT_COLORS["blue"], linewidth=2.5, label="Posterior")
    in_credible_interval = (success_rates >= rate_estimate["ci_low"]) & (success_rates <= rate_estimate["ci_high"])
    ax.fill_between(success_rates[in_credible_interval],
                     posterior_density[in_credible_interval],
                     color=PLOT_COLORS["blue"],
                     alpha=0.2,
                     label=interval_label)
    ax.axvline(prior_mean, color=PLOT_COLORS["orange"], linestyle=":", label=f"Prior mean ({prior_mean:.1%})")
    ax.axvline(posterior_mean, color=PLOT_COLORS["blue"], linestyle="--", label=f"Posterior mean ({posterior_mean:.1%})")
    if observed_rate_available:
        ax.axvline(observed_rate,
                   color=PLOT_COLORS["foreground"],
                   linestyle="-.",
                   label=f"Observed rate ({observed_rate:.1%})")

    title = "Prior → posterior"

    if label:
        title = f"{label} — {title}"
    if not observed_rate_available:
        title += "\nObserved rate unavailable"

    ax.set(title=title, xlabel="Success rate", ylabel="Density")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))

    ax.legend()
    fig.tight_layout()

    return ax
