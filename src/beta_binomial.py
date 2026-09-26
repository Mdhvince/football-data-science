import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import beta as beta_dist

from src.plots import PLOT_COLORS, PLOT_STYLE, plot_rate_estimates, plot_rate_posterior
from src.utils import DEFAULT_DATA_DIR, aggregate_binary_outcomes


def estimate_binary_rate_by_group(grouped_outcomes: pl.DataFrame,
                                  successes_col: str = "successes",
                                  attempts_col: str = "attempts",
                                  prior_mean: float = 0.10,
                                  prior_strength: float = 20,
                                  credibility: float = 0.95) -> pl.DataFrame:
    """
    Estimate a binary success rate for every group in a table.
    Example: "Which players have the most credible success rates given their sample sizes?"

    :param grouped_outcomes: Table containing successes and attempts for each group.
    :param successes_col: Column containing successful outcomes.
    :param attempts_col: Column containing total opportunities.
    :param prior_mean: Expected success rate before observing each group.
    :param prior_strength: Weight of the prior expressed as virtual observations.
    :param credibility: Probability mass included in the credible interval.
    :returns: Original table with observed rates, posterior estimates, interval bounds and credibility.
    """
    if not 0 < prior_mean < 1:
        raise ValueError("prior_mean must be between 0 and 1")

    if prior_strength <= 0:
        raise ValueError("prior_strength must be > 0")

    if not 0 < credibility < 1:
        raise ValueError("credibility must be between 0 and 1")

    required_columns = {successes_col, attempts_col}
    missing_columns = required_columns - set(grouped_outcomes.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    invalid_counts = grouped_outcomes.filter((pl.col(attempts_col) < 0)
                                             | (pl.col(successes_col) < 0)
                                             | (pl.col(successes_col) > pl.col(attempts_col)))

    if invalid_counts.height > 0:
        raise ValueError("Each row must satisfy 0 <= successes <= attempts")

    prior_alpha, prior_beta = compute_beta_prior_parameters(mean=prior_mean, strength=prior_strength)
    tail_probability = (1 - credibility) / 2

    rate_estimates = grouped_outcomes.with_columns(
        pl.when(pl.col(attempts_col) > 0)
        .then(pl.col(successes_col) / pl.col(attempts_col))
        .otherwise(None)
        .alias("naive_rate"),
        pl.lit(prior_mean).alias("prior_mean"),
        pl.lit(prior_strength).alias("prior_strength"),
        pl.lit(prior_alpha).alias("alpha_prior"),
        pl.lit(prior_beta).alias("beta_prior"),
        pl.lit(credibility).alias("credibility"),
        (prior_alpha + pl.col(successes_col)).alias("alpha_post"),
        (prior_beta + pl.col(attempts_col) - pl.col(successes_col)).alias("beta_post"),
    )
    rate_estimates = rate_estimates.with_columns(
        (pl.col("alpha_post") / (pl.col("alpha_post") + pl.col("beta_post"))).alias("post_mean"))
    posterior_parameters = pl.struct("alpha_post", "beta_post")
    rate_estimates = rate_estimates.with_columns(
        posterior_parameters.map_elements(
            lambda parameters: float(beta_dist.ppf(tail_probability, parameters["alpha_post"], parameters["beta_post"])),
            return_dtype=pl.Float64,
        ).alias("ci_low"),
        posterior_parameters.map_elements(
            lambda parameters: float(beta_dist.ppf(1 - tail_probability,
                                                   parameters["alpha_post"],
                                                   parameters["beta_post"])),
            return_dtype=pl.Float64,
        ).alias("ci_high"),
    )
    return rate_estimates


def estimate_binary_rate(successes: int,
                         attempts: int,
                         prior_mean: float = 0.10,
                         prior_strength: float = 20,
                         credibility: float = 0.95) -> dict[str, float | int]:
    """
    Estimate a binary success rate using Beta-Binomial shrinkage.
    Example: "What success rate should I believe after observing 15 successes in 50 attempts?"

    :param successes: Number of successful outcomes observed.
    :param attempts: Total number of opportunities observed.
    :param prior_mean: Expected success rate before observing the data.
    :param prior_strength: Weight of the prior expressed as virtual observations.
    :param credibility: Probability mass included in the credible interval.
    :returns: Observed rate, posterior estimate, interval bounds, credibility and Beta parameters.
    """
    prior_alpha, prior_beta = compute_beta_prior_parameters(mean=prior_mean, strength=prior_strength)
    posterior_alpha, posterior_beta = compute_beta_posterior_parameters(successes=successes,
                                                                       attempts=attempts,
                                                                       prior_alpha=prior_alpha,
                                                                       prior_beta=prior_beta)
    posterior_summary = summarize_beta_distribution(alpha=posterior_alpha, beta=posterior_beta, credibility=credibility)
    observed_rate = successes / attempts if attempts > 0 else np.nan

    return {
        "successes": successes,
        "attempts": attempts,
        "naive_rate": observed_rate,
        "prior_mean": prior_mean,
        "prior_strength": prior_strength,
        "alpha_prior": prior_alpha,
        "beta_prior": prior_beta,
        "alpha_post": posterior_alpha,
        "beta_post": posterior_beta,
        "post_mean": posterior_summary["mean"],
        "ci_low": posterior_summary["ci_low"],
        "ci_high": posterior_summary["ci_high"],
        "credibility": credibility,
    }


def summarize_beta_distribution(alpha: float, beta: float, credibility: float = 0.95) -> dict[str, float]:
    """
    Summarize a Beta distribution with its mean and credible interval.
    Example: "What is my best estimate and how uncertain is it?"

    :param alpha: Success parameter of the Beta distribution.
    :param beta: Failure parameter of the Beta distribution.
    :param credibility: Probability mass included in the credible interval.
    :returns: Mean and lower and upper bounds of the credible interval.
    """
    if not 0 < credibility < 1:
        raise ValueError("credibility must be between 0 and 1")

    distribution_mean = alpha / (alpha + beta)
    tail_probability = (1 - credibility) / 2
    interval_low, interval_high = beta_dist.ppf([tail_probability, 1 - tail_probability], alpha, beta)
    return {"mean": distribution_mean, "ci_low": interval_low, "ci_high": interval_high}


def compute_beta_posterior_parameters(successes: int,
                                      attempts: int,
                                      prior_alpha: float,
                                      prior_beta: float) -> tuple[float, float]:
    """
    Update Beta parameters with observed successes and failures.
    Example: "How does my prior change after observing 15 successes in 50 attempts?"

    :param successes: Number of successful outcomes observed.
    :param attempts: Total number of opportunities observed.
    :param prior_alpha: Prior virtual successes.
    :param prior_beta: Prior virtual failures.
    :returns: Alpha and beta parameters of the posterior distribution.
    """
    if attempts < 0:
        raise ValueError("attempts must be >= 0")

    if successes < 0 or successes > attempts:
        raise ValueError("successes must satisfy 0 <= successes <= attempts")

    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + (attempts - successes)
    return posterior_alpha, posterior_beta


def compute_beta_prior_parameters(mean: float, strength: float) -> tuple[float, float]:
    """
    Convert a prior mean and strength into Beta distribution parameters.
    Example: "What prior represents a 10% baseline worth 20 observations?"

    :param mean: Expected success rate before observing the data.
    :param strength: Weight of the prior expressed as virtual observations.
    :returns: Alpha and beta parameters of the Beta prior.
    """
    if not 0 < mean < 1:
        raise ValueError("mean must be between 0 and 1")

    if strength <= 0:
        raise ValueError("strength must be > 0")

    prior_alpha = mean * strength
    prior_beta = (1 - mean) * strength
    return prior_alpha, prior_beta


@plt.rc_context(PLOT_STYLE)
def main() -> None:
    """
    Display Bayesian pass-completion estimates using the project's local sample data.
    """
    match_id = 2004437
    events = pl.read_parquet(DEFAULT_DATA_DIR / "dynamic" / f"{match_id}.parquet")
    pass_attempts = (
        events.filter((pl.col("event_type") == "player_possession")
                      & (pl.col("end_type") == "pass")
                      & pl.col("pass_outcome").is_not_null())
        .with_columns((pl.col("pass_outcome") == "successful").alias("is_success"))
    )
    grouped_pass_outcomes = aggregate_binary_outcomes(observations=pass_attempts,
                                                      group_cols="player_name",
                                                      success_col="is_success")
    population_pass_rate = grouped_pass_outcomes.select(pl.col("successes").sum() / pl.col("attempts").sum()).item()
    print(f"Population pass completion: {population_pass_rate:.1%}")

    pass_rate_estimates = estimate_binary_rate_by_group(grouped_outcomes=grouped_pass_outcomes,
                                                       prior_mean=population_pass_rate,
                                                       prior_strength=20,
                                                       credibility=0.95)
    displayed_pass_rates = pass_rate_estimates.filter(pl.col("attempts") >= 10)
    ax = plot_rate_estimates(rate_estimates=displayed_pass_rates,
                            group_col="player_name",
                            top_n=20,
                            sort_by="post_mean",
                            title="Passing reliability (at least 10 attempts)")
    ax.axvline(population_pass_rate,
               color=PLOT_COLORS["orange"],
               linestyle="--",
               alpha=0.7,
               label=f"Population rate ({population_pass_rate:.1%})")
    ax.legend()

    player_name = "Mohamed Salah"
    player_rate_estimate = pass_rate_estimates.filter(pl.col("player_name") == player_name).row(0, named=True)
    plot_rate_posterior(rate_estimate=player_rate_estimate, label=player_name)
    plt.show()


if __name__ == "__main__":
    main()
