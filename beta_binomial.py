import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import beta as beta_dist

from plots import plot_rate_estimates, plot_rate_posterior
from utils import aggregate_binary_outcomes


def estimate_binary_rate_by_group(df: pl.DataFrame,
                                  successes_col: str = "successes",
                                  attempts_col: str = "attempts",
                                  prior_mean: float = 0.10,
                                  prior_strength: float = 20,
                                  credibility: float = 0.95) -> pl.DataFrame:
    """
    Estimate an underlying binary success rate for every group in a table.
    Example: "Which players have the most credible success rates given their sample sizes?"

    :param df: Aggregated table containing successes and attempts for each group.
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

    required = {successes_col, attempts_col}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    invalid = df.filter((pl.col(attempts_col) < 0)
                        | (pl.col(successes_col) < 0)
                        | (pl.col(successes_col) > pl.col(attempts_col)))

    if invalid.height > 0:
        raise ValueError("Each row must satisfy 0 <= successes <= attempts")

    alpha_prior, beta_prior = make_beta_prior(mean=prior_mean, strength=prior_strength)
    tail = (1 - credibility) / 2

    result = df.with_columns(
        pl.when(pl.col(attempts_col) > 0)
        .then(pl.col(successes_col) / pl.col(attempts_col))
        .otherwise(None)
        .alias("naive_rate"),
        pl.lit(prior_mean).alias("prior_mean"),
        pl.lit(prior_strength).alias("prior_strength"),
        pl.lit(alpha_prior).alias("alpha_prior"),
        pl.lit(beta_prior).alias("beta_prior"),
        pl.lit(credibility).alias("credibility"),
        (alpha_prior + pl.col(successes_col)).alias("alpha_post"),
        (beta_prior + pl.col(attempts_col) - pl.col(successes_col)).alias("beta_post"),
    )
    result = result.with_columns(
        (pl.col("alpha_post") / (pl.col("alpha_post") + pl.col("beta_post"))).alias("post_mean"))
    posterior = pl.struct("alpha_post", "beta_post")
    result = result.with_columns(
        posterior.map_elements(lambda row: float(beta_dist.ppf(tail, row["alpha_post"], row["beta_post"])),
                               return_dtype=pl.Float64).alias("ci_low"),
        posterior.map_elements(lambda row: float(beta_dist.ppf(1 - tail, row["alpha_post"], row["beta_post"])),
                               return_dtype=pl.Float64).alias("ci_high"),
    )
    return result


def estimate_binary_rate(successes: int,
                         attempts: int,
                         prior_mean: float = 0.10,
                         prior_strength: float = 20,
                         credibility: float = 0.95) -> dict[str, float | int]:
    """
    Estimate an underlying binary success rate using Beta-Binomial shrinkage.
    Example: "What success rate should I believe after observing 15 successes in 50 attempts?"

    :param successes: Number of successful outcomes observed.
    :param attempts: Total number of opportunities observed.
    :param prior_mean: Expected success rate before observing the data.
    :param prior_strength: Weight of the prior expressed as virtual observations.
    :param credibility: Probability mass included in the credible interval.
    :returns: Observed rate, posterior estimate, interval bounds, credibility and Beta parameters.
    """
    alpha_prior, beta_prior = make_beta_prior(mean=prior_mean, strength=prior_strength)
    alpha_post, beta_post = beta_binomial_posterior(successes=successes,
                                                  attempts=attempts,
                                                  alpha_prior=alpha_prior,
                                                  beta_prior=beta_prior)
    posterior = summarize_beta(alpha=alpha_post, beta_=beta_post, credibility=credibility)
    naive_rate = successes / attempts if attempts > 0 else np.nan

    return {
        "successes": successes,
        "attempts": attempts,
        "naive_rate": naive_rate,
        "prior_mean": prior_mean,
        "prior_strength": prior_strength,
        "alpha_prior": alpha_prior,
        "beta_prior": beta_prior,
        "alpha_post": alpha_post,
        "beta_post": beta_post,
        "post_mean": posterior["mean"],
        "ci_low": posterior["ci_low"],
        "ci_high": posterior["ci_high"],
        "credibility": credibility,
    }


def summarize_beta(alpha: float, beta_: float, credibility: float = 0.95) -> dict[str, float]:
    """
    Summarize a Beta distribution with its mean and credible interval.
    Example: "What is my best estimate and how uncertain is it?"

    :param alpha: Success parameter of the Beta distribution.
    :param beta_: Failure parameter of the Beta distribution.
    :param credibility: Probability mass included in the credible interval.
    :returns: Mean and lower and upper bounds of the credible interval.
    """
    if not 0 < credibility < 1:
        raise ValueError("credibility must be between 0 and 1")

    mean = alpha / (alpha + beta_)
    tail = (1 - credibility) / 2
    ci_low, ci_high = beta_dist.ppf([tail, 1 - tail], alpha, beta_)
    return {"mean": mean, "ci_low": ci_low, "ci_high": ci_high}


def beta_binomial_posterior(successes: int, attempts: int, alpha_prior: float, beta_prior: float) -> tuple[float, float]:
    """
    Update a Beta prior with observed binary successes and failures.
    Example: "How does my prior change after observing 15 successes in 50 attempts?"

    :param successes: Number of successful outcomes observed.
    :param attempts: Total number of opportunities observed.
    :param alpha_prior: Prior virtual successes.
    :param beta_prior: Prior virtual failures.
    :returns: Alpha and beta parameters of the posterior distribution.
    """
    if attempts < 0:
        raise ValueError("attempts must be >= 0")

    if successes < 0 or successes > attempts:
        raise ValueError("successes must satisfy 0 <= successes <= attempts")

    alpha_post = alpha_prior + successes
    beta_post = beta_prior + (attempts - successes)
    return alpha_post, beta_post


def make_beta_prior(mean: float, strength: float) -> tuple[float, float]:
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

    alpha = mean * strength
    beta_ = (1 - mean) * strength
    return alpha, beta_


def main() -> None:
    """
    Display Bayesian pass-completion estimates for the sample match.
    """
    match_id = 2004437
    events = pl.read_parquet(f"data/dynamic/{match_id}.parquet")
    pass_attempts = (
        events.filter((pl.col("event_type") == "player_possession")
                      & (pl.col("end_type") == "pass")
                      & pl.col("pass_outcome").is_not_null())
        .with_columns((pl.col("pass_outcome") == "successful").alias("is_success"))
    )
    pass_rates = aggregate_binary_outcomes(df=pass_attempts, group_cols="player_name", success_col="is_success")
    prior_mean = pass_rates.select(pl.col("successes").sum() / pl.col("attempts").sum()).item()
    print(f"Population pass completion: {prior_mean:.1%}")

    pass_estimates = estimate_binary_rate_by_group(df=pass_rates,
                                                  prior_mean=prior_mean,
                                                  prior_strength=20,
                                                  credibility=0.95)
    plot_data = pass_estimates.filter(pl.col("attempts") >= 10)
    ax = plot_rate_estimates(df=plot_data,
                            group_col="player_name",
                            top_n=20,
                            sort_by="post_mean",
                            title="Passing reliability (at least 10 attempts)")
    ax.axvline(prior_mean, color="orange", linestyle="--", alpha=0.7, label=f"Population rate ({prior_mean:.1%})")
    ax.legend()

    player_name = "Mohamed Salah"
    player = pass_estimates.filter(pl.col("player_name") == player_name).row(0, named=True)
    plot_rate_posterior(row=player, label=player_name)
    plt.show()


if __name__ == "__main__":
    main()
