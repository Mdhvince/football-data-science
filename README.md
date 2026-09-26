# Football data science

The hierarchical factory keeps data preparation in Polars and uses PyMC 5 / ArviZ
InferenceData for inference. Existing tracking loaders, metadata enrichment and
binary aggregation remain in `utils.py`. Model functions and the executable
example live together in `hierarchical_soccer_factor_model.py`; the example runs
only under `if __name__ == "__main__":`.

## Run

```sh
uv sync --group dev
uv run pytest -q
uv run python hierarchical_soccer_factor_model.py
```

The example loads match `2004437`, fits both models on the same usable shots,
prints diagnostics and player/team/position effects, and displays prior,
posterior predictive and effect comparison plots. Default fits use four chains,
1,000 tuning steps and 1,000 retained draws per chain. The bundled match is an
integration example, not enough data to rank player ability reliably.

```python
from hierarchical_soccer_factor_model import (
    prepare_match_shots,
    DEFAULT_GROUP_COLS, compute_empirical_prior_anchors,
    build_hierarchical_binomial_model, build_hierarchical_bernoulli_model,
    fit_hierarchical_model, summarize_sampling_diagnostics, summarize_group_effects,
)
from shot_features import SHOT_FEATURES
from utils import aggregate_binary_outcomes
import polars as pl

shots = prepare_match_shots(2004437)
usable = shots.filter(pl.col("tracking_status") == "ok").drop_nulls(
    list(DEFAULT_GROUP_COLS.values())
)
groups = aggregate_binary_outcomes(usable, list(DEFAULT_GROUP_COLS.values()), "is_goal")
anchors = compute_empirical_prior_anchors(usable)
baseline = build_hierarchical_binomial_model(groups, anchors)
adjusted = build_hierarchical_bernoulli_model(usable, anchors, SHOT_FEATURES)
trace = fit_hierarchical_model(adjusted)
print(summarize_sampling_diagnostics(trace))
print(summarize_group_effects(trace, "player"))
```

## Data conventions

- Shots are `player_possession` events ending in `shot`. `lead_to_goal` supplies
  the binary outcome; missing outcomes raise an error instead of becoming misses.
- `frame_end` is the default shot moment because shooting ends the possession.
  Pass `frame_col="frame_start"` to reproduce the notebook's frame choice. Both
  are exact joins: no nearest-frame substitution. Provider semantics should be
  rechecked for another event feed.
- Shooter and opponent positions both come from the enriched tracking, joined
  by `match_id`, `period`, frame and player ID. Event positions are never mixed
  with tracking positions. In the bundled match the event coordinates are
  attack-normalized, unlike the tracking; event end positions match the shooter
  at `frame_end` after orientation.
- Tracking uses centred metres; `R` attacks positive x and `L` negative x.
  Metadata specifies a 105 m pitch in the sample, so goals are at ±52.5 m.
  Pitch length is required for extraction; goal width defaults to 7.32 m.
  Prepare matches separately before concatenating, especially on different pitches.
- The cone is the triangle from the shooter to both goalposts, including its
  edges. A shooter on the goal line produces a `degenerate_cone` status.
- Inferred positions are retained by default, as in the existing tracking.
  `detected_only=True` excludes them for the shooter and opponents.
  `include_goalkeeper=False` excludes goalkeepers and players with unknown roles
  from both defensive features.
- Missing/nonfinite positions and unknown teams cannot contribute as opponents.
  `n_opponents` exposes the number used; `ok` means computable, not complete
  coverage of all eleven opponents. No available opponent produces null features,
  not zero pressure. All input shots and their `tracking_status` are preserved.
- Attack direction enrichment now uses `team_id/period` for each match.
  Legacy `attack_dir_lookup(directions, team_name_map)` and name-based lookups
  are still accepted. Enrich each match before concatenating.

## Model interpretation

The binomial model estimates additive population, player, team and position
log-odds using non-centred Normal effects and HalfNormal group scales. The
Bernoulli model adds standardized distance, cone count and nearest-opponent
distance with Normal(0, 1) coefficients. Means and scales are retained in model
data and fitted `constant_data`; constant features become zero.

Prior anchors smooth successes and failures by 0.5. Group scales use empirical
logit standard deviations for groups with at least 40 attempts (configurable);
fewer than two qualifying groups or zero variation use a 0.5 fallback. Returned
`eligible_groups` makes this visible. Anchoring on fitting data is empirical
Bayes; use separate reference data or explicit priors when appropriate, and
assess sensitivity to the anchors.

Effects are conditional log-odds deviations, not isolated causal skill or
conversion probabilities. Player/team/position effects can be confounded,
especially with little movement between teams or positions. Inspect divergences,
R-hat, ESS, BFMI and predictive checks before interpretation. In-sample posterior
predictive intervals do not establish held-out calibration. Compare models on
the same shots and priors to avoid selection effects.

Tests include short real NUTS runs for both likelihoods, predictive sampling,
labelled effect summaries and rendered plots. These runs test integration, not
convergence. A local-data test verifies all 21 sample shots and their coordinate
alignment; it skips when the sample metadata is absent.

The fit uses PyTensor's Python backend by default, so no C++ toolchain or
`PYTENSOR_FLAGS` setting is needed to run the example. This avoids the local
macOS `library 'd64' not found` compilation error. Set `use_cxx=True` in
`fit_hierarchical_model` to enable compilation on a working toolchain. The
compiler setting is restored after each fit. The Python backend is slower.
