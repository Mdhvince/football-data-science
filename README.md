# Football data science

Application modules are in `src/`. Tests are in `tests/`.

Run commands and open `football_ds.ipynb` from the project root so local `data/` paths resolve correctly.

```bash
uv run pytest
uv run python -m src.beta_binomial
uv run python -m src.hierarchical_soccer_factor_model
```

The examples require the local match data. The hierarchical example also runs posterior sampling.

Import project functions through `src`:

```python
from src.plots import plot_rate_posterior
from src.utils import load_metadata
```
