# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Petersburg is a Python framework for modeling and analyzing probabilistic decision processes as directed acyclic graphs (DAGs). It enables simulation, prediction, and analysis of complex decision networks with uncertain outcomes. Pure library: no web app, no services, no database, no required environment variables.

## Stack

- Python >= 3.9 (CI tests 3.9–3.14), packaged with setuptools (`pyproject.toml`)
- uv for environment and dependency management (venv at `.venv/`)
- Runtime deps: numpy, scikit-learn; dev extras: pytest, pytest-cov, ruff, black, mypy

## Commands

```bash
# Install (first time)
uv venv
uv pip install -e ".[dev,examples]"

# Tests
uv run pytest                                # full suite with coverage
uv run pytest --doctest-modules petersburg/  # doctests

# Lint / format / types (ruff and black run in CI as hard gates; mypy is continue-on-error)
uv run ruff check petersburg/ tests/ examples/
uv run black --check petersburg/ tests/ examples/
uv run mypy petersburg/

# Examples — several call plt.show(); always set MPLBACKEND=Agg unattended
MPLBACKEND=Agg uv run python examples/stpetersburg.py
make examples        # basic examples
make case-studies    # 4 case studies in examples/case_studies/

# Docs (Sphinx sources in docs/)
uv pip install -e ".[docs]"
sphinx-build -b html docs docs/_build
```

Notes:

- `examples/print.py` needs a system Graphviz (plus pygraphviz from the `[graphviz]` extra) for its `plot()` call.
- `examples/stpetersburg_w_bankroll.py` simulates 10M games — takes minutes.

## CI (GitHub Actions, `.github/workflows/ci.yml`)

Three jobs run on every push/PR:

1. **test** — Python 3.9–3.14 matrix: ruff, black, mypy (continue-on-error), pytest, doctests.
2. **examples** — Python 3.12, installs only the `[examples]` extra (proving it is sufficient on its own), then runs an explicit allowlist of example scripts with `MPLBACKEND=Agg`. `examples/stpetersburg_w_bankroll.py` is excluded (10M-game simulation); `examples/print.py` runs, but its `plot()` image render needs a system Graphviz the runner does not have — the script catches that failure itself.
3. **build** — `uv build` + `twine check`.

## Core Architecture

### Graph Structure ([graph.py](petersburg/graph.py))

- `Graph.from_dict(d)` builds the graph from a dictionary spec (convention below); `Graph.to_dict()` serializes back — an exact round trip for numeric weights (classifier weights raise `ValidationError`).
- `Graph.from_adj_matrix(A, labels=None, clf_matrix=None)`: a nonzero `A[row, col]` is an edge row → col, so `A[0, 1] = 1` builds the chain `-1 -> 0 -> 1`; a root node (ID `-1`) is added automatically and weights are normalized by row sums.
- `get_outcome()` is one Monte Carlo walk (node payoffs minus edge costs); `get_options()` compares the start node's choices by expected value, with `extended_stats=True` and opt-in `distribution=True` outcome statistics (`std`, `percentiles`, `p_loss`, `var_alpha`, `cvar_alpha`, and raw `samples` with `return_samples=True`).
- `analyze_sensitivity()` / `identify_critical_parameters()` / `print_sensitivity_report()` — automatic sensitivity analysis over edge weights, edge costs, and node payoffs.
- `to_mermaid()` / `to_networkx()` / `plot()` — exports; `plot()` requires the `[graphviz]` extra (pygraphviz) plus a system Graphviz install.

### Dictionary spec convention

Keys are node IDs; each node's `after` list is its **predecessors**. The node with an empty `after` list is where walks start. At each node, the next edge is chosen with probability proportional to `weight` across that node's outgoing edges (normalized to sum to 1); a weight on a node's single outgoing edge has no effect.

### Distribution node types ([nodes.py](petersburg/nodes.py))

`Node` (fixed payoff) plus `UniformNode`, `GaussianNode`, `LogNormalNode`, and `PowerLawNode`, selected with the `type` key and type-specific parameters (`min_payoff`/`max_payoff`, `mean`/`std`, `mu`/`sigma`, `scale`/`alpha`). Weighted choice lives in `weighted_choice()`.

### Estimators ([estimators.py](petersburg/estimators.py))

`FrequencyEstimator` (observed transition frequencies) and `MixedModeEstimator` (frequencies plus per-transition classifiers where at least `min_samples` observations exist) both follow the scikit-learn contract: hyperparameters in `__init__` stored verbatim (`min_samples`, `clf`, `clf_args` for the mixed mode), `classes_` and `n_features_in_` after `fit`, `predict()` returning a one-dimensional array of fitted terminal labels, `predict_proba()` with rows summing to 1, `partial_fit()` for incremental updates, and `clone()`-safe construction. `y` has one column per decision layer.

### Errors ([exceptions.py](petersburg/exceptions.py))

`PetersburgError` is the base class for every error the public API raises. `ValidationError` (also a `ValueError`) covers invalid input values; `SpecValidationError` (also an `AttributeError`) covers invalid graph specifications. Raise these (never bare `AttributeError`/`ValueError`) when adding new validation paths, so `except PetersburgError` keeps working as the single catch-all.

## Conventions

- Every PR appends a CHANGELOG entry under `[Unreleased]` (Keep a Changelog format).
- Conventional commit prefixes (`feat:`, `fix:`, `docs:`, ...).
- New behavior ships with tests; run the full suite locally before pushing — CI is the final gate.
