# Codebase map — wdm0006/petersburg

Folder-level overview, depth ≤ 2. `petersburg/` is the library; everything else
is tests, demos, or repo plumbing.

| Path | Kind | Purpose |
|---|---|---|
| `petersburg/` | package | The entire library. `graph.py` — `Graph` class, `from_dict`/`from_adj_matrix`, `get_outcome` simulation, sensitivity analysis, Mermaid export. `nodes.py` — `Node`, weighted outcome choice, fixed/uniform/gaussian/lognormal/powerlaw node types. `edges.py` — `Edge` (cost + endpoints). `estimators.py` — scikit-learn style `FrequencyEstimator`, `MixedModeEstimator`. `__init__.py` — public exports. |
| `tests/` | tests | pytest suite: `test_graph.py` (~2,000 lines), `test_distribution_nodes.py`, `test_estimators.py`, `test_node_weights.py`. Run via `uv run pytest`. |
| `examples/` | demos | Runnable scripts (St. Petersburg paradox, necktie paradox, two envelopes, sensitivity, distribution nodes, outsourcing, costwise gradient). Needs `.[examples]` extra and `MPLBACKEND=Agg`. |
| `examples/case_studies/` | demos | Four documented applications, each with `analyze.py`: `drug_development/`, `litigation_strategy/`, `product_launch/`, `startup_funding/`. |
| `examples/estimation/` | demos | `estimator_example.py`, `multimode_estimation.py` — end-to-end estimator usage. |
| `examples/analysis/` | demos | Package marker only (`__init__.py`) — no scripts yet. |
| `examples/img/` | assets | PNG outputs committed from example runs. |
| `.github/workflows/` | CI | `ci.yml` — test matrix (3.9–3.14: ruff, black --check, mypy continue-on-error, pytest, doctests), examples allowlist job, build + twine check. |
| `/` (root) | files | `pyproject.toml` (deps, tool config), `uv.lock`, `Makefile` (install/test/examples/case-studies/clean), `setup.py`, `README.md`, `CLAUDE.md` (architecture guidance), `CHANGELOG.md`, `LICENSE.md` (BSD-3-Clause). |
