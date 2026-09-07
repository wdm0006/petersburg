---
name: local-dev
description: How to stand up and verify a working local dev environment for wdm0006/petersburg (Python library, uv, no services)
---

# local-dev — petersburg onboarding runbook

Recorded from a successful onboarding run on 2026-09-07. This repo is a pure
Python library: no servers, databases, or ports. "Local dev" = venv + tests +
lint + a working simulation.

## Steps

1. **Install uv** if missing: `pip3 install uv` (sandbox had Python 3.13.14; repo requires >= 3.9).
2. **Create venv + install** (repo-preferred, per Makefile/README):
   ```bash
   uv venv
   uv pip install -e ".[dev,examples]"
   ```
   `.[dev]` alone is enough for library work; `examples` adds pandas/networkx/matplotlib.
3. **Check for leftover state**: no `.venv` or lock files should pre-exist on a fresh checkout (`make clean` removes them).
4. **Verify**:
   ```bash
   uv run pytest                                   # 193 tests, 72 subtests, ~95% cov
   uv run pytest --doctest-modules petersburg/     # doctests
   uv run ruff check petersburg/ tests/ examples/  # lint
   uv run black --check petersburg/ tests/ examples/
   uv run mypy petersburg/                         # 36 known errors, CI continue-on-error
   ```
5. **Exercise a primary flow** (README quick start):
   ```python
   from petersburg import Graph
   g = Graph()
   g.from_dict({0: {'payoff': 0, 'after': []},
                1: {'payoff': 100, 'after': [{'node_id': 0, 'cost': 10, 'weight': 1.0}]},
                2: {'payoff': -50, 'after': [{'node_id': 0, 'cost': 5, 'weight': 1.0}]},
                3: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 0, 'weight': 0.3},
                                          {'node_id': 2, 'cost': 0, 'weight': 0.7}]}})
   print(sum(g.get_outcome() for _ in range(10000)) / 10000)
   ```
   Also try `Graph(random_state=7)` for reproducibility, `g.to_mermaid()`, and a
   distribution node (`{'type': 'lognormal', 'mu': 4.5, 'sigma': 0.5, ...}`).
6. **Run an example** (always `MPLBACKEND=Agg` unattended — several call `plt.show()`):
   ```bash
   MPLBACKEND=Agg uv run python examples/stpetersburg.py
   MPLBACKEND=Agg uv run python examples/case_studies/drug_development/analyze.py
   ```

## Gotchas

- `examples/print.py` needs system Graphviz + pygraphviz; `examples/stpetersburg_w_bankroll.py` simulates 10M games (minutes).
- mypy reports 36 pre-existing errors — expected; CI runs it with `continue-on-error: true`.
- pytest config (`pyproject.toml`) auto-enables coverage with HTML output to `htmlcov/` (gitignored).
- uv may re-sync the venv to `uv.lock` versions on first `uv run` — that is normal.

## Verified state

Snapshot `iiowqd7s70bnfmyfrl3s3` (2026-09-07T18:54:40.717Z) captured master @
18f0c02 with the venv installed and all checks green. Restore it to skip steps 1–2.
