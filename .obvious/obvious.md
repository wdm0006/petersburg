# petersburg — agent guide

Repo: `wdm0006/petersburg` — a Python framework for analyzing probabilistic
decision processes as directed graphs with automatic sensitivity analysis.
Pure library: no web app, no services, no ports, no required env vars.

## Stack

| Item | Value |
|---|---|
| Language | Python >= 3.9 (CI tests 3.9–3.14; sandbox runs 3.13.14) |
| Package manager | uv (uv.lock committed; venv at `.venv/`) |
| Build backend | setuptools (`pyproject.toml`) |
| Runtime deps | numpy, scikit-learn |
| Dev extras | pytest, pytest-cov, ruff, black, mypy (`.[dev]`) |
| Examples extras | networkx, matplotlib, pandas (`.[examples]` / `.[all]`) |
| CI | GitHub Actions — `.github/workflows/ci.yml` (test + examples + build jobs) |

## Commands

```bash
# Install (repo-preferred)
uv venv
uv pip install -e ".[dev,examples]"   # or just ".[dev]" for library work only

# Tests (193 tests + 72 subtests, ~95% coverage)
uv run pytest
uv run pytest --doctest-modules petersburg/   # doctests

# Lint / format / typecheck
uv run ruff check petersburg/ tests/ examples/
uv run black --check petersburg/ tests/ examples/
uv run mypy petersburg/    # 36 pre-existing errors; CI runs this with continue-on-error

# Examples (several call plt.show() — always set MPLBACKEND=Agg unattended)
MPLBACKEND=Agg uv run python examples/stpetersburg.py
make examples        # stpetersburg, two_envelope_problem, necktie_paradox
make case-studies    # 4 case studies in examples/case_studies/

# Makefile also has: install, test, clean
```

Notes:
- `examples/print.py` needs system Graphviz (pygraphviz) for its `plot()` call.
- `examples/stpetersburg_w_bankroll.py` simulates 10M games — takes minutes.
- CI's examples job is an explicit allowlist (see ci.yml comments).

## Codebase map

See [codebase-map.md](codebase-map.md). One paragraph orientation: `petersburg/`
is the whole library — `graph.py` (Graph, simulation, sensitivity, Mermaid),
`nodes.py` (Node + weighted choice, distribution node types), `edges.py` (Edge),
`estimators.py` (FrequencyEstimator, MixedModeEstimator). `tests/` is the
pytest suite; `examples/` holds runnable demos and case studies.

## Local verification

A healthy dev environment is: fresh venv + `uv pip install -e ".[dev,examples]"`,
then `uv run pytest` green, `uv run ruff check` clean, and one example script
running under `MPLBACKEND=Agg`. No database, no Docker, no env vars required.

## Local Verification Summary

Generated 2026-09-07 by the onboarding worker (sandbox cmp_bPRFaxwz):

- Install: `uv venv` + `uv pip install -e ".[dev,examples]"` — ok (Python 3.13.14, uv 0.12.10)
- `uv run pytest` — **193 passed, 72 subtests passed**, coverage 95% (15.6s)
- `uv run pytest --doctest-modules petersburg/` — 1 passed
- `uv run ruff check petersburg/ tests/ examples/` — all checks passed
- `uv run black --check petersburg/ tests/ examples/` — 31 files unchanged
- `uv run mypy petersburg/` — 36 pre-existing errors (matches CI's continue-on-error state; not a blocker)
- Primary user flow (library): README quick-start graph built from dict,
  10,000 Monte Carlo simulations → expected value $17.72; seeded
  `random_state=7` reproducible; `to_mermaid()` export; lognormal
  distribution node; `identify_critical_parameters()` returned 5 params — all ok
- Examples: `examples/stpetersburg.py`, `examples/necktie_paradox.py`,
  `examples/case_studies/drug_development/analyze.py` all ran to exit 0 under
  `MPLBACKEND=Agg`
- Dev stack: **healthy**

## Snapshot

- Snapshot ID: `iiowqd7s70bnfmyfrl3s3`
- Built at: 2026-09-07T18:54:40.717Z (UTC)
- Contents: repo checkout at master (18f0c02) + `.venv` with dev/examples extras
  installed and verified — future workers can resume from this state.

See [skills/local-dev/SKILL.md](skills/local-dev/SKILL.md) for the onboarding runbook.
