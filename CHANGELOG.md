# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-26

First release published to PyPI since `0.0.1`. The `0.1.0` section below was tagged in the
changelog but never uploaded, so installing from PyPI today still gets `0.0.1`; upgrading to
`0.2.0` picks up everything in both sections.

### Added

- **Distribution-based payoff nodes.** A node's payoff can now be sampled from a distribution
  instead of being a fixed number. `UniformNode`, `GaussianNode`, `LogNormalNode`, and
  `PowerLawNode` are exported from `petersburg`, and `Graph.from_dict()` selects one through a
  per-node `type` key (`"fixed"` (default), `"uniform"`, `"gaussian"`, `"lognormal"`,
  `"powerlaw"`) alongside the matching parameters, e.g.
  `{"type": "gaussian", "mean": 100, "std": 25, "after": [...]}`. The accepted spellings are
  available as `petersburg.graph.NODE_TYPES`. Sensitivity analysis scales these nodes correctly:
  Gaussian scales mean and standard deviation, log-normal shifts `mu` while preserving `sigma`,
  and power-law scales `scale` while preserving `alpha`.
- **Seedable simulation.** `Graph(random_state=...)` accepts an integer seed or an existing
  `numpy.random.Generator` and shares one generator with every node it builds, so edge selection
  and every stochastic payoff draw are reproducible. With the default `None`, graphs continue to
  draw from the global `numpy.random` stream.
- **Reproducible estimator predictions.** `FrequencyEstimator` and `MixedModeEstimator` take
  `random_state=None`, stored verbatim per the scikit-learn `get_params`/`clone` contract and
  passed to the graph each `predict()` builds. With an integer seed, repeated `predict()` calls on
  one fitted estimator return identical output.
- **Estimator `score()`.** Both estimators implement `score(X, y, sample_weight=None)` as accuracy
  between the predicted terminal labels and the final column of the path target. The inherited
  `ClassifierMixin.score()` previously raised a mixed-target `ValueError` on any real two-layer
  target.
- **Feature vectors in option comparison.** `Graph.get_options(..., feature_vector=...)` forwards
  features to classifier-weighted edges, so classifier-backed graphs can compare initial options.
  Previously those edges received `None` and scikit-learn raised.
- **Richer NetworkX export.** `Graph.to_networkx()` attaches each node's `payoff` and each edge's
  `cost` and transition `probability` (`None` for a classifier-weighted edge) alongside the
  pre-existing `weight`.
- **Sensitivity analysis controls and reporting.** `analyze_sensitivity()`,
  `identify_critical_parameters()`, and `print_sensitivity_report()` accept `max_params`
  (default 10, `None` for unlimited), applied per parameter type. Results report
  `candidate_parameters`, `parameters_analyzed`, and `max_params`, and the printed report says
  `Parameters Analyzed: N of M` so a truncated analysis is no longer silent.
  `analyze_sensitivity()` also accepts a precomputed `baseline_ev`.
- **Public validation helpers.** `petersburg.graph.NODE_TYPES`,
  `petersburg.graph.SENSITIVITY_PARAMETER_TYPES`, `petersburg.graph.validate_sample_count()`, and
  `petersburg.nodes.is_numeric_weight()`.
- **New optional extras.** `examples` (pandas plus `petersburg[visualization]`, everything the
  bundled scripts need) and `all` (`dev`, `visualization`, `docs`, `examples`).

### Changed

- **BREAKING** — `FrequencyEstimator.predict()` and `MixedModeEstimator.predict()` now return the
  fitted label of the simulated terminal category rather than that category's internal index into
  `_categories`. The returned array is object-dtype, so non-numeric (e.g. string) labels round-trip
  instead of being coerced into a float array. *Migration:* if you wrote `labels[int(pred)]`, drop
  the lookup and use `pred` directly; if you bucketed the float indices (e.g. with
  `np.histogram`), count the labels instead. This matches the scikit-learn contract that `predict`
  returns fitted labels, and is what makes `score()` meaningful.
- **BREAKING** — `Graph.get_options()` no longer collapses parallel first options. When two or more
  of the start node's edges reach the *same* destination, the second and later are keyed
  `(node_id, occurrence)` (occurrence counting from 1) instead of silently overwriting the earlier
  entry. *Migration:* graphs whose initial options all have distinct destinations are unaffected and
  keep plain node-id keys; if you have parallel first options, expect tuple keys for the duplicates
  — and note you were previously losing one of those options entirely.
- **BREAKING** — `Graph.get_options()` values now include one sample of the start node's own payoff,
  putting option values on the same scale as `get_outcome()`. *Migration:* option values shift by
  the start payoff. Graphs with the usual payoff-0 start node see no change; relative ranking is
  unaffected in every case.
- **BREAKING** — `Graph.to_mermaid()` emits sanitized, prefixed node identifiers
  (`node_<sanitized-id>`, collisions suffixed from 2) with the original id preserved in the quoted
  display label. Previously the raw node id was interpolated as the Mermaid identifier, so ids
  containing a space, `$`, parentheses, a quote, or the Mermaid keywords `end`/`graph` produced
  plausible-looking text that no renderer would parse. *Migration:* text that matches on the
  emitted identifier (e.g. `0(` or `class 0 terminal`) must match `node_0` instead. Diagram
  appearance is unchanged.
- **BREAKING** — Python 3.8 is no longer supported; `requires-python` is `>=3.9`. CI covers
  3.9 through 3.14. Nothing is known to break on 3.8; the claim was simply untested.
- **Invalid inputs now raise descriptive errors instead of silently misbehaving.** Each of these was
  previously accepted and produced a wrong answer, an opaque `KeyError`/`IndexError`/`AttributeError`
  from deep inside the library, or a well-formed but empty report. Code that was passing valid input
  is unaffected.
  - `Graph.from_dict()` rejects unknown node `type` values, `after` entries naming a node id absent
    from the specification, and cyclic specifications (with the cycle path in the message). A failed
    rebuild leaves any existing `start_node` intact.
  - `Graph.from_adj_matrix()` coerces array-likes with `np.asarray` and rejects non-2D, non-square,
    non-real-numeric, infinite, and negative matrices before building anything. `NaN` is still the
    absent-edge marker.
  - Transition weights are validated at selection time: each must be finite and non-negative and
    their total positive and finite, whether it came from `add_outcome`, a dictionary spec, an
    adjacency matrix, or a classifier's `predict_proba`. A single zero weight beside a positive one
    is still valid.
  - `analyze_sensitivity()` rejects an unrecognized `parameter_type` (previously a typo such as
    `"payoff"` for `"payoffs"` returned an empty report after burning the baseline simulations) and
    requires `0 < perturbation < 1`.
  - Sample counts are validated at the API boundary: `Graph.get_options(iters=...)`,
    `analyze_sensitivity(num_simulations=...)`, `identify_critical_parameters(num_simulations=...)`,
    and both estimators' `num_simulations` must be positive integers.
    `Graph.get_outcome(iters=...)` is deliberately excluded — zero bankroll iterations coherently
    returns the starting bank.
  - Estimator path targets (`y`) must be a populated 2D array with at least two decision layers, at
    `fit()` and `partial_fit()`, before any fitted state is touched. Plain lists of lists are now
    accepted where they previously failed on `.shape`.
  - Estimator feature matrices (`X`) must be 2D at `predict()`, `score()`, and
    `MixedModeEstimator.fit()`, which additionally requires `X` and `y` to have the same number of
    rows. `FrequencyEstimator.fit(None, y)` is unchanged and still documented as ignoring `X`.
  - Calling `predict()` or `score()` before `fit()` raises scikit-learn's `NotFittedError` naming
    the class, instead of `AttributeError: 'NoneType' object has no attribute 'shape'`.
  - `FrequencyEstimator.partial_fit()` raises a `ValueError` naming the estimator, value, and column
    for a category outside the fitted set, instead of a bare `(0, 'b') is not in list`.
- **Transition probabilities in exports are normalized.** `Graph.to_networkx()` edge `probability`
  values sum to one per fully numeric source, and `Graph.to_mermaid()` labels multi-way branches
  with those normalized values. Previously the raw relative weight was exported, so a 3/1 branch
  reported `3` and `1` rather than `0.75` and `0.25`. Any classifier-weighted sibling makes the
  static probability `None` for that whole source; a sole-outcome edge exports `1.0` and omits the
  Mermaid `P:` label.
- `Graph.to_networkx()` adds nodes and edges in a stable order (start node first, then ascending
  node id; edges sorted by endpoints, cost, and outcome position) instead of iterating
  identity-hashed sets, so a returned `DiGraph` iterates identically across processes. A `DiGraph`
  still holds one edge per node pair, so parallel edges collapse — but which one survives is now
  deterministic rather than decided by set order.
- `Graph.to_mermaid()` is likewise deterministic and start-node-first, so repeated exports are
  byte-for-byte identical and `max_nodes` truncation can no longer drop the start node.
- Sensitivity analysis selects its candidate parameters deterministically (sorted by node/edge id)
  rather than slicing an identity-hashed set, so the same graph analyzes the same parameters on
  every run.
- Edge-weight sensitivity skips edges that are their source node's only outcome. Transition weights
  are relative, so such a weight is mathematically inert; it no longer consumes a `max_params` slot,
  burns simulations, or gets recommended for improvement.
- `Node.get_nodes()`/`get_edges()` — and therefore `Graph.node_list()`, `edge_list()`,
  `to_networkx()`, `to_mermaid()`, and all sensitivity analysis — visit each node once instead of
  enumerating every distinct path. On a converging DAG this is the difference between exponential
  and linear: a 46-node layered graph went from roughly 8 seconds to well under a millisecond.
  `Node.to_tree()` is memoized per call for the same reason.
- Edge weights may be NumPy scalars, not just Python numbers.

### Fixed

- `Graph.plot()` no longer raises `AttributeError`. It called `nx.pygraphviz_layout`, removed in
  networkx 2.x; it now uses `nx.nx_agraph.graphviz_layout` and writes the figure to the requested
  filename with `savefig` rather than only calling `plt.show()`.
- Payoff sensitivity works for distribution nodes: each node type scales its own payoff parameters
  and restores them afterwards.
- Elasticity is reported as a non-negative magnitude for every parameter type. The three
  `analyze_sensitivity` branches had drifted apart — weights and costs divided by `baseline_ev`
  while payoffs divided by `abs(baseline_ev)` — so a graph with negative baseline EV produced one
  ranked table mixing negative and positive percentages. Positive-baseline graphs are numerically
  unchanged.
- The edge-weight sensitivity decrease arm applies the exact perturbed weight. It previously floored
  at a hardcoded `0.01`, so for any weight at or below roughly `0.011` the "-10%" run silently
  *raised* the weight instead — on a 0.5%/99.5% rare-event graph the decrease arm reported a higher
  expected value than both the baseline and the increase arm. This mainly affected graphs built by
  `from_adj_matrix`, whose weights are normalized probabilities.
- Sensitivity analysis restores perturbed edge weights and costs even when a simulation raises.
  Previously an exception mid-analysis left the graph permanently mutated.
- `identify_critical_parameters()` draws one Monte Carlo baseline and shares it across all three
  parameter types. Each type previously computed its own, so the elasticities in a single merged
  report were divided by three different estimates and the reported `baseline_ev` depended on which
  type happened to rank first.
- `Graph.to_mermaid()` styles the graph's actual terminal nodes. It previously hardcoded
  `class 0 terminal`, styling a possibly nonexistent node 0 and leaving real terminals unstyled.
- `Graph.from_adj_matrix()` ignores `NaN` cells when summing rows for normalization, so a row mixing
  finite entries and `NaN` keeps correct weights for its valid transitions.
- `Graph.from_dict()` raises a clear `AttributeError` naming the node and the unknown id for a
  dangling `after` reference, instead of an opaque `KeyError`. The `from_adj_matrix` docstring's
  edge direction is corrected to match the code: a nonzero `A[row, col]` means `row -> col`.
- Estimator categories are built in first-appearance order (`dict.fromkeys`) rather than by
  iterating a `set`. Previously, string labels were assigned a different index in every interpreter
  run because `str` hashing is randomized per process, so the same fitted model returned a different
  prediction — pointing at a different label — in each run.
- A simulated terminal node id outside the fitted category range (notably the synthetic root `-1`
  injected by `Graph.from_adj_matrix`) now raises a descriptive `ValueError` instead of silently
  indexing the last category.
- The `Graph` class docstring example is a valid, executable, seeded doctest and is run in CI. It
  previously used `>>>` on continuation lines, so it had never been executed and in fact raised a
  `SyntaxError`.
- The bundled example scripts run again. `examples/decision_vs_research.py` and `examples/print.py`
  called `Graph` methods that have never existed (`adjacency_matrix()`, `nodes`), and four scripts
  imported pandas, which no dependency group declared. CI now executes the examples and case
  studies on every push.

### Removed

- Python 3.8 support (see Changed).

## [0.1.0] - 2025-10-11

### Added

- **Automatic Sensitivity Analysis**: New methods on `Graph` class to automatically identify critical parameters
  - `analyze_sensitivity()`: Analyze sensitivity for specific parameter types
  - `identify_critical_parameters()`: Find most sensitive parameters across all types
  - `print_sensitivity_report()`: Generate formatted sensitivity report
- **Mermaid Diagram Export**: New `to_mermaid()` method on `Graph` class for visualization
- **Real-World Case Studies**: Four comprehensive case studies with documentation and implementations
  - Drug Development: Pharmaceutical R&D pipeline analysis
  - Startup Funding: VC funding journey modeling
  - Product Launch: New product introduction strategy
  - Litigation Strategy: Settlement vs. trial decisions
- **Modern Python Packaging**: Comprehensive `pyproject.toml` with all configuration
  - Build system configuration (setuptools)
  - Project metadata and dependencies
  - Optional dependency groups: dev, visualization, docs
  - Tool configurations: black, ruff, pytest, coverage, mypy
- **Development Tools Configuration**: Standardized code quality tools
  - Black for code formatting (line-length: 100)
  - Ruff for linting with curated rule set
  - MyPy for type checking
  - Pytest with coverage reporting
- **CI/CD**: GitHub Actions workflow for automated testing
  - Tests on Python 3.8-3.12
  - Automated linting, formatting checks, and type checking
  - Package build verification
- **Enhanced Documentation**:
  - Updated README with modern installation instructions
  - Quick start examples for new features
  - Case study overview and links
  - Development workflow documentation
  - CLAUDE.md for AI assistant guidance
- **Makefile**: Convenient targets for common development tasks
  - `make install`: Install package with dependencies
  - `make test`: Run test suite
  - `make examples`: Run example scripts
  - `make case-studies`: Run all case study analyses
  - `make clean`: Clean build artifacts

### Changed

- **Version**: Bumped from 0.0.1 to 0.1.0
- **Python Support**: Officially support Python 3.8-3.12
- **Dependencies**: Updated minimum versions
  - scikit-learn >= 1.0.0
  - numpy >= 1.20.0
- **Package Configuration**: Migrated from setup.py/setup.cfg/requirements.txt to pyproject.toml
- **Installation**: Recommend using `uv` for faster dependency resolution

### Fixed

- **Node Payoff Accumulation**: Fixed critical bug in `petersburg/nodes.py:81` where node payoffs weren't being accumulated during graph traversal
  - Previously: `return payoff, cost + edge.get_cost()`
  - Now: `return payoff + self.payoff, cost + edge.get_cost()`
- **Graph Structure Validation**: Improved handling of terminal nodes in graph construction

### Deprecated

- `setup.py`: Now a minimal shim pointing to pyproject.toml (will be removed in future release)

### Removed

- Legacy configuration files: `setup.cfg`, `requirements.txt`, `.travis.yml`, `MANIFEST.in`

## [0.0.1] - Initial Release

### Added

- Basic `Graph`, `Node`, and `Edge` classes
- Monte Carlo simulation for decision graphs
- `FrequencyEstimator` for learning graph structure from data
- `MixedModeEstimator` for hybrid explicit/learned models
- Basic examples including St. Petersburg Paradox
