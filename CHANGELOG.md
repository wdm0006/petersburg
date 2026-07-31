# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BREAKING**: `FrequencyEstimator.predict()` and `MixedModeEstimator.predict()` now return the
  fitted label of the simulated terminal category rather than that category's internal index into
  `_categories`. The returned array is object-dtype, so non-numeric (e.g. string) labels round-trip
  instead of being coerced into a float array. Callers that consumed the index must now consume the
  label; this matches the scikit-learn estimator contract that `predict` returns fitted labels, and
  makes the inherited `score()` comparison against `y` meaningful.

### Fixed

- Estimator categories are now built in first-appearance order (`dict.fromkeys`) rather than by
  iterating a `set`. Previously, string labels were assigned a different index in every interpreter
  run because `str` hashing is randomized per process, so the same fitted model returned a different
  prediction — pointing at a different label — in each run.
- A simulated terminal node id outside the fitted category range (notably the synthetic root `-1`
  injected by `Graph.from_adj_matrix`) now raises a descriptive `ValueError` instead of silently
  indexing the last category.

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
