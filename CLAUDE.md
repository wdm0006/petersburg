# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Petersburg is a Python framework for modeling and analyzing probabilistic decision processes as directed acyclic graphs (DAGs). It enables simulation, prediction, and analysis of complex decision networks with uncertain outcomes.

## Core Architecture

### Graph Structure

The framework models decision networks as DAGs with three key components:

1. **Nodes** ([nodes.py](petersburg/nodes.py)) - Represent decision points with:
   - `payoff`: Reward for reaching this node
   - `outcomes`: List of possible edges (next steps) with associated weights
   - Probabilistic selection of outcomes via `weighted_choice()`

2. **Edges** ([edges.py](petersburg/edges.py)) - Represent transitions between nodes with:
   - `cost`: Cost of traversing this edge
   - `from_node` and `to_node`: Connected nodes

3. **Graph** ([graph.py](petersburg/graph.py)) - Manages the entire network:
   - `start_node`: Single entry point (required)
   - `from_dict()`: Build graph from dictionary specification
   - `from_adj_matrix()`: Build graph from adjacency matrix
   - `get_outcome()`: Simulate single walk through the graph
   - `get_options()`: Compare expected values of initial choices

### Prediction Models

Two scikit-learn style estimators in [estimators.py](petersburg/estimators.py):

1. **FrequencyEstimator** - Uses observed transition frequencies to build a graph and predict outcomes via simulation
2. **MixedModeEstimator** - Combines frequency counts with logistic regression classifiers where sufficient training data exists (>= `_min_samples`)

Both estimators:
- Accept `X` (features) and `y` (multi-column array where each column represents a layer in the decision hierarchy)
- Build adjacency matrices from observed transitions
- Convert matrices to petersburg graphs
- Predict final outcomes through Monte Carlo simulation

### Edge Weights

Edge weights can be either:
- Static floats/ints representing fixed probabilities
- Trained classifiers that predict edge weights dynamically based on feature vectors

When a classifier is provided, `Node.get_weights()` calls `predict_proba()` to determine traversal probability.

## Development Commands

### Installation
```bash
pip install -r requirements.txt
python setup.py install
```

### Testing
```bash
# Run tests with coverage (as configured in Travis CI)
nosetests --with-coverage --cover-package=petersburg
```

### Running Examples
```bash
# Basic simulation examples
python examples/stpetersburg.py
python examples/necktie_paradox.py
python examples/two_envelope_problem.py

# Estimation examples
python examples/estimation/estimator_example.py
python examples/estimation/multimode_estimation.py
```

## Important Patterns

### Building Graphs from Dictionaries

Dictionary keys are node IDs. Each node specifies:
- `payoff`: Reward value
- `after`: List of edges, each with `node_id`, `cost`, and optional `weight`

The starting node is identified by an empty `after` list.

Example structure:
```python
{
    1: {'payoff': 0, 'after': []},  # Starting node
    2: {'payoff': 10, 'after': [{'node_id': 1, 'cost': 5, 'weight': 0.7}]}
}
```

### Adjacency Matrix Convention

Matrices follow the convention: `A[row, col]` means edge from `col` to `row`. The `from_adj_matrix()` method:
- Normalizes weights by row sums
- Automatically adds a root node (ID: -1)
- Supports optional `clf_matrix` for classifier-based edge weights

### Feature Vectors in Prediction

When calling `get_outcome(feature_vector)`, the vector propagates through the graph. At nodes with classifier-based edges, the feature vector is passed to `predict_proba()` to determine traversal probabilities dynamically.

### Automatic Sensitivity Analysis

The Graph class includes built-in sensitivity analysis to automatically identify which parameters have the most impact on outcomes:

```python
# Automatic sensitivity report
g.print_sensitivity_report(num_simulations=1000, perturbation=0.10, top_n=5)

# Programmatic access
results = g.identify_critical_parameters(num_simulations=1000, perturbation=0.10, top_n=5)

# Analyze specific parameter type
analysis = g.analyze_sensitivity(parameter_type='edge_weights', num_simulations=1000)
```

This automatically tests ±10% changes in all edge weights, costs, and node payoffs, then ranks them by impact on expected value. Key features:
- Tests all parameter types (edge weights, costs, payoffs)
- Ranks by absolute sensitivity ($ change in EV) and elasticity (% change in EV)
- Provides actionable recommendations on where to focus improvements
- No manual parameter testing required

See [examples/automatic_sensitivity_demo.py](examples/automatic_sensitivity_demo.py) for full demonstration.

## Package Structure

- [petersburg/](petersburg/) - Core package
  - [__init__.py](petersburg/__init__.py) - Exports `Graph`, `Node`, `Edge`, `FrequencyEstimator`, `MixedModeEstimator`
  - [graph.py](petersburg/graph.py) - Graph class and simulation logic
  - [nodes.py](petersburg/nodes.py) - Node class and weighted choice logic
  - [edges.py](petersburg/edges.py) - Edge class (simple wrapper)
  - [estimators.py](petersburg/estimators.py) - Scikit-learn style prediction models
- [examples/](examples/) - Demonstration scripts for various decision paradoxes
- [tests/](tests/) - Test suite (currently minimal)
