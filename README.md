petersburg
==========

![CI](https://github.com/wdm0006/petersburg/workflows/CI/badge.svg)
![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)

version number: 0.2.0
author: Will McGinnis

Overview
========

A framework for analyzing probabilistic decision processes as directed graphs with automatic sensitivity analysis.

Simulating and Predicting uncertain decisions
---------------------------------------------

Petersburg is a framework based on the decision theoretic concept of an individual agent approaching a network
of discrete decisions or probabilistic options. We model these networks as directed acyclic graphs that have a 
 few extra concepts:
 
 * Node Payoff: a potential reward for reaching some point
 * Edge Cost: a cost of taking a certain path
 * Edge Weight: a term related to the conditional likelihood of that edge being traversed (either a static number of classification model)

Using petersburg, you can:

 * **Build and simulate** complex decision graphs
    * The most likely outcomes
    * The worst/best case scenarios
    * Distributions of outcomes through Monte Carlo simulation
 * **Model uncertainty** with distribution-based node types
    * UniformNode: Outcomes uniformly distributed in a range
    * GaussianNode: Normally distributed outcomes
    * LogNormalNode: Log-normal distributions (always positive, heavy right tail)
    * PowerLawNode: Power law distributions for rare high-value events
 * **Automatic sensitivity analysis** - identify which parameters impact outcomes the most
 * **Visualize** decision graphs with Mermaid diagram export
 * **Predict** outcomes using machine learning
    * FrequencyEstimator: Learn graph structure from historical data
    * MixedModeEstimator: Combine explicit structure with learned probabilities
 * **Real-world applications** (see examples/case_studies/)
    * Drug development pipelines
    * Startup funding journeys
    * Product launches
    * Litigation strategy

## Emergent Complexity: Start Simple, Grow Sophisticated

One of petersburg's core design principles is **progressive refinement**: you can start with a simplified model that captures the basic structure of a complex system, then gradually add realism as your understanding deepens or your data improves.

### The Incremental Modeling Approach

**Stage 1: Fixed Payoffs** - Begin with deterministic outcomes to understand the structure
```python
# Simple model: Fixed $5B blockbuster exit
{'payoff': 5000, 'after': [...]}
```

**Stage 2: Distribution-Based Nodes** - Add realistic uncertainty to outcomes
```python
# More realistic: LogNormal distribution around $5B
{'type': 'lognormal', 'mu': 8.52, 'sigma': 0.4, 'after': [...]}
```

**Stage 3: Learned Transitions** - Use historical data to set edge probabilities
```python
# Edge weights learned from data
from petersburg import FrequencyEstimator
estimator = FrequencyEstimator()
estimator.fit(X_features, y_outcomes)
```

**Stage 4: Dynamic Edge Weights** - Transitions depend on context/features
```python
# Edge weights predicted by classifier based on features
{'node_id': 2, 'cost': 100, 'weight': trained_classifier}
```

### Emergent Power Laws from Simple Compositions

A remarkable property of petersburg models is that **simple binary transitions + continuous distributions = complex emergent behavior** that matches real-world power laws.

For example, in our startup funding case study:
- Binary decisions at each stage (continue/exit/fail)
- LogNormal distributions at exit nodes
- Simple stage-by-stage filtering

This produces a portfolio outcome distribution that exhibits:
- Heavy right tails (rare mega-exits)
- Realistic failure rates (80%+ fail)
- Power law returns matching empirical VC data

**You don't need to explicitly model power law complexity.** By composing:
1. Sequential filtering (survival rates at each stage)
2. Multiplicative processes (ownership dilution)
3. Log-normal outcomes (valuation distributions)

...the framework naturally generates the complex emergent patterns we observe in real venture portfolios, pharmaceutical pipelines, and product launches.

### When to Add Complexity

Use this decision framework:

| Model Stage | Use When | Example |
|------------|----------|---------|
| **Fixed payoffs** | Exploring structure, teaching concepts, rapid prototyping | "What if we add a pilot stage?" |
| **Distribution nodes** | Modeling real uncertainty, capturing tail risks | "Exit values range from $50M-$500M" |
| **Frequency estimation** | Have historical transition data, want empirical probabilities | "Learn success rates from past 100 drugs" |
| **Mixed-mode with classifiers** | Outcomes depend on features, sufficient training data | "Success rate varies by team experience, market size" |

**Start simple. Add complexity only when:**
- Sensitivity analysis shows a parameter matters
- You have data to support more sophisticated modeling
- Simpler models produce unrealistic outcomes

The goal is the **simplest model that captures the essential dynamics** of your decision problem.

Installation
============

### Using pip

```bash
pip install petersburg
```

Optional extras:

```bash
pip install petersburg[visualization]  # networkx + matplotlib (to_networkx, plot)
pip install petersburg[graphviz]       # pygraphviz for Graph.plot() — also needs a system Graphviz
pip install petersburg[docs]           # sphinx toolchain for building docs/
pip install petersburg[examples]       # pandas on top of visualization (example scripts)
pip install petersburg[all]            # everything above plus dev tooling
```

### From source (recommended for development)

```bash
git clone https://github.com/wdm0006/petersburg.git
cd petersburg
uv pip install -e ".[dev]"
```

Or using standard pip:

```bash
pip install -e ".[dev]"
```

Quick Start
===========

### Basic Graph Simulation

```python
from petersburg import Graph

# Build a simple decision graph
spec = {
    0: {'payoff': 0, 'after': []},  # Start node — the one with an empty 'after' list
    1: {'payoff': 100, 'after': [{'node_id': 0, 'cost': 10, 'weight': 0.3}]},  # Success branch (30%)
    2: {'payoff': -50, 'after': [{'node_id': 0, 'cost': 5, 'weight': 0.7}]},   # Failure branch (70%)
    3: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 0}, {'node_id': 2, 'cost': 0}]},  # Terminal node
}
g = Graph().from_dict(spec)

# Run simulation
outcomes = [g.get_outcome() for _ in range(10000)]
print(f"Expected value: ${sum(outcomes)/len(outcomes):.2f}")  # ≈ -$11.50 = 0.3·(+90) + 0.7·(−55)
```

How to read a graph dictionary:

 * Keys are node IDs. A walk starts at the node whose `after` list is empty and ends
   at a node that nothing follows.
 * A node's `after` list is its **predecessors**: `2: {'after': [{'node_id': 1, ...}]}`
   draws an edge `1 → 2`.
 * At each node the walk picks among the edges leaving it with probability proportional
   to each edge's `weight` (normalized to sum to 1). A weight on a node's single outgoing
   edge has no effect — that edge is always taken. The 30/70 split in the example above
   therefore lives on the edges *out of the start node* (declared in nodes 1 and 2's
   `after` lists).

### Reproducibility

Pass an integer seed to reproduce a simulation, or pass an existing NumPy
`Generator` to control its random stream:

```python
import numpy as np

seeded_graph = Graph(random_state=7).from_dict(spec)
generator_graph = Graph(random_state=np.random.default_rng(7)).from_dict(spec)
```

Graphs created with the same seed produce the same sequence of edge choices and
stochastic payoffs. Omitting `random_state` keeps the default unseeded behavior.

### Automatic Sensitivity Analysis

```python
# Identify the most critical parameters
g.print_sensitivity_report(num_simulations=1000, perturbation=0.1, top_n=5)

# At most max_params parameters per type are analyzed (default 10). The report says
# how many of the model's parameters were covered; pass max_params=None for all.
g.print_sensitivity_report(num_simulations=1000, perturbation=0.1, top_n=5, max_params=None)
```

### Visualizing the Graph

```python
# Mermaid text export (no extra dependencies)
print(g.to_mermaid())

# Rendered image — needs the [graphviz] extra (pygraphviz) plus a system Graphviz
# install, and networkx/matplotlib from the [visualization] extra
g.plot("graph.png")
```

### Distribution-Based Node Types

Model uncertainty with stochastic payoffs using different probability distributions:

```python
from petersburg import Graph

# Create a graph with different distribution types
g = Graph()
g.from_dict({
    0: {'type': 'fixed', 'payoff': 0, 'after': []},  # Terminal node
    1: {
        'type': 'uniform',        # Uniform distribution
        'min_payoff': 50,
        'max_payoff': 150,
        'after': [{'node_id': 0, 'cost': 10}]
    },
    2: {
        'type': 'gaussian',       # Normal distribution
        'mean': 100,
        'std': 20,
        'after': [{'node_id': 0, 'cost': 10}]
    },
    3: {
        'type': 'lognormal',      # Log-normal (always positive)
        'mu': 4.5,
        'sigma': 0.5,
        'after': [{'node_id': 0, 'cost': 10}]
    },
    4: {
        'type': 'powerlaw',       # Power law (heavy tails)
        'scale': 50,
        'alpha': 2,
        'after': [{'node_id': 0, 'cost': 10}]
    },
    5: {
        'type': 'fixed',
        'payoff': 0,
        'after': [
            {'node_id': 1, 'cost': 0, 'weight': 0.25},
            {'node_id': 2, 'cost': 0, 'weight': 0.25},
            {'node_id': 3, 'cost': 0, 'weight': 0.25},
            {'node_id': 4, 'cost': 0, 'weight': 0.25},
        ]
    }
})

# Each simulation samples from the distributions
outcomes = [g.get_outcome() for _ in range(1000)]
```

**Available Distribution Types:**

- **UniformNode**: Payoffs uniformly distributed between min and max
  - Use case: Equal probability across a range (e.g., uncertain market size)
  - Parameters: `min_payoff`, `max_payoff`

- **GaussianNode**: Normally distributed payoffs
  - Use case: Natural variation around a mean (e.g., product sales)
  - Parameters: `mean`, `std`

- **LogNormalNode**: Log-normally distributed payoffs (always positive)
  - Use case: Multiplicative processes, skewed positive outcomes (e.g., startup valuations)
  - Parameters: `mu`, `sigma` (parameters of underlying normal distribution)

- **PowerLawNode**: Power law (Pareto) distributed payoffs
  - Use case: Heavy-tailed distributions with rare extreme events (e.g., viral content, breakthrough innovations)
  - Parameters: `scale` (minimum value), `alpha` (tail heaviness)

See [examples/distribution_nodes_demo.py](examples/distribution_nodes_demo.py) for detailed examples.

Comparing Initial Options
=========================

`get_options()` simulates the walk once per outgoing edge of the start node, so you can
compare the initial choices by expected value:

```python
options = g.get_options(iters=10_000)
# {1: 90.0, 2: -55.0}  # expected profit, keyed by each option's destination node id
```

Pass `extended_stats=True` for mean/max/min/count per option, or `distribution=True` for
the full simulated outcome distribution of each option:

```python
options = g.get_options(iters=10_000, distribution=True, alpha=0.05)
# each option carries mean/max/min/count plus the distribution keys:
# {
#     "std": 63.2,                       # sample standard deviation
#     "percentiles": {"p5": ..., "p25": ..., "p50": ..., "p75": ..., "p95": ...},
#     "p_loss": 0.184,                   # fraction of simulated outcomes < 0
#     "var_alpha": -18.4,                # alpha-quantile of outcomes (value at risk, outcome space)
#     "cvar_alpha": -31.7,               # mean of outcomes <= var_alpha (expected shortfall)
# }
```

`alpha` (default 0.05) is the quantile level for `var_alpha`/`cvar_alpha`;
`cvar_alpha <= var_alpha` always holds, because expected shortfall averages the outcomes at
or below the quantile. `return_samples=True` (only valid together with `distribution=True`)
also returns each option's raw per-walk outcome samples as a numpy array. Calls without the
new kwargs keep the historical return shape.

Error Handling
==============

Every error petersburg raises on invalid use subclasses `PetersburgError`:

 * `ValidationError` (also a `ValueError`) — invalid input values: adjacency matrices,
   sensitivity arguments, estimator targets and feature matrices, node payoffs, transition
   weights, and argument combinations such as `get_options(return_samples=True)` without
   `distribution=True`.
 * `SpecValidationError` (a `ValidationError` that is also an `AttributeError`) — invalid
   graph specifications: unrecognized node types, references to unknown nodes, cycles,
   missing or multiple starting nodes, non-dict specifications.

```python
from petersburg import PetersburgError

try:
    Graph().from_dict({1: {'payoff': 1, 'after': [{'node_id': 999}]}})
except PetersburgError as err:
    print(f"invalid graph: {err}")
```

Existing `except ValueError` and `except AttributeError` handlers keep working unchanged;
new code should catch `PetersburgError` or `ValidationError`.

Serialization
=============

`Graph.to_dict()` produces exactly the dictionary `Graph.from_dict()` consumes, so graphs
with numeric weights round-trip losslessly:

```python
reloaded = Graph().from_dict(g.to_dict())
assert reloaded.to_dict() == g.to_dict()
```

Serialization covers nodes (including per-type distribution payoff parameters), edge costs,
and numeric transition weights. Edges weighted by estimator objects (trained classifiers)
are deliberately not serializable — `to_dict()` raises `ValidationError` for them; rebuild
such graphs from a numeric-weight specification and attach the classifiers afterwards.

Case Studies
============

See [examples/case_studies/](examples/case_studies/) for detailed real-world applications:

- **Drug Development**: Pharmaceutical R&D pipeline with Phase I-III trials
- **Startup Funding**: VC funding journey from pre-seed to exit
- **Product Launch**: New product introduction with market testing
- **Litigation Strategy**: Settlement vs. trial decision analysis

Each case study includes:
- Detailed markdown documentation with business context
- Python implementation with the petersburg framework
- Mermaid diagrams visualizing the decision flow
- Sensitivity analysis identifying critical parameters

Development
===========

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Format code
uv run black petersburg/ tests/ examples/

# Lint
uv run ruff check petersburg/ tests/ examples/

# Type check
uv run mypy petersburg/
```

### Running Examples

The scripts in `examples/` need pandas and matplotlib on top of the runtime
dependencies. Install the `examples` extra first (it is also part of `all`):

```bash
uv pip install -e ".[examples]"
```

Several examples call `plt.show()`, which blocks under an interactive matplotlib
backend, so set `MPLBACKEND=Agg` when running them unattended:

```bash
# Run a single example
MPLBACKEND=Agg python examples/necktie_paradox.py

# Run all examples
make examples

# Run all case studies
make case-studies
```

`examples/print.py` additionally needs a system Graphviz (plus pygraphviz) for
its `plot()` call, and `examples/stpetersburg_w_bankroll.py` simulates 10 million
games, so it takes several minutes.

Documentation
=============

The Sphinx sources live in [docs/](docs/). Build them locally with:

```bash
uv pip install -e ".[docs]"
sphinx-build -b html docs docs/_build
```

Contributing
============

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Example Static Graph
====================

Here is a simple example of simulating the St. Petersburg Paradox game, with some slight variations. In this case the 
entrance fee is $10, and the game only has a maximum of 10,000 flips and is played 10,000,000 times.

    from petersburg import Graph
    
    if __name__ == '__main__':
        g = Graph()
    
        # st petersburg paradox w/ $10 entrance fee and only 10000 possible flips
        entrance_fee = 10
        gd = {1: {'payoff': 0, 'after': []}, 2: {'payoff': 0, 'after': [{'node_id': 1, 'cost': entrance_fee}]}}
        nn = 3
        for idx in range(10000):
            node_id = 2 * (idx + 1)
            payoff = 2 ** (idx + 1)
            gd[nn] = {'payoff': payoff, 'after': [{'node_id': node_id, 'cost': 0, 'weight': 1}]}
            nn += 1
            gd[nn] = {'payoff': 0, 'after': [{'node_id': node_id, 'cost': 0, 'weight': 1}]}
            nn += 1
        g.from_dict(gd)
    
        outcomes = []
        for _ in range(10000000):
            outcomes.append(g.get_outcome())
    
        print('\n\nSimulated Output')
        print(sum(outcomes))

Via simulation, the outcome of this is a profit of: $197,592,288.  This will, of course, vary depending on the run, but
will approach infinity as the number of games goes to infinity, regardless of cost-to-play.

Prediction (scikit-learn estimators)
====================================

There are two prediction objects, both scikit-learn style estimators that learn a decision
graph from historical paths and predict outcomes through it:

 * `FrequencyEstimator` — transition frequencies observed in the data
 * `MixedModeEstimator` — frequencies plus per-transition classifiers (logistic regression
   by default) where enough training data exists

Both follow the scikit-learn contract:

```python
import numpy as np
from petersburg import MixedModeEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

est = MixedModeEstimator(min_samples=100, clf=LogisticRegression())
est.fit(X, y)                  # y: one column per decision layer

est.classes_                   # fitted terminal labels
est.n_features_in_             # number of feature columns seen at fit

proba = est.predict_proba(X)   # shape (n_samples, n_classes); rows sum to 1
pred = est.predict(X)          # shape (n_samples,) — fitted terminal labels
est.score(X, y)                # terminal-layer accuracy

est.partial_fit(X_more, y_more)  # incremental update, both estimators

# hyperparameters live in __init__, so they are tunable:
search = GridSearchCV(
    est, {"min_samples": [10, 50, 100], "clf__C": [0.1, 1.0, 10.0]}, cv=3
)
```

Notes:

 * `predict()` returns a one-dimensional label array `(n_samples,)` (it returned
   `(n_samples, 1)` before 0.2.0).
 * Columns of `predict_proba` follow `classes_`; with a seeded `random_state`, the modal
   column always matches `predict()`.
 * `FrequencyEstimator.fit(X=None, y)` fits without features; `n_features_in_` is absent
   in that case.
 * Hyperparameters (`min_samples`, `clf`, `clf_args`) are stored verbatim per the
   scikit-learn parameter contract, so `get_params()`, `set_params()`, and `clone()` see
   exactly what was passed.

Full working examples in [examples/estimation/](examples/estimation/).