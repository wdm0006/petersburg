petersburg
==========

![CI](https://github.com/wdm0006/petersburg/workflows/CI/badge.svg)
![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)

version number: 0.1.0
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
g = Graph()
g.from_dict({
    0: {'payoff': 0, 'after': []},  # Terminal node
    1: {'payoff': 100, 'after': [{'node_id': 0, 'cost': 10, 'weight': 1.0}]},  # Success
    2: {'payoff': -50, 'after': [{'node_id': 0, 'cost': 5, 'weight': 1.0}]},   # Failure
    3: {'payoff': 0, 'after': [
        {'node_id': 1, 'cost': 0, 'weight': 0.3},  # 30% success
        {'node_id': 2, 'cost': 0, 'weight': 0.7},  # 70% failure
    ]},  # Starting node
})

# Run simulation
outcomes = [g.get_outcome() for _ in range(10000)]
print(f"Expected value: ${sum(outcomes)/len(outcomes):.2f}")
```

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

### Export to Mermaid Diagram

```python
# Generate a Mermaid diagram for visualization
mermaid_code = g.to_mermaid()
print(mermaid_code)
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

```bash
# Run all examples
make examples

# Run all case studies
make case-studies
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

Example Prediction
==================

There are two prediction objects, both of which are scikit-learn style classes. 

 * MixedModeEstimator
 * FrequencyEstimator
 
Both have full working examples in the examples/estimation/* directory.