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

### Automatic Sensitivity Analysis

```python
# Identify the most critical parameters
g.print_sensitivity_report(num_simulations=1000, perturbation=0.1, top_n=5)
```

### Export to Mermaid Diagram

```python
# Generate a Mermaid diagram for visualization
mermaid_code = g.to_mermaid()
print(mermaid_code)
```
    
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