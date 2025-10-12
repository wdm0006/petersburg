"""
Simple example demonstrating distribution-based node types.

This shows the basic usage of UniformNode, GaussianNode, LogNormalNode, and PowerLawNode.
"""

from petersburg import Graph

# Example: A simple decision tree with stochastic outcomes
print("Simple Distribution Node Example")
print("=" * 50)

# Create a graph with different distribution types
g = Graph()
g.from_dict(
    {
        0: {"type": "fixed", "payoff": 0, "after": []},  # Terminal node
        1: {
            "type": "uniform",
            "min_payoff": 50,
            "max_payoff": 150,
            "after": [{"node_id": 0, "cost": 0}],
        },
        2: {
            "type": "gaussian",
            "mean": 100,
            "std": 20,
            "after": [{"node_id": 0, "cost": 0}],
        },
        3: {
            "type": "lognormal",
            "mu": 4.5,
            "sigma": 0.5,
            "after": [{"node_id": 0, "cost": 0}],
        },
        4: {
            "type": "powerlaw",
            "scale": 50,
            "alpha": 2,
            "after": [{"node_id": 0, "cost": 0}],
        },
        5: {
            "type": "fixed",
            "payoff": 0,
            "after": [
                {"node_id": 1, "cost": 10, "weight": 0.25},
                {"node_id": 2, "cost": 10, "weight": 0.25},
                {"node_id": 3, "cost": 10, "weight": 0.25},
                {"node_id": 4, "cost": 10, "weight": 0.25},
            ],
        },
    }
)

# Run multiple simulations
num_sims = 1000
outcomes = [g.get_outcome() for _ in range(num_sims)]

print(f"\nResults from {num_sims} simulations:")
print(f"Mean outcome: ${sum(outcomes) / len(outcomes):.2f}")
print(f"Min outcome: ${min(outcomes):.2f}")
print(f"Max outcome: ${max(outcomes):.2f}")

# You can also use the node types directly without from_dict:
print("\n" + "=" * 50)
print("Direct Node Usage Example")
print("=" * 50)

from petersburg import UniformNode, GaussianNode, LogNormalNode, PowerLawNode, Node, Graph

# Create nodes directly
g2 = Graph()
terminal = Node(node_id=0, payoff=0)
uniform_node = UniformNode(node_id=1, min_payoff=100, max_payoff=200)
gaussian_node = GaussianNode(node_id=2, mean=150, std=25)

# Build graph manually
uniform_node.add_outcome(terminal, cost=0, weight=1)
gaussian_node.add_outcome(terminal, cost=0, weight=1)

start = Node(node_id=3, payoff=0)
start.add_outcome(uniform_node, cost=10, weight=0.5)
start.add_outcome(gaussian_node, cost=10, weight=0.5)

g2.start_node = start

# Run simulations
outcomes2 = [g2.get_outcome() for _ in range(num_sims)]

print(f"\nResults from {num_sims} simulations:")
print(f"Mean outcome: ${sum(outcomes2) / len(outcomes2):.2f}")
print(f"Expected: ~$130 (0.5 * (150-10) + 0.5 * (150-10))")
