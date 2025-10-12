"""
Demonstration of Distribution-Based Node Types in Petersburg

This example shows how to use the different node types with stochastic payoffs:
- UniformNode: Payoffs drawn from uniform distribution
- GaussianNode: Payoffs drawn from normal distribution
- LogNormalNode: Payoffs drawn from log-normal distribution (always positive)
- PowerLawNode: Payoffs drawn from power law distribution (heavy tails)

These node types are useful for modeling real-world scenarios where outcomes
are uncertain and follow known statistical distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from petersburg import Graph

# Example 1: Investment with Uncertain Returns
# =============================================
# Compare a safe investment with fixed returns vs. a risky investment
# with returns that follow different distributions

print("=" * 80)
print("Example 1: Investment Decision with Uncertain Returns")
print("=" * 80)

# Safe investment: Fixed 5% return on $10,000
# Risky stock: Returns follow a Gaussian distribution (mean 8%, std 15%)
# Startup investment: Returns follow a log-normal distribution (high upside, positive only)
# Venture capital: Returns follow a power law (rare big wins)

investment_graph = Graph()
investment_graph.from_dict(
    {
        0: {"type": "fixed", "payoff": 0, "after": []},  # Terminal node
        1: {
            "type": "fixed",
            "payoff": 500,
            "after": [{"node_id": 0, "cost": 10000}],
        },  # Safe: $500 return
        2: {
            "type": "gaussian",
            "mean": 800,
            "std": 1500,
            "after": [{"node_id": 0, "cost": 10000}],
        },  # Risky stock
        3: {
            "type": "lognormal",
            "mu": 6.21,
            "sigma": 1.5,
            "after": [{"node_id": 0, "cost": 10000}],
        },  # Startup
        4: {
            "type": "powerlaw",
            "scale": 100,
            "alpha": 1.5,
            "after": [{"node_id": 0, "cost": 10000}],
        },  # VC
        5: {
            "type": "fixed",
            "payoff": 0,
            "after": [
                {"node_id": 1, "cost": 0, "weight": 0.25},
                {"node_id": 2, "cost": 0, "weight": 0.25},
                {"node_id": 3, "cost": 0, "weight": 0.25},
                {"node_id": 4, "cost": 0, "weight": 0.25},
            ],
        },
    }
)

# Run simulations for each investment option
num_simulations = 10000
safe_outcomes = []
stock_outcomes = []
startup_outcomes = []
vc_outcomes = []

for _ in range(num_simulations):
    outcome = investment_graph.get_outcome()
    # Which path was taken is random, so we collect all outcomes
    if -10000 < outcome < -9000:
        safe_outcomes.append(outcome)
    elif -12000 < outcome < -5000:
        stock_outcomes.append(outcome)

# Let's simulate each path directly for clearer analysis
safe_graph = Graph()
safe_graph.from_dict(
    {
        0: {"type": "fixed", "payoff": 0, "after": []},
        1: {"type": "fixed", "payoff": 500, "after": [{"node_id": 0, "cost": 10000}]},
    }
)

stock_graph = Graph()
stock_graph.from_dict(
    {
        0: {"type": "fixed", "payoff": 0, "after": []},
        1: {
            "type": "gaussian",
            "mean": 800,
            "std": 1500,
            "after": [{"node_id": 0, "cost": 10000}],
        },
    }
)

startup_graph = Graph()
startup_graph.from_dict(
    {
        0: {"type": "fixed", "payoff": 0, "after": []},
        1: {
            "type": "lognormal",
            "mu": 6.21,
            "sigma": 1.5,
            "after": [{"node_id": 0, "cost": 10000}],
        },
    }
)

vc_graph = Graph()
vc_graph.from_dict(
    {
        0: {"type": "fixed", "payoff": 0, "after": []},
        1: {
            "type": "powerlaw",
            "scale": 100,
            "alpha": 1.5,
            "after": [{"node_id": 0, "cost": 10000}],
        },
    }
)

safe_outcomes = [safe_graph.get_outcome() for _ in range(num_simulations)]
stock_outcomes = [stock_graph.get_outcome() for _ in range(num_simulations)]
startup_outcomes = [startup_graph.get_outcome() for _ in range(num_simulations)]
vc_outcomes = [vc_graph.get_outcome() for _ in range(num_simulations)]

print("\nInvestment Analysis ($10,000 initial investment):")
print("-" * 80)
print(
    f"{'Investment Type':<20} {'Mean ROI':<15} {'Std Dev':<15} {'90th %ile':<15} {'10th %ile':<15}"
)
print("-" * 80)

for name, outcomes in [
    ("Safe Bond", safe_outcomes),
    ("Risky Stock", stock_outcomes),
    ("Startup Equity", startup_outcomes),
    ("VC Fund", vc_outcomes),
]:
    mean_roi = np.mean(outcomes)
    std_roi = np.std(outcomes)
    p90 = np.percentile(outcomes, 90)
    p10 = np.percentile(outcomes, 10)
    print(f"{name:<20} ${mean_roi:>7.2f}        ${std_roi:>7.2f}        ${p90:>7.2f}        ${p10:>7.2f}")

print("\nKey Insights:")
print("- Safe Bond: Guaranteed 5% return ($500)")
print("- Risky Stock: Higher mean (8%) but with significant volatility")
print("- Startup Equity: Log-normal distribution captures 'unicorn' potential")
print("- VC Fund: Power law distribution with rare but massive wins")

# Example 2: Project Outcomes with Different Uncertainty Types
# =============================================================
print("\n" + "=" * 80)
print("Example 2: R&D Project Selection")
print("=" * 80)

# Three R&D projects with different risk profiles:
# Project A: Incremental improvement (uniform uncertainty in narrow range)
# Project B: Applied research (normal distribution around expected value)
# Project C: Basic research (power law - small chance of breakthrough)

project_graph = Graph()
project_graph.from_dict(
    {
        0: {"type": "fixed", "payoff": 0, "after": []},
        1: {
            "type": "uniform",
            "min_payoff": 80000,
            "max_payoff": 120000,
            "after": [{"node_id": 0, "cost": 50000}],
        },  # Project A
        2: {
            "type": "gaussian",
            "mean": 100000,
            "std": 30000,
            "after": [{"node_id": 0, "cost": 50000}],
        },  # Project B
        3: {
            "type": "powerlaw",
            "scale": 10000,
            "alpha": 1.8,
            "after": [{"node_id": 0, "cost": 50000}],
        },  # Project C
        4: {
            "type": "fixed",
            "payoff": 0,
            "after": [
                {"node_id": 1, "cost": 0, "weight": 1.0 / 3},
                {"node_id": 2, "cost": 0, "weight": 1.0 / 3},
                {"node_id": 3, "cost": 0, "weight": 1.0 / 3},
            ],
        },
    }
)

# Simulate each project independently
proj_a_graph = Graph()
proj_a_graph.from_dict(
    {
        0: {"type": "fixed", "payoff": 0, "after": []},
        1: {
            "type": "uniform",
            "min_payoff": 80000,
            "max_payoff": 120000,
            "after": [{"node_id": 0, "cost": 50000}],
        },
    }
)

proj_b_graph = Graph()
proj_b_graph.from_dict(
    {
        0: {"type": "fixed", "payoff": 0, "after": []},
        1: {
            "type": "gaussian",
            "mean": 100000,
            "std": 30000,
            "after": [{"node_id": 0, "cost": 50000}],
        },
    }
)

proj_c_graph = Graph()
proj_c_graph.from_dict(
    {
        0: {"type": "fixed", "payoff": 0, "after": []},
        1: {
            "type": "powerlaw",
            "scale": 10000,
            "alpha": 1.8,
            "after": [{"node_id": 0, "cost": 50000}],
        },
    }
)

proj_a_outcomes = [proj_a_graph.get_outcome() for _ in range(num_simulations)]
proj_b_outcomes = [proj_b_graph.get_outcome() for _ in range(num_simulations)]
proj_c_outcomes = [proj_c_graph.get_outcome() for _ in range(num_simulations)]

print("\nR&D Project Analysis ($50,000 investment each):")
print("-" * 80)
print(
    f"{'Project':<20} {'Mean NPV':<15} {'Std Dev':<15} {'Max Outcome':<15} {'Min Outcome':<15}"
)
print("-" * 80)

for name, outcomes in [
    ("Project A (Incr.)", proj_a_outcomes),
    ("Project B (Applied)", proj_b_outcomes),
    ("Project C (Basic)", proj_c_outcomes),
]:
    mean_npv = np.mean(outcomes)
    std_npv = np.std(outcomes)
    max_outcome = np.max(outcomes)
    min_outcome = np.min(outcomes)
    print(
        f"{name:<20} ${mean_npv:>8.2f}       ${std_npv:>8.2f}       ${max_outcome:>8.2f}       ${min_outcome:>8.2f}"
    )

print("\nKey Insights:")
print("- Project A: Predictable returns ($30k-$70k range)")
print("- Project B: Similar expected value with normal uncertainty")
print("- Project C: Power law creates potential for breakthrough (but higher risk)")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Investment Returns Distribution
axes[0, 0].hist(safe_outcomes, bins=50, alpha=0.6, label="Safe Bond", density=True)
axes[0, 0].hist(stock_outcomes, bins=50, alpha=0.6, label="Risky Stock", density=True)
axes[0, 0].set_xlabel("Net Return ($)")
axes[0, 0].set_ylabel("Probability Density")
axes[0, 0].set_title("Investment Returns: Safe vs. Risky")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: High-Risk Investment Distributions
axes[0, 1].hist(startup_outcomes, bins=100, alpha=0.6, label="Startup", density=True, range=(-15000, 50000))
axes[0, 1].hist(vc_outcomes, bins=100, alpha=0.6, label="VC Fund", density=True, range=(-15000, 50000))
axes[0, 1].set_xlabel("Net Return ($)")
axes[0, 1].set_ylabel("Probability Density")
axes[0, 1].set_title("High-Risk Investments: Heavy Tails")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: R&D Project Distributions
axes[1, 0].hist(proj_a_outcomes, bins=50, alpha=0.6, label="Incremental", density=True)
axes[1, 0].hist(proj_b_outcomes, bins=50, alpha=0.6, label="Applied", density=True)
axes[1, 0].set_xlabel("Net Present Value ($)")
axes[1, 0].set_ylabel("Probability Density")
axes[1, 0].set_title("R&D Projects: Incremental vs. Applied")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Basic Research with Power Law
axes[1, 1].hist(
    proj_c_outcomes,
    bins=100,
    alpha=0.7,
    label="Basic Research",
    density=True,
    color="purple",
    range=(-50000, 200000),
)
axes[1, 1].set_xlabel("Net Present Value ($)")
axes[1, 1].set_ylabel("Probability Density")
axes[1, 1].set_title("Basic Research: Power Law Distribution")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("distribution_nodes_demo.png", dpi=150, bbox_inches="tight")
print("\n" + "=" * 80)
print("Visualization saved as 'distribution_nodes_demo.png'")
print("=" * 80)
