"""
Cost-Wise Gradient Analysis: Finding High-Impact Optimization Opportunities

This example demonstrates a powerful technique for identifying which parts of
a complex decision graph offer the greatest opportunities for improvement.

THE PROBLEM:
You have a complex system with many components (nodes), each with:
- Different costs
- Different probabilities of being used
- Different potential for optimization

Which components should you optimize FIRST to maximize impact?

THE APPROACH: GRADIENT ANALYSIS
For each edge in the graph, calculate its "sensitivity gradient":
  Gradient = (Change in expected outcome) / (Change in probability of using that edge)

This tells you: "If I make this edge 10% more likely, how much does my overall
outcome improve?"

High gradient edges are HIGH LEVERAGE opportunities:
- Small improvements → large overall impact
- Focus optimization efforts here first

Low gradient edges are LOW LEVERAGE:
- Even big improvements have small overall impact
- Optimize these later (or never)

THE GRAPH STRUCTURE:
A randomly generated decision tree with:
- Node 1: Terminal
- Node 2: One endpoint (tracked for frequency)
- Node 3: Another endpoint
- Nodes 4-6: Mid-level decision points
- Nodes 7-12: Starting points (edges we analyze)

Each edge has:
- Random cost (1-10)
- Random weight (1-10)
- Different probability of being traversed

THE ANALYSIS PROCESS:
For each edge (nodes 7-12):
1. Generate 25 variants with different weights (perturb the probability)
2. Simulate 10,000 outcomes for each variant
3. Track the frequency each node is reached
4. Calculate: gradient = Δ(expected_outcome) / Δ(frequency) × 100
5. Find the edge with highest gradient = best optimization target

DUAL-AXIS VISUALIZATION:
- Blue dots (left axis): Gradient = sensitivity to optimization
- Red dots (right axis): Current frequency = how often this edge is used

STRATEGIC INSIGHTS:
1. High gradient + High frequency = CRITICAL PATH (optimize first!)
2. High gradient + Low frequency = High potential, but less impact
3. Low gradient + High frequency = Already optimized or limited potential
4. Low gradient + Low frequency = Ignore these for now

KEY INSIGHTS:
1. Not all optimizations are equal - focus on high-leverage changes
2. Frequently used edges aren't always the best optimization targets
3. Rare but high-impact edges can be worth optimizing
4. Gradient analysis helps prioritize engineering effort
5. This technique applies to: code optimization, process improvement,
   cost reduction, risk mitigation, etc.

FRAMEWORK USAGE:
This example demonstrates:
- Parametric sensitivity analysis
- Graph perturbation and comparison
- Tracking path frequencies with get_outcome_node()
- Dual-axis visualization for multi-metric analysis
- How to identify optimization opportunities in complex systems
- Random graph generation for testing
"""

import random

import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np

matplotlib.style.use("ggplot")
import pandas as pd  # noqa: E402

from petersburg import Graph  # noqa: E402

__author__ = "willmcginnis"


def build_graph(node_id=None):
    """
    Build a decision graph with random costs and weights.
    If node_id is specified, return multiple versions with that node's weight perturbed.

    Parameters:
    - node_id: If None, return single graph with random parameters.
               If specified (7-12), return 25 graphs with that node's weight varied.

    Returns:
    - List of (weight, Graph) tuples
    """
    # Fixed weights for core structure (nodes 1-6)
    # These remain constant to isolate the effect of changing outer nodes
    w1, w2, w3, w4, w5, w6 = 10, 10, 10, 10, 10, 10

    # Random weights for outer nodes (7-12)
    # These represent different activities/paths with different likelihoods
    w7 = random.randint(1, 10)
    w8 = random.randint(1, 10)
    w9 = random.randint(1, 10)
    w10 = random.randint(1, 10)
    w11 = random.randint(1, 10)

    # Fixed costs for core structure (zero cost transitions)
    c1, c2, c3, c4, c5, c6 = 0, 0, 0, 0, 0, 0

    # Random costs for outer nodes (1-10)
    # These represent different cost profiles for different activities
    c7 = random.randint(1, 10)
    c8 = random.randint(1, 10)
    c9 = random.randint(1, 10)
    c10 = random.randint(1, 10)
    c11 = random.randint(1, 10)

    # Build the graph structure
    # This creates a tree with multiple paths converging to endpoints
    template = {
        1: {"payoff": 0, "after": []},  # Terminal node
        2: {"payoff": 0, "after": [{"node_id": 1, "cost": c1, "weight": w1}]},  # Endpoint A
        3: {"payoff": 0, "after": [{"node_id": 1, "cost": c2, "weight": w2}]},  # Endpoint B
        4: {"payoff": 0, "after": [{"node_id": 3, "cost": c3, "weight": w3}]},  # Mid-level
        5: {"payoff": 0, "after": [{"node_id": 3, "cost": c4, "weight": w4}]},  # Mid-level
        6: {"payoff": 0, "after": [{"node_id": 3, "cost": c5, "weight": w5}]},  # Mid-level
        7: {"payoff": 0, "after": [{"node_id": 4, "cost": c6, "weight": w6}]},  # Starting node
        8: {"payoff": 0, "after": [{"node_id": 4, "cost": c7, "weight": w7}]},  # Starting node
        9: {"payoff": 0, "after": [{"node_id": 5, "cost": c8, "weight": w8}]},  # Starting node
        10: {"payoff": 0, "after": [{"node_id": 5, "cost": c9, "weight": w9}]},  # Starting node
        11: {"payoff": 0, "after": [{"node_id": 6, "cost": c10, "weight": w10}]},  # Starting node
        12: {"payoff": 0, "after": [{"node_id": 6, "cost": c11, "weight": w11}]},  # Starting node
    }

    # If no perturbation requested, return single graph
    if node_id is None:
        return [(None, Graph().from_dict(template))]

    # Generate multiple graphs with perturbed weight for specified node
    out = []
    default = template[node_id]["after"][0]["weight"]

    # Test 25 different weight values for this node
    for x in np.linspace(0.1, default + 1, 25):
        template[node_id]["after"][0]["weight"] = x
        g = Graph()
        g.from_dict(template)
        out.append((x, g))

    return out


if __name__ == "__main__":
    print("=" * 70)
    print("COST-WISE GRADIENT ANALYSIS")
    print("=" * 70)
    print("Identifying high-leverage optimization opportunities in a decision graph")
    print()
    print("For each edge (node 7-12), we calculate:")
    print("  Gradient = (Change in expected outcome) / (Change in frequency)")
    print()
    print("High gradient = small improvements have large impact (optimize first!)")
    print("Low gradient = even big improvements have small impact (optimize later)")
    print()
    print("Analyzing 6 different edges with 25 weight perturbations each...")
    print()

    # Analyze sensitivity gradient for each outer node (7-12)
    data = []
    for nid in range(7, 13):
        print(f"Analyzing node {nid}...")

        # Get multiple graphs with this node's weight perturbed
        graphs = build_graph(nid)

        means = []
        mins = []
        maxes = []
        likelyhoods = []

        for _weight, graph in graphs:
            # Simulate outcomes to measure expected value
            outcomes = []
            for _ in range(10000):
                outcomes.append(graph.get_outcome())

            # Track which nodes are reached (path frequency)
            iters = 10000
            output_nodes = {2: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
            for _ in range(iters):
                outnid = graph.get_outcome_node()
                output_nodes[outnid] += 1

            # Collect statistics for this weight value
            likelyhoods.append(output_nodes[nid])
            means.append(float(sum(outcomes)) / len(outcomes))
            mins.append(min(outcomes))
            maxes.append(max(outcomes))

        # Calculate gradient: how much outcome changes per unit change in frequency
        # This is the KEY METRIC for prioritizing optimization efforts
        max_change = max(means) - min(means)  # Max impact on expected outcome
        likelihood_range = max(likelyhoods) - min(likelyhoods)  # Range of frequencies tested
        gradient = float(max_change) / likelihood_range * 100 if likelihood_range > 0 else 0

        data.append(
            [
                nid,
                max_change,
                min(likelyhoods),
                max(likelyhoods),
                gradient,
            ]
        )

    # Create DataFrame with gradient analysis results
    df = pd.DataFrame(
        data, columns=["node_id", "max_change", "min_likelyhood", "max_likelyhood", "gradient"]
    )

    print()
    print("Gradient analysis complete!")
    print()

    # Calculate baseline frequencies (how often each node is reached currently)
    print("Calculating baseline frequencies...")
    iters = 10000
    graph = build_graph(None)[0][1]
    output_nodes = {2: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
    for _ in range(iters):
        outnid = graph.get_outcome_node()
        output_nodes[outnid] += 1

    # Build frequency data for visualization
    data = []
    for node_id in output_nodes.keys():
        if node_id != 2:  # Skip the endpoint node
            data.append([node_id, float(output_nodes[node_id]) / iters])
    df2 = pd.DataFrame(data, columns=["node_id", "frequency"])

    print()
    print("=" * 70)
    print("RESULTS: GRADIENT AND FREQUENCY ANALYSIS")
    print("=" * 70)
    print()
    print("Node | Gradient | Frequency | Priority")
    print("-----|----------|-----------|----------")
    for _, row in df.iterrows():
        nid = int(row["node_id"])
        grad = row["gradient"]
        freq = (
            df2[df2["node_id"] == nid]["frequency"].values[0] if nid in df2["node_id"].values else 0
        )

        # Determine priority based on gradient and frequency
        if grad > df["gradient"].median() and freq > df2["frequency"].median():
            priority = "HIGH (critical path)"
        elif grad > df["gradient"].median():
            priority = "MEDIUM (high leverage)"
        else:
            priority = "LOW (limited impact)"

        print(f"{nid:4d} | {grad:8.2f} | {freq:9.2%} | {priority}")

    print()
    print("=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    print("Creating dual-axis plot...")
    print("  Blue dots (left axis): Gradient = sensitivity to optimization")
    print("  Red dots (right axis): Frequency = how often each edge is used")
    print()

    # Create dual-axis visualization
    ax = df.plot(kind="scatter", x="node_id", y="gradient", s=50, label="Gradient")
    ax2 = ax.twinx()
    df2.plot(
        ax=ax2, kind="scatter", x="node_id", y="frequency", s=50, color="red", label="Frequency"
    )

    # Style the plot
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    plt.title("Gradient Analysis: Finding High-Impact Optimization Targets")
    plt.xlabel("Edge/Node ID")
    ax.set_ylabel("Gradient (Expected Change per Unit Probability)", color="blue")
    ax2.set_ylabel("Current Frequency (Probability of Use)", color="red")
    ax.grid(True, color="w", linestyle="-", linewidth=1)
    ax2.grid(False)
    ax.set_ylim([df["gradient"].min() * 0.8, df["gradient"].max() * 1.2])
    plt.gca().patch.set_facecolor("0.8")
    plt.tight_layout()

    print("=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("Look for edges with:")
    print("1. HIGH gradient (blue) + HIGH frequency (red) = CRITICAL PATH")
    print("   → These are your best optimization targets!")
    print()
    print("2. HIGH gradient + LOW frequency = High leverage but less impact")
    print("   → Consider if you can increase usage of this path")
    print()
    print("3. LOW gradient + HIGH frequency = Common but low-impact")
    print("   → Already optimized or inherently low-value")
    print()
    print("Real-world applications:")
    print("- Code optimization (which functions to optimize first)")
    print("- Process improvement (which bottlenecks to address)")
    print("- Cost reduction (which expenses to tackle)")
    print("- Risk mitigation (which risks to address first)")
    print("=" * 70)

    plt.show()
