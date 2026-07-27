"""
Demonstration of Automatic Sensitivity Analysis

This example shows how petersburg can automatically identify the most
sensitive parameters in a graph without manually testing each one.
"""

from petersburg import Graph


def create_simple_project_graph():
    """
    Create a simple graph representing a project with multiple stages.

    The project has:
    - Initial investment: $100K
    - Stage 1: 80% success, costs $50K
    - Stage 2: 60% success, costs $75K
    - Success outcome: $500K payoff
    - Failure: $0 payoff
    """
    g = Graph()

    graph_dict = {
        0: {"payoff": 0, "after": []},  # Terminal
        # Failure nodes
        1: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed Stage 1
        2: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed Stage 2
        # Success node
        3: {"payoff": 500, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Project success!
        # Stage 2
        4: {
            "payoff": 0,
            "after": [
                {"node_id": 3, "cost": 75, "weight": 0.60},  # 60% success
                {"node_id": 2, "cost": 25, "weight": 0.40},  # 40% failure (cheaper exit)
            ],
        },
        # Stage 1
        5: {
            "payoff": 0,
            "after": [
                {"node_id": 4, "cost": 50, "weight": 0.80},  # 80% success
                {"node_id": 1, "cost": 20, "weight": 0.20},  # 20% failure (cheaper exit)
            ],
        },
        # Start
        6: {
            "payoff": 0,
            "after": [
                {"node_id": 5, "cost": 100, "weight": 1.0},  # Initial investment
            ],
        },
    }

    g.from_dict(graph_dict)
    return g


if __name__ == "__main__":
    print("=" * 80)
    print("AUTOMATIC SENSITIVITY ANALYSIS DEMO")
    print("=" * 80)
    print()
    print("This demo shows how petersburg automatically identifies the most")
    print("impactful parameters in a decision graph.")
    print()
    print("Scenario: A two-stage project with uncertain outcomes at each stage.")
    print()

    g = create_simple_project_graph()

    # Method 1: Print a formatted report
    print("METHOD 1: Print Sensitivity Report")
    print("-" * 80)
    print()
    g.print_sensitivity_report(num_simulations=1000, perturbation=0.10, top_n=5)

    # Method 2: Get programmatic access to results
    print()
    print("METHOD 2: Programmatic Access to Results")
    print("-" * 80)
    print()

    results = g.identify_critical_parameters(num_simulations=1000, perturbation=0.10, top_n=3)

    print(f"Baseline EV: ${results['baseline_ev']:.2f}")
    print(
        f"Parameters analyzed: {results['total_parameters_analyzed']} "
        f"of {results['total_candidate_parameters']}"
    )
    print()
    print("Top 3 parameters:")
    for i, param in enumerate(results["top_parameters"], 1):
        print(f"  {i}. {param['parameter']}")
        print(f"     Sensitivity: ${param['sensitivity']:.2f}")
        print(f"     Elasticity: {param['elasticity']*100:.1f}%")
        print()

    # Method 3: Analyze specific parameter type
    print()
    print("METHOD 3: Analyze Specific Parameter Type (Edge Weights Only)")
    print("-" * 80)
    print()

    weight_analysis = g.analyze_sensitivity(
        parameter_type="edge_weights", num_simulations=1000, perturbation=0.10, max_params=10
    )

    print(
        f"Analyzed {weight_analysis['parameters_analyzed']} "
        f"of {weight_analysis['candidate_parameters']} edge weights"
    )
    print()
    print("Top 3 most sensitive edge weights:")
    for i, result in enumerate(weight_analysis["results"][:3], 1):
        print(f"  {i}. {result['parameter']}")
        print(f"     Original value: {result['original_value']:.2f}")
        print(f"     Sensitivity: ${result['sensitivity']:.2f}")
        print()

    print("=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print()
    print("1. Petersburg automatically tests all parameters (weights, costs, payoffs)")
    print("2. Parameters are ranked by impact on expected value")
    print("3. You get both absolute ($ change) and relative (% change) sensitivity")
    print("4. This tells you WHERE to focus improvement efforts")
    print("5. No manual sensitivity testing required!")
    print()
