"""
Graph Visualization and Export Demo

This example demonstrates the petersburg framework's visualization capabilities,
showing how to generate visual representations of decision graphs.

THE PURPOSE:
Decision graphs can be complex with many nodes, edges, costs, and weights.
Visualization helps you:
1. Understand the structure of your decision tree
2. Identify critical paths and bottlenecks
3. Communicate decision logic to stakeholders
4. Debug graph construction issues
5. Document your analysis

THE EXAMPLE GRAPH:
This uses the same outsourcing decision structure from outsourcing.py:
- Nodes 11-13: Initial decision points for 3 features
- Nodes 7-10: Mid-level decisions with switching costs
- Nodes 2-4: Final cost endpoints (in-house vs outsourced)
- Node 1: Terminal node

PARAMETERS IN THIS EXAMPLE:
- in_house = 1.0 (baseline cost)
- third_party = 0.5 (50% cheaper)
- switching = 2.0 (high cost to revert)
- weight = 0.67 (67% success rate for outsourcing)

THE GRAPH PLOT METHOD:
g.plot(filepath) generates a visualization showing:
- Nodes as circles (labeled with node IDs)
- Edges as arrows (showing transitions)
- Edge labels showing:
  * cost = transition cost
  * weight = relative probability (if not 1.0)
  * payoff = immediate reward at that node (if any)

VISUALIZATION USES:
1. **Initial Design**: Verify your graph structure is correct
2. **Communication**: Share decision logic with non-technical stakeholders
3. **Documentation**: Include in reports and presentations
4. **Debugging**: Identify structural issues (cycles, unreachable nodes, etc.)
5. **Teaching**: Help others understand decision graph concepts

OUTPUT FORMAT:
The plot() method exports to PNG format by default. The framework uses
graphviz/networkx under the hood to create clear, readable visualizations.

TIPS FOR EFFECTIVE VISUALIZATION:
1. Keep node IDs meaningful (e.g., 1 = terminal, 2-4 = outcomes)
2. Use comments in your graph dict to document node purposes
3. For large graphs, consider breaking into subgraphs
4. Use consistent naming conventions (costs, weights, payoffs)
5. Save visualizations in version control to track changes

KEY INSIGHTS:
- Visualization turns abstract graphs into concrete diagrams
- Visual inspection often reveals issues that code inspection misses
- Stakeholders understand pictures better than code
- Documentation through visualization reduces misunderstandings
- Good visualization is part of good analysis

FRAMEWORK USAGE:
This example demonstrates:
- The plot() method for graph visualization
- Proper file path handling with os.path
- Creating output directories for artifacts
- Documenting complex decision structures
- Best practices for graph visualization
"""

import os

from petersburg import Graph

__author__ = "willmcginnis"

if __name__ == "__main__":
    print("=" * 70)
    print("PETERSBURG GRAPH VISUALIZATION DEMO")
    print("=" * 70)
    print()
    print("This example demonstrates how to visualize decision graphs using")
    print("the built-in plot() method.")
    print()

    # Define the graph parameters
    # These represent an outsourcing decision with high switching costs
    c_switch = 2            # High switching cost (200% of baseline)
    in_house = 1            # Baseline in-house development cost
    third_party = 0.5       # Third party is 50% cheaper
    switching = c_switch    # Cost to revert from outsourcing
    weight = 0.67           # 67% probability of outsourcing success

    print("Graph parameters:")
    print(f"  In-house cost: {in_house}")
    print(f"  Third-party cost: {third_party} (50% cheaper)")
    print(f"  Switching cost: {switching} (200% penalty)")
    print(f"  Success weight: {weight} (67% success rate)")
    print()

    # Initialize the graph
    g = Graph()

    # Build a multi-stage outsourcing decision graph
    # This is the same structure used in outsourcing.py
    print("Building graph structure...")
    g.from_dict(
        {
            # Node 1: Terminal node (end of decision path)
            1: {"payoff": 0, "after": []},

            # Nodes 2-3: In-house development path (3 features)
            2: {"payoff": 0, "after": [{"node_id": 1, "cost": in_house}]},
            3: {"payoff": 0, "after": [{"node_id": 2, "cost": in_house}]},

            # Nodes 4-5: Outsourced development path (3 features)
            4: {"payoff": 0, "after": [{"node_id": 1, "cost": third_party}]},
            5: {"payoff": 0, "after": [{"node_id": 4, "cost": third_party}]},

            # Node 6: After first switch, try outsourcing again
            6: {
                "payoff": 0,
                "after": [{"node_id": 4, "cost": switching, "weight": weight}]
            },

            # Node 7: Second feature decision after one switch
            7: {
                "payoff": 0,
                "after": [
                    {"node_id": 6, "cost": in_house},           # Stay in-house
                    {"node_id": 5, "cost": 2 * switching, "weight": weight},  # Switch again
                ],
            },

            # Node 8: Fully in-house path for feature 3
            8: {"payoff": 0, "after": [{"node_id": 3, "cost": in_house}]},

            # Node 9: Fully outsourced path for feature 3
            9: {"payoff": 0, "after": [{"node_id": 5, "cost": third_party}]},

            # Node 10: Third feature decision after two switches
            10: {
                "payoff": 0,
                "after": [
                    {"node_id": 7, "cost": in_house},           # Stay in-house
                    {"node_id": 5, "cost": 3 * switching, "weight": weight},  # Switch third time
                ],
            },

            # Nodes 11-13: Starting decision points for each path
            11: {"payoff": 0, "after": [{"node_id": 8, "cost": 0}]},   # Start with all in-house
            12: {"payoff": 0, "after": [{"node_id": 9, "cost": 0}]},   # Start with all outsourced
            13: {"payoff": 0, "after": [{"node_id": 10, "cost": 0}]},  # Start with mixed strategy
        }
    )

    print("Graph built successfully!")
    print(f"  Total nodes: {len(g.nodes)}")
    print()

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
    os.makedirs(output_dir, exist_ok=True)

    # Generate the visualization
    output_path = os.path.join(output_dir, "print.png")
    print(f"Generating visualization...")
    print(f"  Output path: {output_path}")
    print()

    try:
        g.plot(output_path)
        print("SUCCESS! Graph visualization saved.")
        print()
        print("=" * 70)
        print("INTERPRETING THE VISUALIZATION:")
        print("=" * 70)
        print()
        print("The visualization shows:")
        print("  - Circles: Nodes (labeled with node IDs)")
        print("  - Arrows: Transitions between nodes")
        print("  - Edge labels: Costs, weights, and payoffs")
        print()
        print("Key nodes to look for:")
        print("  - Node 1: Terminal (end of all paths)")
        print("  - Node 2-4: Final cost accumulation nodes")
        print("  - Node 5-10: Mid-level decision points")
        print("  - Node 11-13: Starting points (in-house, outsourced, mixed)")
        print()
        print("Notice how:")
        print("  - Multiple paths lead to the same endpoints")
        print("  - Switching costs accumulate (1x, 2x, 3x)")
        print("  - Weights show probability of success/failure")
        print()
        print("=" * 70)
        print("USE CASES FOR VISUALIZATION:")
        print("=" * 70)
        print()
        print("1. Initial design verification")
        print("   → Ensure your graph structure matches your mental model")
        print()
        print("2. Stakeholder communication")
        print("   → Share decision logic with non-technical team members")
        print()
        print("3. Documentation")
        print("   → Include in reports, presentations, and documentation")
        print()
        print("4. Debugging")
        print("   → Spot structural issues (cycles, orphaned nodes, etc.)")
        print()
        print("5. Teaching")
        print("   → Help others learn decision graph concepts")
        print()
        print("=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print()
        print("1. Open the generated image to see the full visualization")
        print("2. Try modifying the graph parameters and regenerating")
        print("3. Experiment with different graph structures")
        print("4. Use visualization throughout your analysis workflow")
        print()
        print("See other examples for different types of decision graphs:")
        print("  - stpetersburg.py: Simple sequential decisions")
        print("  - two_envelope_problem.py: Symmetric decision trees")
        print("  - outsourcing.py: Complex multi-stage decisions")
        print("=" * 70)

    except Exception as e:
        print(f"ERROR: Failed to generate visualization")
        print(f"  {str(e)}")
        print()
        print("Note: Graph visualization requires graphviz to be installed.")
        print("Install with: pip install graphviz")
        print("Or: brew install graphviz (on macOS)")
        print("=" * 70)
