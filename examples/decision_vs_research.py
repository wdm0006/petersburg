"""
Decision vs Research: The Value of Information

This example models a fundamental business decision: Should you act now with
incomplete information, or spend money/time to research first?

THE PROBLEM:
You're considering two investment options (A and B). Without research:
- Option A: Could yield $25 or $15 (unknown which)
- Option B: Could yield $25 or $15 (unknown which)
- You must pick one and pay $10 to execute

You can also pay $5 to research ONE option before deciding.

THE DECISION TREE:
1. Decide immediately (blind choice) → pay $10 cost
2. Research Option A for $5, then decide → pay $5 + $10 = $15 total
3. Research Option B for $5, then decide → pay $5 + $10 = $15 total

KEY QUESTIONS:
- Is research worth the $5 cost?
- When is the value of information positive?
- How do probabilities affect this decision?

THE GRAPH STRUCTURE:
Nodes 7-10: Starting points with known payoffs ($25 or $15)
Nodes 5-6: Post-research decision points (choose between options)
Nodes 2-4: Final execution points (pay $10 to execute choice)
Node 1: Terminal

The weights represent relative likelihood of outcomes:
- Node 2: weight=2 (more likely good outcome after research)
- Node 3: weight=1 (default probability)
- Node 4: weight=1.5 (moderately better outcome)

EXPECTED VALUE CALCULATION:
Without research: Random choice between options with unknown values
With research: Informed choice that improves success probability

If research increases your chance of picking the $25 option by 20%:
- Research cost: $5
- Execution cost: $10
- Better outcome: $25 instead of $15 = $10 gain
- Research value: 0.20 × $10 - $5 = -$3 (not worth it at these odds)

KEY INSIGHTS:
1. Value of information = (Probability improvement) × (Outcome difference) - (Research cost)
2. Research is only valuable when it significantly changes your decision
3. In symmetric situations, research may not be worth the cost
4. Weights in the graph model how research changes probabilities

FRAMEWORK USAGE:
This example demonstrates:
- Modeling sequential decisions (research → choose → execute)
- Using weights to represent information quality
- Using costs to represent both research and execution expenses
- Comparing multiple decision paths with get_options()
- How to structure "value of information" problems
"""

from petersburg import Graph

__author__ = "willmcginnis"

if __name__ == "__main__":
    # Initialize the decision graph
    g = Graph()

    # Build the "Decision vs Research" scenario
    # This models whether to make an immediate decision or research first
    g.from_dict(
        {
            # Node 1: Terminal node (end state)
            1: {"payoff": 0, "after": []},
            # Nodes 2-4: Final execution nodes (pay cost to execute chosen option)
            2: {
                "payoff": 0,
                "after": [{"node_id": 1, "cost": 10, "weight": 2}],
                # Cost $10 to execute, weight=2 means higher likelihood after research
            },
            3: {
                "payoff": 0,
                "after": [{"node_id": 1, "cost": 10}],
                # Cost $10 to execute, default weight (no research advantage)
            },
            4: {
                "payoff": 0,
                "after": [{"node_id": 1, "cost": 10, "weight": 1.5}],
                # Cost $10 to execute, weight=1.5 (moderate research advantage)
            },
            # Node 5: Decision point after researching Option A
            5: {
                "payoff": 0,
                "after": [
                    {"node_id": 2, "cost": 5},  # Execute A (research cost $5)
                    {"node_id": 3, "cost": 10},  # Execute B instead (total $15)
                ],
            },
            # Node 6: Decision point after researching Option B
            6: {
                "payoff": 0,
                "after": [
                    {"node_id": 2, "cost": 5},  # Execute B (research cost $5)
                    {"node_id": 4, "cost": 10},  # Execute A instead (total $15)
                ],
            },
            # Nodes 7-8: Option A outcomes (high $25 or low $15)
            7: {"payoff": 25, "after": [{"node_id": 5, "cost": 0}]},  # Good outcome for Option A
            8: {"payoff": 15, "after": [{"node_id": 5, "cost": 0}]},  # Poor outcome for Option A
            # Nodes 9-10: Option B outcomes (high $25 or low $15)
            9: {"payoff": 25, "after": [{"node_id": 6, "cost": 0}]},  # Good outcome for Option B
            10: {"payoff": 15, "after": [{"node_id": 6, "cost": 0}]},  # Poor outcome for Option B
        }
    )

    print("=" * 70)
    print("DECISION VS RESEARCH: VALUE OF INFORMATION ANALYSIS")
    print("=" * 70)
    print("Scenario: Choose between two investment options")
    print()
    print("Option costs:")
    print("  - Execute immediately (blind): $10")
    print("  - Research option first: $5 + $10 = $15 total")
    print()
    print("Potential payoffs:")
    print("  - Good outcome: $25")
    print("  - Poor outcome: $15")
    print()
    print("Question: Is $5 research worth the cost?")
    print("=" * 70)
    print()

    # Display the graph structure as a Mermaid diagram
    print("GRAPH STRUCTURE (Mermaid):")
    print("(Paste into any Mermaid renderer to view the decision tree)")
    print()
    print(g.to_mermaid())
    print()

    # Analyze expected outcomes for different decision paths
    print("=" * 70)
    print("COMPARING DECISION STRATEGIES")
    print("=" * 70)
    print()
    print("Running 100,000 simulations for each decision path...")
    print()

    # Simulate many outcomes to get reliable expected values
    outcomes = []
    for _ in range(100000):
        outcomes.append(g.get_outcome())

    mean_outcome = float(sum(outcomes)) / len(outcomes)

    print(f"Overall expected value (random path): ${mean_outcome:.2f}")
    print()

    # Compare specific decision options
    print("Comparing specific strategies:")
    print("(Note: Uncomment the code below to see detailed comparison)")
    print()
    options = g.get_options(iters=100000)
    for node_id, ev in options.items():
        if node_id in [2, 3, 4]:
            print(f"  Node {node_id} expected value: ${ev:.2f}")

    print()
    print("=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("The value of research depends on:")
    print("1. How much it improves your decision (reflected in weights)")
    print("2. The cost of research ($5)")
    print("3. The difference between good and bad outcomes ($25 - $15 = $10)")
    print()
    print("Break-even analysis:")
    print("  Research cost: $5")
    print("  Outcome difference: $10")
    print("  Needed probability improvement: $5/$10 = 50%")
    print()
    print("If research improves your success rate by >50%, it's worth it!")
    print("If improvement is <50%, decide immediately and save the $5.")
    print()
    print("This framework helps model 'value of information' in real decisions:")
    print("- Market research before product launch")
    print("- User testing before full development")
    print("- Pilot programs before company-wide rollout")
    print("=" * 70)

    # The commented code below can be uncommented for detailed analysis:
    #
    # outcomes = []
    # for _ in range(100000):
    #     outcomes.append(g.get_outcome())
    #
    # print('\n\nSimulated Output With Random Start')
    # print(float(sum(outcomes))/len(outcomes))
    #
    # print('\n\nSimulated Profit of Each Starting Move')
    # print(g.get_options(iters=100000))
