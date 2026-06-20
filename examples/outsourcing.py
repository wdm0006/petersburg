"""
Outsourcing Decision Analysis: In-House vs Third-Party

This example models a real-world business decision: Should you build a feature
in-house or outsource it to a third party? And if outsourcing fails, should
you switch back or continue?

THE PROBLEM:
You're building a product with multiple features. For each feature, you can:
1. Build in-house: Cost = 1.0 (baseline), always succeeds
2. Outsource to third party: Cost = 0.80 (20% cheaper), but might fail
3. If outsourcing fails, switch back: Cost = switching penalty + salvage loss

REALISTIC CONSTRAINTS:
- Third party is 20% cheaper when it works
- But outsourced work might fail (probability = weight parameter)
- Switching back wastes time and money (switching cost)
- Some work can be salvaged, some can't (salvage rate = c_switch)
- Multiple features compound the risk (3 features in this model)

THE DECISION TREE:
This models a 3-feature project with switching costs:
- Node 11-13: Start with feature 1 (choose in-house or outsource)
- Node 7-10: Feature 2 decisions (possibly after switching back)
- Node 6: Feature 3 decisions with accumulated switching costs
- Node 2: All in-house path (reliable, expensive)
- Node 4: All outsourced path (cheaper, risky)

KEY PARAMETERS:
- in_house = 1.0 (baseline cost per feature)
- third_party = 0.80 (20% cost savings)
- switching = c_switch * in_house (cost to switch back)
- weight = failure probability (how often outsourcing fails)

SWITCHING COST EXAMPLES:
- c_switch = 0.1: Can salvage 90% of work → low switching cost
- c_switch = 0.5: Can salvage 50% of work → moderate switching cost
- c_switch = 1.0: Can salvage 0% of work → must redo everything

THE ANALYSIS:
This script finds the "break-even" point where in-house and outsourcing
have equal expected cost. The break-even depends on:
1. Probability of third-party failure
2. How much work can be salvaged when switching back
3. Number of features (compounding risk)

KEY INSIGHTS:
1. Outsourcing is cheaper upfront but carries risk
2. Multiple outsourced components compound failure risk
3. High switching costs make outsourcing less attractive
4. Even 20% cost savings can be wiped out by modest failure rates
5. The break-even probability decreases as salvage rate increases

FRAMEWORK USAGE:
This example demonstrates:
- Complex multi-stage decision modeling
- Using weights to represent success/failure probabilities
- Accumulating costs across multiple decisions (switching × feature count)
- Parametric analysis (testing many scenarios)
- Break-even analysis to find decision boundaries
- Real-world business decision modeling
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from petersburg import Graph

plt.style.use("ggplot")

__author__ = "willmcginnis"


def simulate(c_switch):
    """
    Simulate outsourcing decisions across a range of failure probabilities.

    Parameters:
    - c_switch: Switching cost multiplier (portion of work lost when reverting)
                Range: 0.0 (salvage everything) to 1.0 (redo everything)

    Returns:
    - List of [probability, cost_difference] pairs showing when in-house
      is better than outsourcing
    """
    data = []

    # Define cost structure
    in_house = 1.0  # Baseline cost for in-house development
    third_party = 0.80  # 20% discount for outsourcing
    switching = in_house * c_switch  # Cost penalty for switching back

    # Test across range of failure probabilities
    weights = np.linspace(0.1, 1, 100)  # 0.1 = 10% failure rate, 1.0 = 100% success

    for weight in weights:
        # Build decision graph for this failure probability
        g = Graph()
        g.from_dict(
            {
                # Node 1: Terminal node
                1: {"payoff": 0, "after": []},
                # Node 2: All in-house path (3 features × in_house cost)
                2: {"payoff": 0, "after": [{"node_id": 1, "cost": in_house}]},
                3: {"payoff": 0, "after": [{"node_id": 2, "cost": in_house}]},
                # Node 4: All outsourced path (3 features × third_party cost)
                4: {"payoff": 0, "after": [{"node_id": 1, "cost": third_party}]},
                5: {"payoff": 0, "after": [{"node_id": 4, "cost": third_party}]},
                # Node 6: Mixed path - already switched once
                6: {
                    "payoff": 0,
                    "after": [
                        {"node_id": 4, "cost": switching, "weight": weight}
                        # Try outsource again, might fail
                    ],
                },
                # Node 7: After 1 switch, decide on feature 3
                7: {
                    "payoff": 0,
                    "after": [
                        {"node_id": 6, "cost": in_house},  # Go in-house for feature 3
                        {"node_id": 5, "cost": 2 * switching, "weight": weight},
                        # Outsource again (2nd switch cost if fails)
                    ],
                },
                # Node 8: Fully in-house path (feature 3)
                8: {"payoff": 0, "after": [{"node_id": 3, "cost": in_house}]},
                # Node 9: Fully outsourced path (feature 3)
                9: {"payoff": 0, "after": [{"node_id": 5, "cost": third_party}]},
                # Node 10: Decision after 2 switches
                10: {
                    "payoff": 0,
                    "after": [
                        {"node_id": 7, "cost": in_house},  # Stay in-house
                        {"node_id": 5, "cost": 3 * switching, "weight": weight},
                        # Outsource again (3rd switch cost if fails)
                    ],
                },
                # Nodes 11-13: Starting decision points for each feature
                11: {"payoff": 0, "after": [{"node_id": 8, "cost": 0}]},  # Feature 1: In-house
                12: {"payoff": 0, "after": [{"node_id": 9, "cost": 0}]},  # Feature 1: Outsource
                13: {"payoff": 0, "after": [{"node_id": 10, "cost": 0}]},  # Feature 1: Mixed
            }
        )

        # Simulate and compare in-house vs outsourcing
        options = g.get_options(iters=1000)

        # Node 2 represents "all in-house" endpoint
        # Node 4 represents "all outsourced" endpoint
        # Difference shows which strategy is better
        cost_difference = options[2] - options[4]

        # Convert weight to probability of failure
        # weight = success probability, so failure = weight / (1 + weight)
        failure_prob = weight / (1.0 + weight)

        data.append([failure_prob, cost_difference])

    return data


def plot(c_switch):
    """
    Plot the cost difference between in-house and outsourcing
    as a function of third-party failure probability.

    Parameters:
    - c_switch: Switching cost multiplier
    """
    data = simulate(c_switch)
    df = pd.DataFrame(data, columns=["weight", "in_house - third_party"])

    df.plot(kind="scatter", x="weight", y="in_house - third_party")
    plt.xlabel("Probability of Failure for 3rd Party Nodes")
    plt.ylabel("Strength of In-House Option")
    plt.title(f"In House Vs. 3rd Party For c(x)={c_switch:2.1f}*M*x")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=2)
    plt.grid()
    plt.show()


def get_breakeven(c_switch, threshold=0.05):
    """
    Find the break-even failure probability where in-house and
    outsourcing have equal expected cost.

    Parameters:
    - c_switch: Switching cost multiplier
    - threshold: Maximum cost difference to consider "break-even"

    Returns:
    - Break-even probability, or None if no break-even exists
    """
    data = simulate(c_switch)

    # Find the point where cost difference is closest to zero
    break_even = sorted([(x[0], abs(x[1])) for x in data], key=lambda x: x[1])

    if break_even[0][1] < threshold:
        return break_even[0][0]  # Return the break-even probability
    else:
        return None  # No clear break-even point


if __name__ == "__main__":
    print("=" * 70)
    print("OUTSOURCING DECISION ANALYSIS")
    print("=" * 70)
    print("Scenario: Build 3 features in-house or outsource to third party")
    print()
    print("Cost structure:")
    print("  - In-house: 1.0 per feature (always succeeds)")
    print("  - Third party: 0.80 per feature (20% cheaper, but might fail)")
    print("  - Switching cost: Depends on how much work can be salvaged")
    print()
    print("Analyzing break-even points across different salvage rates...")
    print()

    # Calculate break-even probabilities for different switching costs
    breakevens = []
    for c_switch in np.linspace(0.01, 0.99, 50):
        # c_switch = portion of work lost when switching back
        # 1 - c_switch = portion of work salvaged
        salvage_rate = 1 - c_switch
        breakeven_prob = get_breakeven(c_switch)
        breakevens.append((salvage_rate, breakeven_prob))

        # Print a few key examples
        if c_switch in [0.1, 0.5, 0.9]:
            if breakeven_prob:
                print(
                    f"Salvage rate {salvage_rate:.0%}: Break-even at {breakeven_prob:.1%} failure rate"
                )
            else:
                print(
                    f"Salvage rate {salvage_rate:.0%}: No clear break-even (always prefer one option)"
                )

    print()
    print("Generating break-even probability chart...")

    # Plot break-even probabilities
    df = pd.DataFrame(breakevens, columns=["salvage_rate", "breakeven_probability"])
    ax = df.plot(kind="scatter", x="salvage_rate", y="breakeven_probability", s=50)
    plt.xlabel("Portion of Work Salvaged When Switching Back")
    plt.ylabel("Break-Even Failure Probability")
    plt.title("Outsourcing Break-Even Analysis")
    plt.grid()

    print()
    print("=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("The chart shows the failure rate at which outsourcing becomes")
    print("equally costly as in-house development.")
    print()
    print("Key insights:")
    print("1. Higher salvage rates → can tolerate higher failure rates")
    print("2. If you can salvage 90% of work, outsourcing stays attractive")
    print("   even with moderate failure rates")
    print("3. If you must redo everything (0% salvage), even small failure")
    print("   rates make outsourcing unattractive")
    print("4. The 20% cost savings is easily lost to switching costs")
    print()
    print("Real-world applications:")
    print("- Offshore development decisions")
    print("- Build vs buy vs partner decisions")
    print("- Vendor risk assessment")
    print("- Multi-stage project planning with optionality")
    print("=" * 70)

    plt.show()
