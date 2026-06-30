"""
The Necktie Paradox (Wallet Game)

A variant of the Two Envelope Paradox with an interesting twist: both players
can simultaneously believe they have the advantage!

THE PARADOX:
Two people compare their neckties (or wallet contents). Each says:
"Either my tie cost more than yours, or yours cost more than mine.
If mine cost more, I win the value of your tie.
If yours cost more, I lose the value of my tie.

Let's say my tie cost $100:
- If I win, I gain an unknown amount (your tie's value)
- If I lose, I lose $100

But YOUR tie cost at least $50 (conservative estimate) and could be $200.
Expected gain = 0.5 × $50 + 0.5 × $200 = $125
Expected loss = $100
Net expected value = +$25

So I should bet! But YOU can make the same argument, which is contradictory."

THE RESOLUTION:
The fallacy is in assuming symmetric probabilities without a prior distribution.
When you properly model the scenario with known values, the paradox dissolves.

THE SETUP IN THIS SIMULATION:
Two players with ties worth $50 and $40:
- Node 4: Player has $50 tie, must pay $40 (cost node 2) → net = $10
- Node 5: Player has $40 tie, must pay $50 (cost node 3) → net = -$10
- Each player is equally likely (50/50)
- Overall expected value = 0.5 × $10 + 0.5 × (-$10) = $0

This demonstrates:
1. With proper accounting of costs and payoffs, EV = 0 (zero-sum game)
2. The paradox comes from asymmetric information and selective reasoning
3. You can't both have positive EV in a zero-sum game
4. Convergence analysis confirms the game is fair on average

KEY INSIGHTS:
- Both players can't simultaneously have an advantage in a fair game
- The paradox exploits ambiguity about prior distributions
- Proper cost accounting reveals the true expected value
- This is fundamentally a zero-sum game with no net value creation

FRAMEWORK USAGE:
This example demonstrates:
- Modeling costs vs payoffs (cost parameter in transitions)
- Convergence visualization with min/max bounds
- How to resolve paradoxes with proper graph structure
- Simulating games from a neutral starting point
"""

import matplotlib.pyplot as plt
import matplotlib.style
import pandas as pd

from petersburg import Graph

matplotlib.style.use("ggplot")

__author__ = "willmcginnis"

if __name__ == "__main__":
    # Initialize the decision graph
    g = Graph()

    # Build the Necktie Paradox as a decision graph
    # Node 1: Terminal node (game ends)
    # Node 2: End state with net cost of $40
    # Node 3: End state with net cost of $50
    # Node 4: Player wins with $50 tie, pays $40 to other player
    # Node 5: Player wins with $40 tie, pays $50 to other player
    g.from_dict(
        {
            1: {"payoff": 0, "after": []},  # Terminal node
            2: {"payoff": 0, "after": [{"node_id": 1, "cost": 40}]},  # Cost: $40 payment to winner
            3: {"payoff": 0, "after": [{"node_id": 1, "cost": 50}]},  # Cost: $50 payment to winner
            4: {
                "payoff": 50,  # Win $50 (value of your tie)
                "after": [{"node_id": 2, "cost": 0}],  # Pay $40 to other player → net $10
            },
            5: {
                "payoff": 40,  # Win $40 (value of your tie)
                "after": [{"node_id": 3, "cost": 0}],  # Pay $50 to other player → net -$10
            },
        }
    )

    print("=" * 70)
    print("NECKTIE PARADOX SIMULATION")
    print("=" * 70)
    print("Two players compare ties worth $50 and $40")
    print("Winner takes the loser's tie but pays their own tie's value")
    print()
    print("Player 1 logic: 'I have the $50 tie. If I win, net +$10.'")
    print("Player 2 logic: 'I have the $40 tie. If I win, net +$10.'")
    print()
    print("Both can't be right! Let's simulate to find the truth...")
    print()

    # Run simulations with increasing iterations to show convergence
    data = []
    for iter in [5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 1000000]:
        # Simulate the game multiple times
        outcomes = []
        for _ in range(iter):
            outcomes.append(g.get_outcome())

        # Collect statistics
        mean_outcome = float(sum(outcomes)) / len(outcomes)
        min_outcome = min(outcomes)
        max_outcome = max(outcomes)

        data.append([iter, mean_outcome, min_outcome, max_outcome])

        # Print intermediate results for key iteration counts
        if iter in [100, 10000, 1000000]:
            print(f"After {iter:,} iterations: EV = ${mean_outcome:.2f}")

    # Create visualization of convergence
    df = pd.DataFrame(data, columns=["iters", "outcome", "min", "max"])

    # Plot mean outcome over iterations
    ax = df.plot(kind="line", x="iters", y="outcome", label="Expected Value", color="blue")

    # Add min/max markers to show variance
    df.plot(
        ax=ax, kind="scatter", x="iters", y="min", s=100, marker="+", color="red", label="Min/Max"
    )
    df.plot(ax=ax, kind="scatter", x="iters", y="max", s=100, marker="+", color="red")

    # Configure the plot
    plt.title("Necktie Paradox: Convergence to True Expected Value")
    plt.xlabel("Number of Iterations in Simulation")
    plt.ylabel("Net Outcome ($)")
    plt.xscale("log")  # Log scale shows convergence more clearly
    plt.grid(True, color="w", linestyle="-", linewidth=1)
    plt.gca().patch.set_facecolor("0.8")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=2, label="Fair Game (EV=0)")
    plt.legend()

    print()
    print("=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print(f"Final EV converges to: ${data[-1][1]:.2f}")
    print()
    print("As expected, this is a ZERO-SUM GAME!")
    print("- 50% chance: Win $50 tie, pay $40 → net +$10")
    print("- 50% chance: Win $40 tie, pay $50 → net -$10")
    print("- Expected value: 0.5 × $10 + 0.5 × (-$10) = $0")
    print()
    print("The paradox dissolves when you properly account for:")
    print("1. The actual values involved ($50 and $40, not unknowns)")
    print("2. The costs of participation (payment to winner)")
    print("3. Equal probability of each outcome (50/50)")
    print()
    print("Both players CAN'T have positive EV in a zero-sum game!")
    print("The 'paradox' exploits vague reasoning about unknown values.")
    print("=" * 70)

    plt.show()
