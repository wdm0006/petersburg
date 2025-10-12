"""
The Two Envelope Paradox (Exchange Paradox)

This is one of the most famous probability paradoxes, demonstrating how naive
expected value reasoning can lead to contradictory conclusions.

THE PARADOX:
You're given two sealed envelopes containing money. You're told:
- One envelope contains X dollars
- The other contains 2X dollars
- You don't know which is which

You pick an envelope and open it, finding $100. Now you're offered a switch.

FAULTY REASONING:
"The other envelope has either $50 or $200, with equal probability.
Expected value of switching = 0.5 × $50 + 0.5 × $200 = $125
My current envelope = $100
Therefore I should switch! (Gain of $25 expected)"

But wait - you could apply this logic BEFORE opening the envelope too!
And you could apply it AGAIN after switching! This suggests you should
switch forever, which is obviously wrong.

THE RESOLUTION:
The error is assuming equal probability for $50 and $200 given that you
observed $100. The actual probabilities depend on the PRIOR distribution
of possible amounts. With uniform weights, switching gives NO advantage.

THE SETUP IN THIS SIMULATION:
- Node 4: Start with $100, can go to node 2 (keep) or node 3 (switch to $50)
- Node 5: Start with $50, can go to node 3 (keep) or node 2 (switch to $100)
- Each scenario is equally likely (50/50 initial draw)

This shows that:
- If you have $100 and switch → sometimes win $50, sometimes win $0 (end at 3)
- If you have $50 and switch → sometimes win $100, sometimes win $0 (end at 2)
- Net result: Switching gives ZERO expected advantage

KEY INSIGHTS:
1. Expected value calculations require proper conditional probabilities
2. The paradox arises from incorrectly assuming symmetric probabilities
3. The framework correctly shows that switching has no benefit
4. Convergence analysis shows both strategies converge to the same EV

FRAMEWORK USAGE:
This example demonstrates:
- Using get_options() to compare different decision paths
- Extended statistics (min, max, mean, count)
- Convergence visualization with logarithmic scaling
- How to visualize when two strategies are equivalent
"""

import matplotlib.pyplot as plt
import matplotlib.style

matplotlib.style.use("ggplot")
import pandas as pd

from petersburg import Graph

__author__ = "willmcginnis"

if __name__ == "__main__":
    # Initialize the decision graph
    g = Graph()

    # Build the Two Envelope Paradox as a decision graph
    # Node 1: Terminal node (end of game)
    # Node 2: End with envelope containing $100 (either kept or switched to)
    # Node 3: End with envelope containing $50 (either kept or switched to)
    # Node 4: Start state - have $100, decide to keep (→2) or switch (→3)
    # Node 5: Start state - have $50, decide to keep (→3) or switch (→2)
    g.from_dict(
        {
            1: {"payoff": 0, "after": []},  # Terminal node
            2: {"payoff": 0, "after": [{"node_id": 1}]},  # End with $100 envelope
            3: {"payoff": 0, "after": [{"node_id": 1}]},  # End with $50 envelope
            4: {
                "payoff": 100,  # Currently holding envelope with $100
                "after": [
                    {"node_id": 2, "cost": 0},  # Option 1: Keep the $100
                    {"node_id": 3, "cost": 0},  # Option 2: Switch (get $50)
                ]
            },
            5: {
                "payoff": 50,  # Currently holding envelope with $50
                "after": [
                    {"node_id": 3, "cost": 0},  # Option 1: Keep the $50
                    {"node_id": 2, "cost": 0},  # Option 2: Switch (get $100)
                ]
            },
        }
    )

    print("=" * 70)
    print("TWO ENVELOPE PARADOX SIMULATION")
    print("=" * 70)
    print("Testing the 'always switch' strategy vs 'never switch'")
    print("Running simulations with increasing iterations to show convergence...")
    print()

    # Collect data for both strategies across different iteration counts
    data_2 = []  # "Always switch" outcomes (end at node 2)
    data_3 = []  # "Never switch" outcomes (end at node 3)

    # Test with increasing numbers of iterations to demonstrate convergence
    for iter in [5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 1000000]:
        # Get outcomes for each decision path
        # Node 2 represents the "switch" decision endpoint
        # Node 3 represents the "don't switch" decision endpoint
        outcomes = g.get_options(iters=iter, extended_stats=True)

        # Collect statistics for switching strategy (node 2)
        data_2.append(
            [outcomes[2]["count"], outcomes[2]["mean"], outcomes[2]["min"], outcomes[2]["max"]]
        )

        # Collect statistics for non-switching strategy (node 3)
        data_3.append(
            [outcomes[3]["count"], outcomes[3]["mean"], outcomes[3]["min"], outcomes[3]["max"]]
        )

    # Create DataFrames for visualization
    df = pd.DataFrame(data_2, columns=["iters", "outcome", "min", "max"])
    df2 = pd.DataFrame(data_3, columns=["iters", "outcome", "min", "max"])

    # Plot convergence of both strategies
    # Blue: Switch strategy
    ax = df.plot(kind="line", x="iters", y="outcome", label="switch", color="blue")
    df.plot(ax=ax, kind="scatter", x="iters", y="min", s=100, marker="+", color="blue")
    df.plot(ax=ax, kind="scatter", x="iters", y="max", s=100, marker="+", color="blue")

    # Red: Don't switch strategy
    df2.plot(ax=ax, kind="line", x="iters", y="outcome", label="don't switch", color="red")
    df2.plot(ax=ax, kind="scatter", x="iters", y="min", s=100, marker="+", color="red")
    df2.plot(ax=ax, kind="scatter", x="iters", y="max", s=100, marker="+", color="red")

    # Configure the plot
    plt.title("Two Envelope Paradox: Strategy Convergence Analysis")
    plt.xlabel("Number of Iterations in Simulation")
    plt.ylabel("Expected Outcome ($)")
    plt.xscale("log")  # Log scale shows convergence more clearly
    plt.grid(True, color="w", linestyle="-", linewidth=1)
    plt.gca().patch.set_facecolor("0.8")

    print("INTERPRETATION:")
    print("=" * 70)
    print("The plot shows both strategies converge to the SAME expected value.")
    print()
    print("Blue line (switch): Expected value when you always switch")
    print("Red line (don't switch): Expected value when you never switch")
    print()
    print("Notice: Both converge to approximately $75")
    print("  = 0.5 × $50 + 0.5 × $100")
    print("  = The average of the two envelope amounts")
    print()
    print("The '+' markers show min/max variation, which decreases as")
    print("iterations increase (law of large numbers).")
    print()
    print("CONCLUSION: Switching provides NO advantage!")
    print("The paradox comes from incorrect probability reasoning.")
    print("=" * 70)

    plt.show()
