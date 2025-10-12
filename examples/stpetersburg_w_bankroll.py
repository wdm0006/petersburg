"""
St. Petersburg Paradox with Casino Bankroll Constraints

This example demonstrates how the casino's finite bankroll fundamentally changes
the St. Petersburg game from infinite expected value to finite expected value.

PROBLEM:
The classic St. Petersburg paradox assumes the casino can pay ANY amount:
- Flip 1 tails: Win $2
- Flip 2 tails: Win $4
- Flip 3 tails: Win $8
- Flip 40 tails: Win $1,099,511,627,776 (over $1 trillion!)

But no real casino has infinite money. What happens when we cap payouts?

REALISTIC CONSTRAINTS:
This simulation tests different casino bankrolls:
1. Small casino: $100 (can only pay up to flip 6: $64)
2. Regional casino: $10 million (can pay up to flip 23: $8.4M)
3. Major casino: $10 billion (can pay up to flip 33: $8.6B)
4. Bill Gates in 2014: $79.2 billion (can pay up to flip 36: $68.7B)

KEY INSIGHT:
As the bankroll increases, expected value grows BUT converges to a finite limit!
The EV is approximately: (number of possible flips) / 2

For example:
- With $100 bankroll: 6 flips possible → EV ≈ $3
- With $10M bankroll: 23 flips possible → EV ≈ $11.50
- With infinite bankroll: infinite flips → EV = ∞

This shows that the "infinite" EV is purely theoretical. Any real constraint
makes the game finite and much less valuable than theory suggests.

FRAMEWORK USAGE:
This example shows how petersburg handles:
- Dynamic graph construction (building nodes until payoff exceeds bankroll)
- Large-scale Monte Carlo simulation (10 million iterations)
- Testing multiple scenarios in a loop
"""

from petersburg import Graph

__author__ = "willmcginnis"

if __name__ == "__main__":
    # Test four different casino bankroll levels
    # Each represents a different scale of financial institution
    bankroll_levels = [
        100,           # Small local casino
        10e6,          # $10 million - regional casino
        10e9,          # $10 billion - major casino company
        79200000000    # $79.2 billion - Bill Gates' wealth in 2014
    ]

    print("=" * 70)
    print("ST. PETERSBURG PARADOX: CASINO BANKROLL CONSTRAINTS")
    print("=" * 70)
    print("Theoretical EV with infinite bankroll: INFINITE")
    print("What happens when the casino has limited funds?\n")

    for bankroll in bankroll_levels:
        # Initialize a new graph for this bankroll level
        g = Graph()

        # No entrance fee in this version - we're measuring pure EV
        entrance_fee = 0

        # Start building the decision graph
        # Node 1: Terminal node (game ends)
        # Node 2: Entry point (free to play in this version)
        gd = {
            1: {"payoff": 0, "after": []},  # Terminal node
            2: {"payoff": 0, "after": [{"node_id": 1, "cost": entrance_fee}]},  # Entry point
        }

        # Build the coin flip sequence dynamically
        # Stop when the next payout would exceed the casino's bankroll
        nn = 3  # Next node ID to assign
        idx = 0  # Flip counter
        payoff = 2 ** (idx + 1)  # Calculate payout for this flip ($2, $4, $8, ...)

        # Keep adding flips while the casino can afford to pay
        max_flips = 0
        while payoff <= bankroll:
            node_id = 2 * (idx + 1)  # Determine next node in the chain

            # Heads outcome: Win the payout for this flip
            gd[nn] = {"payoff": payoff, "after": [{"node_id": node_id, "cost": 0, "weight": 1}]}
            nn += 1

            # Tails outcome: Game ends, no additional payout
            gd[nn] = {"payoff": 0, "after": [{"node_id": node_id, "cost": 0, "weight": 1}]}
            nn += 1

            # Move to next flip
            idx += 1
            max_flips = idx
            payoff = 2 ** (idx + 1)

        # Load the dynamically constructed graph
        g.from_dict(gd)

        # Run large-scale Monte Carlo simulation
        # Using 10 million iterations for high precision
        print(f"Simulating with ${bankroll:,.0f} casino bankroll...")
        print(f"  Maximum flips possible: {max_flips}")
        print(f"  Maximum payout: ${2**max_flips:,.0f}")

        outcomes = []
        for _ in range(10000000):
            outcomes.append(g.get_outcome())

        ev = float(sum(outcomes)) / len(outcomes)
        print(f"  Expected Value: ${ev:.2f}")
        print(f"  EV / Max Flips: ${ev / max_flips:.2f}")
        print()

    print("=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("Notice how EV grows logarithmically with bankroll, not linearly.")
    print("Doubling the casino's money does NOT double the game's value.")
    print()
    print("The pattern: EV ≈ (max_flips) / 2")
    print("This is because each flip has 50% chance and adds ~$1 to EV.")
    print()
    print("Key takeaway: 'Infinite' EV requires infinite resources.")
    print("Real-world constraints make this game worth only a few dollars!")
    print("=" * 70)
