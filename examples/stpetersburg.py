"""
The St. Petersburg Paradox

This is the classic paradox that inspired the petersburg framework.

PROBLEM:
A casino offers you a coin-flip game with these rules:
- Pay $10 entrance fee
- Flip a coin repeatedly until you get tails
- If tails appears on flip 1: Win $2
- If tails appears on flip 2: Win $4
- If tails appears on flip 3: Win $8
- And so on, doubling each time: $2^n for n flips

The expected value is INFINITE:
EV = (1/2)×$2 + (1/4)×$4 + (1/8)×$8 + (1/16)×$16 + ...
   = $1 + $1 + $1 + $1 + ...
   = ∞

THE PARADOX:
Despite infinite expected value, no rational person would pay $10,000 to play.
Why? Because:
1. You have finite wealth (bankroll constraint)
2. The casino has finite wealth (can't actually pay $2^100)
3. Utility of money isn't linear (diminishing marginal utility)

This example demonstrates realistic constraints:
- $10 entrance fee
- $1,000 starting bankroll
- Only 10 coin flips maximum (max payout: $2^10 = $1,024)

KEY INSIGHTS:
- Expected value calculations must account for real-world constraints
- Infinite theoretical EV ≠ infinite practical value
- Risk of ruin matters when you have a limited bankroll
"""

from petersburg import Graph

__author__ = "willmcginnis"

if __name__ == "__main__":
    # Initialize the petersburg graph
    g = Graph()

    # Configuration: St. Petersburg game with realistic constraints
    entrance_fee = 10  # Cost to play the game
    starting_bankroll = 1000  # Your available capital
    max_flips = 10  # Maximum coin flips (casino's constraint)

    # Build the decision graph
    # Node 1: Terminal node (game ends)
    # Node 2: Entry point (pay entrance fee to play)
    gd = {
        1: {"payoff": 0, "after": []},  # Terminal: game over
        2: {
            "payoff": 0,
            "after": [{"node_id": 1, "cost": entrance_fee}],
        },  # Entry: pay to play
    }

    # Build the coin flip sequence
    # For each flip n:
    #   - Node (odd):  Heads → win $2^n and continue to end
    #   - Node (even): Tails → win $0 and continue to end
    nn = 3  # Next node ID to assign
    for idx in range(max_flips):
        # On flip n, you either:
        # 1. Get HEADS: Win $2^(n+1)
        # 2. Get TAILS: Game ends, no additional payout
        node_id = 2 * (idx + 1)  # Where to go next (towards terminal)
        payoff = 2 ** (idx + 1)  # Payout for heads on this flip: $2, $4, $8, ...

        # Heads outcome: Win the payoff for this round
        gd[nn] = {"payoff": payoff, "after": [{"node_id": node_id, "cost": 0, "weight": 1}]}
        nn += 1

        # Tails outcome: No additional payout, game ends
        gd[nn] = {"payoff": 0, "after": [{"node_id": node_id, "cost": 0, "weight": 1}]}
        nn += 1

    # Load the graph structure
    g.from_dict(gd)

    # Simulate the game with realistic constraints
    print("=" * 70)
    print("ST. PETERSBURG PARADOX SIMULATION")
    print("=" * 70)
    print(f"Entrance Fee: ${entrance_fee}")
    print(f"Starting Bankroll: ${starting_bankroll}")
    print(f"Maximum Flips: {max_flips} (max payout: ${2**max_flips})")
    print()
    print("Running 1,000 simulations with risk of ruin...")
    print()

    # Run Monte Carlo simulation with ruin detection
    # Each simulation plays the game 1,000 times or until bankroll = 0
    outcomes = []
    for _ in range(1000):
        # get_outcome with ruin=True stops if bankroll reaches 0
        outcome = g.get_outcome(iters=1000, ruin=True, starting_bank=starting_bankroll)
        outcomes.append(outcome)

    # Calculate statistics
    mean_outcome = float(sum(outcomes)) / len(outcomes)
    min_outcome = min(outcomes)
    max_outcome = max(outcomes)

    print(f"Mean Outcome: ${mean_outcome:.2f}")
    print(f"Best Outcome: ${max_outcome:.2f}")
    print(f"Worst Outcome: ${min_outcome:.2f}")
    print()
    print("INTERPRETATION:")
    print(f"  Starting with $1,000 and paying ${entrance_fee} per game, your bankroll")
    print(f"  changes by an average of ${mean_outcome:.2f} after 1,000 games.")
    print()
    if mean_outcome < 0:
        print("  Despite 'infinite' theoretical EV, you LOSE money on average!")
        print("  This is the paradox: theory vs. reality with finite constraints.")
    else:
        print("  You make money on average, but with high variance.")
        print(f"  Best case: +${max_outcome:.2f} | Worst case: ${min_outcome:.2f} (ruin)")
    print("=" * 70)
