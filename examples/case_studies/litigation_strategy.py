"""
Litigation Strategy Decision Analysis

This example models the decision to pursue litigation through trial or settle
at various stages. It demonstrates settlement vs. trial analysis and the value
of information gained during discovery.
"""

import numpy as np

from petersburg import Graph

__author__ = "willmcginnis"


def build_litigation_graph(case_strength="moderate"):
    """
    Models litigation from filing through trial and potential appeal.

    Stages:
    1. Pre-Filing Investigation
    2. Filing & Pleadings
    3. Discovery
    4. Pre-Trial
    5. Trial
    6. Appeal (if applicable)

    case_strength: 'strong', 'moderate', or 'weak'
    """

    g = Graph()

    # Costs in thousands
    prefiling_cost = 50
    filing_cost = 100
    discovery_cost = 800
    pretrial_cost = 300
    trial_cost = 1500
    appeal_cost = 400

    # Success probabilities based on case strength
    if case_strength == "strong":
        proceed_to_file = 0.80
        survive_dismissal = 0.90
        reach_trial = 0.40  # 60% settle
        win_at_trial = 0.70
        survive_appeal = 0.85
        settlement_multiplier = 0.60  # Settlements are 60% of expected verdict
    elif case_strength == "weak":
        proceed_to_file = 0.40
        survive_dismissal = 0.50
        reach_trial = 0.20  # 80% settle or dismissed
        win_at_trial = 0.30
        survive_appeal = 0.70
        settlement_multiplier = 0.30  # Settlements are only 30% of expected verdict
    else:  # moderate
        proceed_to_file = 0.60
        survive_dismissal = 0.70
        reach_trial = 0.30  # 70% settle
        win_at_trial = 0.50
        survive_appeal = 0.80
        settlement_multiplier = 0.45

    # Outcomes (in thousands)
    # These represent damages awarded or settlement amounts
    large_verdict = 15000  # $15M jury verdict
    moderate_verdict = 8000  # $8M verdict
    small_verdict = 3000  # $3M verdict

    # Settlement outcomes (occur during discovery/pre-trial)
    large_settlement = large_verdict * settlement_multiplier
    moderate_settlement = moderate_verdict * settlement_multiplier
    small_settlement = small_verdict * settlement_multiplier

    # Verdict distribution (if case goes to trial)
    large_verdict_prob = 0.20
    moderate_verdict_prob = 0.50
    small_verdict_prob = 0.30

    graph_dict = {
        # Single terminal node (required by petersburg)
        0: {"payoff": 0, "after": []},
        # Terminal failure nodes (all point to node 0)
        1: {
            "payoff": 0,
            "after": [{"node_id": 0, "cost": 0, "weight": 1.0}],
        },  # Decided not to file
        2: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Dismissed
        3: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Lost at trial
        4: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Reversed on appeal
        # Settlement outcome nodes (during discovery/pre-trial, all point to node 0)
        5: {"payoff": large_settlement, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
        6: {"payoff": moderate_settlement, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
        7: {"payoff": small_settlement, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
        # Trial verdict nodes (if win at trial, all point to node 0)
        8: {"payoff": large_verdict, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
        9: {"payoff": moderate_verdict, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
        10: {"payoff": small_verdict, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
        # Verdict distribution (if win at trial and survive appeal)
        11: {
            "payoff": 0,
            "after": [
                {"node_id": 8, "cost": 0, "weight": large_verdict_prob},
                {"node_id": 9, "cost": 0, "weight": moderate_verdict_prob},
                {"node_id": 10, "cost": 0, "weight": small_verdict_prob},
            ],
        },
        # Appeal outcomes (after trial win)
        12: {
            "payoff": 0,
            "after": [
                {"node_id": 11, "cost": 0, "weight": survive_appeal},  # Win upheld
                {"node_id": 4, "cost": 0, "weight": 1 - survive_appeal},  # Reversed
            ],
        },
        # Trial outcomes
        13: {
            "payoff": 0,
            "after": [
                {"node_id": 12, "cost": appeal_cost, "weight": win_at_trial},  # Win (then appeal)
                {"node_id": 3, "cost": 0, "weight": 1 - win_at_trial},  # Lose
            ],
        },
        # Settlement distribution (if settle during discovery)
        14: {
            "payoff": 0,
            "after": [
                {"node_id": 5, "cost": 0, "weight": large_verdict_prob},
                {"node_id": 6, "cost": 0, "weight": moderate_verdict_prob},
                {"node_id": 7, "cost": 0, "weight": small_verdict_prob},
            ],
        },
        # Pre-trial decision: Settle or go to trial
        15: {
            "payoff": 0,
            "after": [
                {"node_id": 13, "cost": trial_cost, "weight": reach_trial},  # Go to trial
                {"node_id": 14, "cost": 0, "weight": 1 - reach_trial},  # Settle
            ],
        },
        # Discovery phase
        16: {
            "payoff": 0,
            "after": [
                {"node_id": 15, "cost": pretrial_cost, "weight": 1.0},
            ],
        },
        # Filing and pleadings (dismissal risk)
        17: {
            "payoff": 0,
            "after": [
                {"node_id": 16, "cost": discovery_cost, "weight": survive_dismissal},
                {"node_id": 2, "cost": 0, "weight": 1 - survive_dismissal},
            ],
        },
        # Pre-filing decision
        18: {
            "payoff": 0,
            "after": [
                {"node_id": 17, "cost": filing_cost, "weight": proceed_to_file},
                {"node_id": 1, "cost": 0, "weight": 1 - proceed_to_file},
            ],
        },
        # Starting node
        19: {
            "payoff": 0,
            "after": [
                {"node_id": 18, "cost": prefiling_cost, "weight": 1.0},
            ],
        },
    }

    g.from_dict(graph_dict)

    return g


def run_litigation_simulation(case_strength="moderate", num_trials=100000):
    """
    Simulate litigation outcomes.
    """
    strength_names = {"strong": "Strong Case", "moderate": "Moderate Case", "weak": "Weak Case"}

    print("=" * 80)
    print(f"LITIGATION STRATEGY ANALYSIS - {strength_names[case_strength]}")
    print("=" * 80)
    print()

    g = build_litigation_graph(case_strength)

    # Simulate many cases
    outcomes = []
    for _ in range(num_trials):
        outcome = g.get_outcome()
        outcomes.append(outcome)

    outcomes = np.array(outcomes)

    print(f"Simulation Results ({num_trials:,} cases)")
    print("-" * 80)
    print()

    # Expected value (net recovery after costs)
    expected_value = np.mean(outcomes)
    print(f"Expected Net Recovery: ${expected_value:.0f}K (${expected_value/1000:.2f}M)")
    print()

    # Outcome distribution
    print("Outcome Distribution:")
    losses = np.sum(outcomes <= 0)
    loss_rate = (losses / num_trials) * 100
    print(f"  Net Losses (no recovery or small recovery): {losses:,} ({loss_rate:.2f}%)")

    recoveries = np.sum(outcomes > 0)
    recovery_rate = (recoveries / num_trials) * 100
    print(f"  Positive Net Recovery: {recoveries:,} ({recovery_rate:.2f}%)")
    print()

    # Break down by outcome size
    if recoveries > 0:
        large_recovery = np.sum(outcomes > 5000)
        moderate_recovery = np.sum((outcomes > 2000) & (outcomes <= 5000))
        small_recovery = np.sum((outcomes > 0) & (outcomes <= 2000))

        print("  Recovery Size Breakdown:")
        print(f"    Large (>$5M): {large_recovery:,} ({large_recovery/num_trials*100:.2f}%)")
        print(
            f"    Moderate ($2-5M): {moderate_recovery:,} ({moderate_recovery/num_trials*100:.2f}%)"
        )
        print(f"    Small (<$2M): {small_recovery:,} ({small_recovery/num_trials*100:.2f}%)")
        print()

    # Settlement vs. Trial breakdown
    # Settlements are typically in the range of 2-9M (60-5400K)
    # Trial verdicts are higher (3-15M = 3000-15000K)
    settlements = np.sum((outcomes > 1500) & (outcomes < 10000))
    trials = np.sum(outcomes >= 10000)

    print("  Settlement vs. Trial Outcomes:")
    print(f"    Settled cases: {settlements:,} ({settlements/num_trials*100:.2f}%)")
    print(f"    Trial verdicts: {trials:,} ({trials/num_trials*100:.2f}%)")
    print()

    # Risk metrics
    print("Risk Metrics:")
    print(f"  Median Outcome: ${np.median(outcomes):.0f}K")
    print(f"  25th Percentile: ${np.percentile(outcomes, 25):.0f}K")
    print(f"  75th Percentile: ${np.percentile(outcomes, 75):.0f}K")
    print(f"  90th Percentile: ${np.percentile(outcomes, 90):.0f}K")
    print(f"  Best Case (99th): ${np.percentile(outcomes, 99):.0f}K")
    print()

    # Investment analysis
    print("Cost Analysis:")
    avg_loss = np.mean(outcomes[outcomes < 0])
    print(f"  Average Loss (when lose): ${avg_loss:.0f}K")
    print(f"  Probability of Positive Recovery: {recovery_rate:.2f}%")
    print()

    return outcomes


def settlement_decision_analysis():
    """
    Analyze whether to accept settlement offers at different stages.
    """
    print("=" * 80)
    print("SETTLEMENT DECISION ANALYSIS")
    print("=" * 80)
    print()

    print("Scenario: You are offered a settlement. Should you accept?")
    print()

    # Build a moderate case
    g = build_litigation_graph("moderate")

    # Simulate to get expected value of continuing
    outcomes = []
    for _ in range(50000):
        outcomes.append(g.get_outcome())
    ev_continue = np.mean(outcomes)

    print(f"Expected Value of Continuing to Trial: ${ev_continue:.0f}K")
    print()

    # Settlement offers at different stages
    offers = [
        ("Early (after filing)", 2000, 50 + 100),  # Offer, costs already sunk
        ("Mid-discovery", 3500, 50 + 100 + 400),
        ("Pre-trial", 4500, 50 + 100 + 800 + 150),
    ]

    print("Settlement Offers at Different Stages:")
    print("-" * 80)

    for stage, offer, sunk_cost in offers:
        # Remaining costs to get to trial
        remaining_cost = (50 + 100 + 800 + 300 + 1500 + 400) - sunk_cost

        # Expected net value of continuing
        ev_net = ev_continue - sunk_cost

        # Settlement net value
        settlement_net = offer - sunk_cost

        # Compare
        print(f"\n{stage}:")
        print(f"  Settlement Offer: ${offer:.0f}K")
        print(f"  Costs Already Sunk: ${sunk_cost:.0f}K")
        print(f"  Remaining Costs to Trial: ${remaining_cost:.0f}K")
        print(f"  Expected Value (continue): ${ev_net:.0f}K")
        print(f"  Settlement Value (net): ${settlement_net:.0f}K")

        if settlement_net > ev_net:
            print("  DECISION: ACCEPT settlement")
            print(f"  Gain from settling: ${settlement_net - ev_net:.0f}K")
        else:
            print("  DECISION: REJECT settlement, continue litigation")
            print(f"  Expected gain from continuing: ${ev_net - settlement_net:.0f}K")

    print()


def compare_case_strengths():
    """
    Compare outcomes across different case strengths.
    """
    print("=" * 80)
    print("CASE STRENGTH COMPARISON")
    print("=" * 80)
    print()

    strengths = ["weak", "moderate", "strong"]
    results = {}

    for strength in strengths:
        g = build_litigation_graph(strength)
        outcomes = []
        for _ in range(50000):
            outcomes.append(g.get_outcome())
        outcomes = np.array(outcomes)

        results[strength] = {
            "ev": np.mean(outcomes),
            "median": np.median(outcomes),
            "positive_rate": np.sum(outcomes > 0) / len(outcomes) * 100,
            "p75": np.percentile(outcomes, 75),
        }

    print("Comparison by Case Strength:")
    print("-" * 80)
    print(f"{'Case Strength':<15} {'EV':>15} {'Median':>15} {'Positive %':>15} {'75th %ile':>15}")
    print("-" * 80)

    for strength in strengths:
        r = results[strength]
        print(
            f"{strength.capitalize():<15} "
            f"${r['ev']:>13.0f}K "
            f"${r['median']:>13.0f}K "
            f"{r['positive_rate']:>14.1f}% "
            f"${r['p75']:>13.0f}K"
        )

    print()
    print("Key Insight: Case strength dramatically affects expected value.")
    print(
        f"Strong cases have {results['strong']['ev'] / results['weak']['ev']:.1f}x "
        f"better EV than weak cases."
    )
    print()


def defendant_analysis():
    """
    Analyze from defendant's perspective (inverse problem).
    """
    print("=" * 80)
    print("DEFENDANT'S SETTLEMENT ANALYSIS")
    print("=" * 80)
    print()

    print("From Defendant's Perspective:")
    print()

    # Defendant faces inverse probabilities
    # If plaintiff has 50% win rate, defendant has 50% loss rate
    plaintiff_win_prob = 0.50
    expected_verdict = 8000  # $8M average verdict if plaintiff wins
    defendant_trial_cost = 2000  # $2M to defend through trial

    expected_loss = (plaintiff_win_prob * expected_verdict) + defendant_trial_cost

    print(f"  Plaintiff's Win Probability: {plaintiff_win_prob*100:.0f}%")
    print(
        f"  Expected Verdict (if plaintiff wins): ${expected_verdict:.0f}K (${expected_verdict/1000:.1f}M)"
    )
    print(
        f"  Defendant's Trial Costs: ${defendant_trial_cost:.0f}K (${defendant_trial_cost/1000:.1f}M)"
    )
    print(f"  Expected Total Loss: ${expected_loss:.0f}K (${expected_loss/1000:.1f}M)")
    print()

    # Settlement analysis
    print("Settlement Decision:")
    settlement_offers = [3000, 4000, 5000, 6000]

    for offer in settlement_offers:
        savings = expected_loss - offer
        print(f"  Offer ${offer:.0f}K: ", end="")
        if savings > 0:
            print(f"ACCEPT (saves ${savings:.0f}K vs. expected loss)")
        else:
            print(f"REJECT (costs ${-savings:.0f}K more than expected loss)")

    print()
    print(f"Defendant's Maximum Settlement: ${expected_loss:.0f}K")
    print()

    # Settlement range
    plaintiff_minimum = 3500  # Plaintiff won't accept less (given their costs and EV)
    defendant_maximum = expected_loss

    print("Settlement Range:")
    print(f"  Plaintiff's Minimum: ${plaintiff_minimum:.0f}K")
    print(f"  Defendant's Maximum: ${defendant_maximum:.0f}K")

    if defendant_maximum > plaintiff_minimum:
        print(f"  Settlement Zone: ${plaintiff_minimum:.0f}K - ${defendant_maximum:.0f}K")
        print(f"  Zone Width: ${defendant_maximum - plaintiff_minimum:.0f}K")
        print("  Prediction: Case will likely settle in this range")
    else:
        print("  No settlement zone - parties' valuations don't overlap")
        print("  Prediction: Case will go to trial")

    print()


if __name__ == "__main__":
    # Run simulations for different case strengths
    outcomes_weak = run_litigation_simulation("weak", num_trials=100000)

    print()

    outcomes_moderate = run_litigation_simulation("moderate", num_trials=100000)

    print()

    outcomes_strong = run_litigation_simulation("strong", num_trials=100000)

    print()

    # Compare case strengths
    compare_case_strengths()

    print()

    # Settlement decision analysis
    settlement_decision_analysis()

    print()

    # Defendant perspective
    defendant_analysis()

    print()
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("1. Most cases settle (70-80%) because there's usually a settlement")
    print("   range where both parties prefer certainty over trial risk.")
    print()
    print("2. Case strength is crucial: weak cases should be dropped early")
    print("   or settled cheaply. Strong cases justify going to trial.")
    print()
    print("3. Discovery costs are substantial: often $500K-$2M. This creates")
    print("   pressure to settle before incurring these costs.")
    print()
    print("4. Settlement timing matters: early settlements save costs, but")
    print("   discovery provides information that can justify higher settlements.")
    print()
    print("5. The 'settlement range' exists when defendant's expected loss")
    print("   exceeds plaintiff's minimum acceptable settlement.")
    print()
    print("6. Risk aversion favors settlement: a certain $4M is often preferred")
    print("   to a 50% chance of $8M, even though EV is the same.")
    print()
    print("7. Inversion is powerful: 'What would have to be true for trial")
    print("   to be worth it?' often reveals trial is not justified.")
    print()
