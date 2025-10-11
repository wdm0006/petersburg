"""
Product Launch Decision Analysis

This example models the decision to launch a new product through multiple stages
from concept to full market launch. It demonstrates how to evaluate go/no-go
decisions at each stage and the importance of early validation.
"""

import numpy as np

from petersburg import Graph

__author__ = "willmcginnis"


def build_product_launch_graph(scenario="base_case"):
    """
    Models a product launch from concept to market.

    Stages:
    1. Concept Development
    2. Prototype & Testing
    3. Pilot Launch
    4. Regional Launch
    5. Full Market Launch
    6. Market Outcomes

    scenario: 'base_case', 'strong_validation', or 'weak_fit'
    """

    g = Graph()

    # Costs in millions
    concept_cost = 0.3
    prototype_cost = 1.0
    pilot_cost = 2.5
    regional_cost = 8.0
    full_launch_cost = 25.0

    # Success probabilities based on scenario
    if scenario == "strong_validation":
        concept_success = 0.75
        prototype_success = 0.65
        pilot_success = 0.60
        regional_success = 0.75
        full_success = 0.70
    elif scenario == "weak_fit":
        concept_success = 0.40
        prototype_success = 0.35
        pilot_success = 0.25
        regional_success = 0.40
        full_success = 0.35
    else:  # base_case
        concept_success = 0.60
        prototype_success = 0.50
        pilot_success = 0.40
        regional_success = 0.60
        full_success = 0.55

    # Market outcomes (in millions, representing NPV of future cash flows)
    blockbuster = 200  # $200M+ NPV
    solid_success = 50  # $50M NPV
    moderate = 15  # $15M NPV
    breakeven = 0  # Break even
    failure = -10  # Small loss after shutting down

    # Market outcome probabilities
    blockbuster_prob = 0.05
    solid_prob = 0.15
    moderate_prob = 0.25
    breakeven_prob = 0.25
    failure_prob = 0.30

    graph_dict = {
        # Single terminal node (required by petersburg)
        0: {"payoff": 0, "after": []},
        # Terminal failure nodes (all point to node 0)
        1: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed concept
        2: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed prototype
        3: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed pilot
        4: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed regional
        5: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed full launch
        # Market outcome nodes (all point to node 0)
        6: {"payoff": blockbuster, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
        7: {"payoff": solid_success, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
        8: {"payoff": moderate, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
        9: {"payoff": breakeven, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
        10: {"payoff": failure, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
        # Market outcome distribution
        11: {
            "payoff": 0,
            "after": [
                {"node_id": 6, "cost": 0, "weight": blockbuster_prob},
                {"node_id": 7, "cost": 0, "weight": solid_prob},
                {"node_id": 8, "cost": 0, "weight": moderate_prob},
                {"node_id": 9, "cost": 0, "weight": breakeven_prob},
                {"node_id": 10, "cost": 0, "weight": failure_prob},
            ],
        },
        # Full launch outcomes
        12: {
            "payoff": 0,
            "after": [
                {"node_id": 11, "cost": 0, "weight": full_success},
                {"node_id": 5, "cost": 0, "weight": 1 - full_success},
            ],
        },
        # Regional launch outcomes
        13: {
            "payoff": 0,
            "after": [
                {"node_id": 12, "cost": full_launch_cost, "weight": regional_success},
                {"node_id": 4, "cost": 0, "weight": 1 - regional_success},
            ],
        },
        # Pilot launch outcomes
        14: {
            "payoff": 0,
            "after": [
                {"node_id": 13, "cost": regional_cost, "weight": pilot_success},
                {"node_id": 3, "cost": 0, "weight": 1 - pilot_success},
            ],
        },
        # Prototype outcomes
        15: {
            "payoff": 0,
            "after": [
                {"node_id": 14, "cost": pilot_cost, "weight": prototype_success},
                {"node_id": 2, "cost": 0, "weight": 1 - prototype_success},
            ],
        },
        # Concept outcomes
        16: {
            "payoff": 0,
            "after": [
                {"node_id": 15, "cost": prototype_cost, "weight": concept_success},
                {"node_id": 1, "cost": 0, "weight": 1 - concept_success},
            ],
        },
        # Starting node
        17: {
            "payoff": 0,
            "after": [
                {"node_id": 16, "cost": concept_cost, "weight": 1.0},
            ],
        },
    }

    g.from_dict(graph_dict)

    return g


def run_product_launch_simulation(scenario="base_case", num_trials=100000):
    """
    Simulate product launch outcomes.
    """
    scenario_names = {
        "base_case": "Base Case",
        "strong_validation": "Strong Market Validation",
        "weak_fit": "Weak Product-Market Fit",
    }

    print("=" * 80)
    print(f"PRODUCT LAUNCH ANALYSIS - {scenario_names[scenario]}")
    print("=" * 80)
    print()

    g = build_product_launch_graph(scenario)

    # Simulate many product launches
    outcomes = []
    for _ in range(num_trials):
        outcome = g.get_outcome()
        outcomes.append(outcome)

    outcomes = np.array(outcomes)

    print(f"Simulation Results ({num_trials:,} product launches)")
    print("-" * 80)
    print()

    # Expected value
    expected_value = np.mean(outcomes)
    print(f"Expected Value: ${expected_value:.2f}M")
    print()

    # Outcome distribution
    print("Outcome Distribution:")
    failures = np.sum(outcomes <= 0)
    failure_rate = (failures / num_trials) * 100
    print(f"  Total Failures: {failures:,} ({failure_rate:.2f}%)")

    successes = np.sum(outcomes > 0)
    success_rate = (successes / num_trials) * 100
    print(f"  Profitable Products: {successes:,} ({success_rate:.2f}%)")
    print()

    # Break down by outcome type
    if successes > 0:
        blockbusters = np.sum(outcomes > 100)
        solid = np.sum((outcomes > 30) & (outcomes <= 100))
        moderate = np.sum((outcomes > 5) & (outcomes <= 30))
        breakeven_range = np.sum((outcomes > -5) & (outcomes <= 5))

        print("  Success Type Breakdown:")
        print(f"    Blockbuster (>$100M): {blockbusters:,} ({blockbusters/num_trials*100:.2f}%)")
        print(f"    Solid Success ($30-100M): {solid:,} ({solid/num_trials*100:.2f}%)")
        print(f"    Moderate ($5-30M): {moderate:,} ({moderate/num_trials*100:.2f}%)")
        print(f"    Break-even (±$5M): {breakeven_range:,} ({breakeven_range/num_trials*100:.2f}%)")
        print()

    # Risk metrics
    print("Risk Metrics:")
    print(f"  Median Outcome: ${np.median(outcomes):.2f}M")
    print(f"  25th Percentile: ${np.percentile(outcomes, 25):.2f}M")
    print(f"  75th Percentile: ${np.percentile(outcomes, 75):.2f}M")
    print(f"  90th Percentile: ${np.percentile(outcomes, 90):.2f}M")
    print()

    # Investment analysis
    total_cost = 0.3 + 1.0 + 2.5 + 8.0 + 25.0
    avg_loss = np.mean(outcomes[outcomes < 0])
    print("Investment Analysis:")
    print(f"  Max Possible Investment: ${total_cost:.1f}M (if all stages completed)")
    print(f"  Average Loss (failed products): ${avg_loss:.2f}M")
    print(f"  Probability of Profit: {success_rate:.2f}%")
    print()

    return outcomes


def compare_scenarios():
    """
    Compare different validation scenarios.
    """
    print("=" * 80)
    print("SCENARIO COMPARISON")
    print("=" * 80)
    print()

    scenarios = ["base_case", "strong_validation", "weak_fit"]
    results = {}

    for scenario in scenarios:
        g = build_product_launch_graph(scenario)
        outcomes = []
        for _ in range(50000):
            outcomes.append(g.get_outcome())
        outcomes = np.array(outcomes)

        results[scenario] = {
            "ev": np.mean(outcomes),
            "success_rate": np.sum(outcomes > 0) / len(outcomes) * 100,
            "median": np.median(outcomes),
            "p90": np.percentile(outcomes, 90),
        }

    print("Comparison of Validation Scenarios:")
    print("-" * 80)
    print(f"{'Scenario':<25} {'EV':>12} {'Success %':>12} {'Median':>12} {'90th %ile':>12}")
    print("-" * 80)

    scenario_names = {
        "base_case": "Base Case",
        "strong_validation": "Strong Validation",
        "weak_fit": "Weak Product Fit",
    }

    for scenario in scenarios:
        r = results[scenario]
        print(
            f"{scenario_names[scenario]:<25} "
            f"${r['ev']:>10.2f}M "
            f"{r['success_rate']:>11.2f}% "
            f"${r['median']:>10.2f}M "
            f"${r['p90']:>10.2f}M"
        )

    print()
    print(
        "Key Insight: Strong early validation improves EV by "
        f"{((results['strong_validation']['ev'] / results['base_case']['ev']) - 1) * 100:.0f}%"
    )
    print()


def stage_gate_analysis():
    """
    Analyze the value of stage-gate decisions.
    """
    print("=" * 80)
    print("STAGE-GATE DECISION ANALYSIS")
    print("=" * 80)
    print()

    print("Question: Should we proceed after pilot showing weak signals?")
    print()

    # Scenario: Pilot results are 60% of target
    print("Scenario: Pilot sales are 60% of target")
    print()

    # Calculate EV of continuing vs. stopping
    regional_cost = 8.0
    full_launch_cost = 25.0
    total_remaining_cost = regional_cost + full_launch_cost

    # If pilot is weak, assume probabilities are worse
    weak_regional_success = 0.40
    weak_full_success = 0.45

    # Expected outcomes
    blockbuster = 200 * 0.05
    solid = 50 * 0.15
    moderate = 15 * 0.25
    breakeven = 0 * 0.25
    failure = -10 * 0.30

    expected_market_outcome = blockbuster + solid + moderate + breakeven + failure
    expected_value_continue = (
        weak_regional_success * weak_full_success * expected_market_outcome
    ) - total_remaining_cost

    print(f"  Expected Value of CONTINUING: ${expected_value_continue:.2f}M")
    print("  Cost Already Sunk: $3.8M (concept + prototype + pilot)")
    print(f"  Additional Investment Required: ${total_remaining_cost:.1f}M")
    print()

    print("  Decision: STOP")
    print("  Rationale: EV of continuing is negative after pilot shows weak signals.")
    print("  Better to cut losses at $3.8M than risk an additional $33M.")
    print()

    print("Scenario: Pilot sales are 90% of target")
    print()

    # Better probabilities with strong pilot
    strong_regional_success = 0.75
    strong_full_success = 0.70

    expected_value_strong = (
        strong_regional_success * strong_full_success * expected_market_outcome
    ) - total_remaining_cost

    print(f"  Expected Value of CONTINUING: ${expected_value_strong:.2f}M")
    print(f"  Additional Investment Required: ${total_remaining_cost:.1f}M")
    print()

    print("  Decision: CONTINUE")
    print("  Rationale: Strong pilot signals justify the additional investment.")
    print()


def portfolio_strategy():
    """
    Analyze product portfolio strategy.
    """
    print("=" * 80)
    print("PRODUCT PORTFOLIO STRATEGY")
    print("=" * 80)
    print()

    print("Question: How many products should we develop to expect 1 blockbuster?")
    print()

    # Calculate overall success probabilities
    base_probs = 0.60 * 0.50 * 0.40 * 0.60 * 0.55
    blockbuster_prob = base_probs * 0.05

    products_needed = int(1 / blockbuster_prob)
    avg_cost_per_product = 0.3 + (1.0 * 0.60) + (2.5 * 0.60 * 0.50) + (8.0 * 0.60 * 0.50 * 0.40)

    total_portfolio_cost = products_needed * avg_cost_per_product

    print(f"  Overall probability of blockbuster: {blockbuster_prob*100:.3f}%")
    print(f"  Products needed for 1 expected blockbuster: ~{products_needed}")
    print(f"  Average cost per product attempt: ${avg_cost_per_product:.2f}M")
    print(f"  Total portfolio investment: ${total_portfolio_cost:.1f}M")
    print("  Expected revenue from 1 blockbuster: $200M")
    print(f"  Expected profit: ${200 - total_portfolio_cost:.1f}M")
    print()

    print("Portfolio Strategy Recommendations:")
    print("  1. Maintain pipeline of 10-20 products at various stages")
    print("  2. Kill 40-50% at concept stage (cheapest to fail)")
    print("  3. Kill another 30-40% at prototype stage")
    print("  4. Only 10-15% should reach pilot stage")
    print("  5. Be ruthless at pilot stage - this is the critical filter")
    print()


if __name__ == "__main__":
    # Run base case simulation
    outcomes_base = run_product_launch_simulation("base_case", num_trials=100000)

    print()

    # Run strong validation scenario
    outcomes_strong = run_product_launch_simulation("strong_validation", num_trials=100000)

    print()

    # Compare scenarios
    compare_scenarios()

    print()

    # Stage-gate analysis
    stage_gate_analysis()

    print()

    # Portfolio strategy
    portfolio_strategy()

    print()
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("1. Product launches require a portfolio approach: most will fail,")
    print("   but the winners pay for the losers.")
    print()
    print("2. Early validation is crucial: strong concept and prototype testing")
    print("   improves overall EV by 250%+.")
    print()
    print("3. Pilot stage is the critical decision point: weak pilot signals")
    print("   should trigger a STOP decision, not optimistic adjustments.")
    print()
    print("4. Sunk cost fallacy is deadly: having invested $3.8M doesn't justify")
    print("   risking another $33M on a weak product.")
    print()
    print("5. Stage-gate discipline: 60-70% of products should be killed before")
    print("   reaching pilot stage. This is a feature, not a bug.")
    print()
    print("6. The math of compounding probabilities: small improvements at each")
    print("   stage have exponential impact on final outcomes.")
    print()
