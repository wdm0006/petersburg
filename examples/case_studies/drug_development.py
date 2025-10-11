"""
Drug Development Decision Analysis

This example models the pharmaceutical drug development process as a petersburg graph.
It demonstrates how to analyze a multi-billion dollar decision with sequential phases,
uncertain outcomes, and the potential for massive returns or total loss.
"""

import numpy as np

from petersburg import Graph

__author__ = "willmcginnis"


def analyze_drug_development():
    """
    Models drug development from pre-clinical through FDA approval.

    Phases:
    1. Pre-Clinical -> Phase I
    2. Phase I -> Phase II
    3. Phase II -> Phase III
    4. Phase III -> FDA Review
    5. FDA Review -> Market Launch

    At each phase there's a probability of success (continue) or failure (stop).
    """

    g = Graph()

    # Costs in millions of dollars
    preclinical_cost = 150
    phase1_cost = 10
    phase2_cost = 40
    phase3_cost = 225
    fda_cost = 15

    # Success probabilities
    preclinical_success = 0.70
    phase1_success = 0.70
    phase2_success = 0.33
    phase3_success = 0.27
    fda_success = 0.87

    # Payoffs (in millions)
    # Different market outcomes after approval
    blockbuster_revenue = 10000  # $10B over drug lifetime
    moderate_success = 2000  # $2B
    modest_success = 500  # $500M
    commercial_failure = 50  # $50M (approved but didn't sell)

    # Market outcome probabilities (given FDA approval)
    blockbuster_prob = 0.05
    moderate_prob = 0.15
    modest_prob = 0.50
    failure_prob = 0.30

    graph_dict = {
        # Single terminal node (required by petersburg)
        0: {"payoff": 0, "after": []},
        # Terminal failure nodes (all point to node 0)
        1: {
            "payoff": 0,
            "after": [{"node_id": 0, "cost": 0, "weight": 1.0}],
        },  # Failed pre-clinical
        2: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed Phase I
        3: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed Phase II
        4: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed Phase III
        5: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed FDA review
        # Market outcome nodes (all point to node 0)
        6: {
            "payoff": blockbuster_revenue,
            "after": [{"node_id": 0, "cost": 0, "weight": 1.0}],
        },  # Blockbuster
        7: {
            "payoff": moderate_success,
            "after": [{"node_id": 0, "cost": 0, "weight": 1.0}],
        },  # Moderate success
        8: {
            "payoff": modest_success,
            "after": [{"node_id": 0, "cost": 0, "weight": 1.0}],
        },  # Modest success
        9: {
            "payoff": commercial_failure,
            "after": [{"node_id": 0, "cost": 0, "weight": 1.0}],
        },  # Commercial failure
        # Market outcome decision (after FDA approval)
        10: {
            "payoff": 0,
            "after": [
                {"node_id": 6, "cost": 0, "weight": blockbuster_prob},
                {"node_id": 7, "cost": 0, "weight": moderate_prob},
                {"node_id": 8, "cost": 0, "weight": modest_prob},
                {"node_id": 9, "cost": 0, "weight": failure_prob},
            ],
        },
        # FDA Review outcomes
        11: {
            "payoff": 0,
            "after": [
                {"node_id": 10, "cost": 0, "weight": fda_success},  # FDA approval
                {"node_id": 5, "cost": 0, "weight": 1 - fda_success},  # FDA rejection
            ],
        },
        # Phase III outcomes
        12: {
            "payoff": 0,
            "after": [
                {"node_id": 11, "cost": fda_cost, "weight": phase3_success},  # Phase III success
                {"node_id": 4, "cost": 0, "weight": 1 - phase3_success},  # Phase III failure
            ],
        },
        # Phase II outcomes
        13: {
            "payoff": 0,
            "after": [
                {"node_id": 12, "cost": phase3_cost, "weight": phase2_success},  # Phase II success
                {"node_id": 3, "cost": 0, "weight": 1 - phase2_success},  # Phase II failure
            ],
        },
        # Phase I outcomes
        14: {
            "payoff": 0,
            "after": [
                {"node_id": 13, "cost": phase2_cost, "weight": phase1_success},  # Phase I success
                {"node_id": 2, "cost": 0, "weight": 1 - phase1_success},  # Phase I failure
            ],
        },
        # Pre-clinical outcomes
        15: {
            "payoff": 0,
            "after": [
                {
                    "node_id": 14,
                    "cost": phase1_cost,
                    "weight": preclinical_success,
                },  # Pre-clinical success
                {
                    "node_id": 1,
                    "cost": 0,
                    "weight": 1 - preclinical_success,
                },  # Pre-clinical failure
            ],
        },
        # Starting node (decision to start drug development)
        16: {
            "payoff": 0,
            "after": [
                {"node_id": 15, "cost": preclinical_cost, "weight": 1.0},
            ],
        },
    }

    g.from_dict(graph_dict)

    return g


def run_simulation(num_trials=100000):
    """
    Run Monte Carlo simulation of drug development outcomes.
    """
    print("=" * 80)
    print("DRUG DEVELOPMENT DECISION ANALYSIS")
    print("=" * 80)
    print()

    g = analyze_drug_development()

    # Simulate many drug development attempts
    outcomes = []
    for _ in range(num_trials):
        outcome = g.get_outcome()
        outcomes.append(outcome)

    outcomes = np.array(outcomes)

    # Calculate statistics
    print(f"Simulation Results ({num_trials:,} trials)")
    print("-" * 80)
    print()

    # Expected value
    expected_value = np.mean(outcomes)
    print(f"Expected Value (EV): ${expected_value:.2f}M")
    print()

    # Distribution of outcomes
    print("Outcome Distribution:")
    total_failures = np.sum(outcomes <= 0)
    failure_rate = (total_failures / num_trials) * 100
    print(f"  Total Failures (loss): {total_failures:,} ({failure_rate:.2f}%)")

    approvals = np.sum(outcomes > 0)
    approval_rate = (approvals / num_trials) * 100
    print(f"  FDA Approvals (profit): {approvals:,} ({approval_rate:.2f}%)")
    print()

    # Break down by success level
    blockbusters = np.sum(outcomes > 5000)
    moderate = np.sum((outcomes > 1000) & (outcomes <= 5000))
    modest = np.sum((outcomes > 100) & (outcomes <= 1000))
    commercial_fail = np.sum((outcomes > 0) & (outcomes <= 100))

    print("  Detailed Outcome Breakdown:")
    print(f"    Blockbuster (>$5B): {blockbusters:,} ({blockbusters/num_trials*100:.2f}%)")
    print(f"    Moderate ($1-5B): {moderate:,} ({moderate/num_trials*100:.2f}%)")
    print(f"    Modest ($100M-1B): {modest:,} ({modest/num_trials*100:.2f}%)")
    print(
        f"    Commercial Failure (<$100M): {commercial_fail:,} ({commercial_fail/num_trials*100:.2f}%)"
    )
    print()

    # Risk metrics
    print("Risk Metrics:")
    print(f"  Median Outcome: ${np.median(outcomes):.2f}M")
    print(f"  Best Case (99th percentile): ${np.percentile(outcomes, 99):.2f}M")
    print(f"  Worst Case: ${np.min(outcomes):.2f}M")
    print(f"  Standard Deviation: ${np.std(outcomes):.2f}M")
    print()

    # Investment analysis
    avg_cost = -np.mean(outcomes[outcomes <= 0])
    print("Investment Analysis:")
    print(f"  Average Cost of Failed Drug: ${avg_cost:.2f}M")
    print(f"  Probability of Positive Return: {approval_rate:.2f}%")

    # Calculate the probability-weighted success needed
    total_cost_estimate = 150 + 10 + 40 + 225 + 15  # Sum of all phase costs
    print(f"  Total Cost if All Phases Completed: ${total_cost_estimate:.2f}M")
    print()

    return outcomes


def inversion_analysis():
    """
    Inversion analysis: What would have to be true for this to be profitable?

    Given that the expected value is likely negative for a single drug,
    pharma companies need a portfolio strategy.
    """
    print("=" * 80)
    print("INVERSION ANALYSIS: What Would Have to Be True?")
    print("=" * 80)
    print()

    # Question: How many drugs do we need in our pipeline to expect 1 blockbuster?
    approval_rate = 0.70 * 0.70 * 0.33 * 0.27 * 0.87  # Combined probability
    blockbuster_rate = approval_rate * 0.05

    print(f"Overall Probability of Approval: {approval_rate*100:.2f}%")
    print(f"Overall Probability of Blockbuster: {blockbuster_rate*100:.4f}%")
    print()

    drugs_needed_for_approval = int(1 / approval_rate)
    drugs_needed_for_blockbuster = int(1 / blockbuster_rate)

    print(f"Drugs needed in pipeline to expect 1 approval: ~{drugs_needed_for_approval}")
    print(f"Drugs needed in pipeline to expect 1 blockbuster: ~{drugs_needed_for_blockbuster}")
    print()

    # Portfolio analysis
    print("Portfolio Strategy:")
    print(f"  If we start {drugs_needed_for_blockbuster} drugs in pre-clinical:")
    avg_cost_per_drug = (
        150
        + (10 * 0.70)
        + (40 * 0.70 * 0.70)
        + (225 * 0.70 * 0.70 * 0.33)
        + (15 * 0.70 * 0.70 * 0.33 * 0.27)
    )
    total_portfolio_cost = drugs_needed_for_blockbuster * avg_cost_per_drug
    print(f"  Expected Total Portfolio Cost: ${total_portfolio_cost:.2f}M")
    print("  Expected Revenue from 1 Blockbuster: $10,000M")
    print(f"  Expected Net Profit: ${10000 - total_portfolio_cost:.2f}M")
    print()

    # What if we could improve Phase II success rate?
    print("Sensitivity Analysis: Improving Phase II Success Rate")
    for improved_phase2 in [0.40, 0.50, 0.60]:
        improved_approval = 0.70 * 0.70 * improved_phase2 * 0.27 * 0.87
        improved_blockbuster = improved_approval * 0.05
        improved_drugs_needed = int(1 / improved_blockbuster)
        improved_cost = improved_drugs_needed * avg_cost_per_drug
        print(
            f"  Phase II @ {improved_phase2*100:.0f}%: Need {improved_drugs_needed} drugs, ${improved_cost:.2f}M cost"
        )
    print()


def sensitivity_analysis_phase2():
    """
    Perform sensitivity analysis: how does Phase II success rate affect outcomes?

    This is a key question for pharma companies investing in better trial design.
    """
    print("=" * 80)
    print("SENSITIVITY ANALYSIS: Phase II Success Rate")
    print("=" * 80)
    print()
    print("Question: How sensitive are outcomes to Phase II success rate?")
    print()

    # Test range of Phase II success rates
    phase2_rates = np.arange(0.20, 0.70, 0.05)
    results = {
        "phase2_rate": [],
        "expected_value": [],
        "approval_rate": [],
        "median_outcome": [],
        "positive_rate": [],
    }

    print("Running simulations across Phase II success rates...")
    print()

    for rate in phase2_rates:
        # Rebuild graph with modified Phase II success rate
        g = Graph()

        # Costs
        preclinical_cost = 150
        phase1_cost = 10
        phase2_cost = 40
        phase3_cost = 225
        fda_cost = 15

        # Probabilities (only Phase II varies)
        preclinical_success = 0.70
        phase1_success = 0.70
        phase2_success = rate  # VARYING THIS
        phase3_success = 0.27
        fda_success = 0.87

        # Payoffs
        blockbuster_revenue = 10000
        moderate_success = 2000
        modest_success = 500
        commercial_failure = 50

        blockbuster_prob = 0.05
        moderate_prob = 0.15
        modest_prob = 0.50
        failure_prob = 0.30

        graph_dict = {
            0: {"payoff": 0, "after": []},
            1: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            2: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            3: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            4: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            5: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            6: {"payoff": blockbuster_revenue, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            7: {"payoff": moderate_success, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            8: {"payoff": modest_success, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            9: {"payoff": commercial_failure, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            10: {
                "payoff": 0,
                "after": [
                    {"node_id": 6, "cost": 0, "weight": blockbuster_prob},
                    {"node_id": 7, "cost": 0, "weight": moderate_prob},
                    {"node_id": 8, "cost": 0, "weight": modest_prob},
                    {"node_id": 9, "cost": 0, "weight": failure_prob},
                ],
            },
            11: {
                "payoff": 0,
                "after": [
                    {"node_id": 10, "cost": 0, "weight": fda_success},
                    {"node_id": 5, "cost": 0, "weight": 1 - fda_success},
                ],
            },
            12: {
                "payoff": 0,
                "after": [
                    {"node_id": 11, "cost": fda_cost, "weight": phase3_success},
                    {"node_id": 4, "cost": 0, "weight": 1 - phase3_success},
                ],
            },
            13: {
                "payoff": 0,
                "after": [
                    {"node_id": 12, "cost": phase3_cost, "weight": phase2_success},
                    {"node_id": 3, "cost": 0, "weight": 1 - phase2_success},
                ],
            },
            14: {
                "payoff": 0,
                "after": [
                    {"node_id": 13, "cost": phase2_cost, "weight": phase1_success},
                    {"node_id": 2, "cost": 0, "weight": 1 - phase1_success},
                ],
            },
            15: {
                "payoff": 0,
                "after": [
                    {"node_id": 14, "cost": phase1_cost, "weight": preclinical_success},
                    {"node_id": 1, "cost": 0, "weight": 1 - preclinical_success},
                ],
            },
            16: {
                "payoff": 0,
                "after": [
                    {"node_id": 15, "cost": preclinical_cost, "weight": 1.0},
                ],
            },
        }

        g.from_dict(graph_dict)

        # Run simulation
        outcomes = []
        for _ in range(10000):
            outcomes.append(g.get_outcome())
        outcomes = np.array(outcomes)

        # Store results
        results["phase2_rate"].append(rate)
        results["expected_value"].append(np.mean(outcomes))
        results["approval_rate"].append(np.sum(outcomes > 0) / len(outcomes) * 100)
        results["median_outcome"].append(np.median(outcomes))
        results["positive_rate"].append(np.sum(outcomes > 0) / len(outcomes) * 100)

    # Print results table
    print("Results:")
    print("-" * 80)
    print(f"{'Phase II Rate':<15} {'Expected Value':<20} {'Approval Rate':<20} {'Median':<15}")
    print("-" * 80)

    for i in range(len(results["phase2_rate"])):
        print(
            f"{results['phase2_rate'][i]*100:>13.0f}%  "
            f"${results['expected_value'][i]:>17.2f}M  "
            f"{results['approval_rate'][i]:>18.2f}%  "
            f"${results['median_outcome'][i]:>13.2f}M"
        )

    print()
    print("Key Findings:")
    ev_improvement = ((results["expected_value"][-1] / results["expected_value"][0]) - 1) * 100
    print(f"  - Improving Phase II from 20% to 65% increases EV by {ev_improvement:.0f}%")

    baseline_ev = results["expected_value"][int(len(results["phase2_rate"]) * 0.26)]  # ~33%
    improved_ev = results["expected_value"][int(len(results["phase2_rate"]) * 0.46)]  # ~43%
    ev_gain = improved_ev - baseline_ev
    print(f"  - A 10-point improvement (33%→43%) adds ~${ev_gain:.0f}M in EV per drug")

    print()
    print("ASCII Chart: Expected Value vs Phase II Success Rate")
    print("-" * 80)

    # Simple ASCII chart
    max_ev = max(results["expected_value"])
    min_ev = min(results["expected_value"])
    chart_width = 60

    for i in range(len(results["phase2_rate"])):
        rate = results["phase2_rate"][i]
        ev = results["expected_value"][i]

        # Normalize to chart width
        if max_ev != min_ev:
            bar_length = int(((ev - min_ev) / (max_ev - min_ev)) * chart_width)
        else:
            bar_length = chart_width // 2

        bar = "█" * bar_length
        print(f"{rate*100:>5.0f}% | {bar} ${ev:.0f}M")

    print("-" * 80)
    print()


def automatic_sensitivity_analysis():
    """
    Use the built-in automatic sensitivity analysis to identify critical parameters.
    """
    print("=" * 80)
    print("AUTOMATIC PARAMETER SENSITIVITY DETECTION")
    print("=" * 80)
    print()
    print("Using petersburg's built-in sensitivity analysis to automatically")
    print("identify the most impactful parameters in the drug development graph...")
    print()

    g = analyze_drug_development()

    # Run automatic sensitivity analysis
    g.print_sensitivity_report(num_simulations=1000, perturbation=0.10, top_n=10)

    print("Interpretation:")
    print("  - Edge weights represent transition probabilities (e.g., Phase II success rate)")
    print("  - Costs represent phase expenses")
    print("  - Node payoffs represent market outcomes")
    print()
    print("This analysis automatically tests ±10% changes in each parameter and")
    print("ranks them by impact on expected value, saving manual sensitivity testing!")
    print()


if __name__ == "__main__":
    # Run the main simulation
    outcomes = run_simulation(num_trials=100000)

    print()

    # Perform inversion analysis
    inversion_analysis()

    print()

    # Sensitivity analysis
    sensitivity_analysis_phase2()

    # Automatic sensitivity analysis
    automatic_sensitivity_analysis()

    print()
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("1. Drug development is a classic 'negative expected value' decision for")
    print("   individual drugs, but becomes profitable with a portfolio strategy.")
    print()
    print("2. The key is having enough shots on goal - most pharma companies run")
    print("   dozens of drugs through the pipeline simultaneously.")
    print()
    print("3. The asymmetry is extreme: 95%+ probability of loss, but the winners")
    print("   can return 100x+ on investment.")
    print()
    print("4. Early-stage success rates (Phase I/II) are critical leverage points.")
    print("   Small improvements in Phase II success dramatically improve economics.")
    print()
    print("5. This is why pharma companies invest heavily in:")
    print("   - Better target selection (computational biology)")
    print("   - Biomarkers (better patient selection in Phase II)")
    print("   - Platform technologies (improve success rates across portfolio)")
    print()
    print("6. SENSITIVITY: Phase II success rate is highly elastic - each 10-point")
    print("   improvement can add hundreds of millions in expected value per drug.")
    print()
