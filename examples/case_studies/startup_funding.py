"""
Startup Funding Journey Analysis

This example models the venture capital funding process from pre-seed to exit.
It demonstrates the power law dynamics of startup outcomes and helps answer
questions about portfolio construction and optimal decision points.
"""

import numpy as np

from petersburg import Graph

__author__ = "willmcginnis"


def build_startup_journey_graph():
    """
    Models a startup's journey through funding rounds to exit.

    Stages:
    1. Pre-Seed -> Seed
    2. Seed -> Series A
    3. Series A -> Series B
    4. Series B -> Series C
    5. Series C -> Exit

    At each stage, the startup can:
    - Succeed and raise the next round
    - Fail and shut down
    """

    g = Graph()

    # Costs (capital raised at each round, in millions)
    preseed_raise = 0.3
    seed_raise = 1.5
    seriesa_raise = 10
    seriesb_raise = 30
    seriesc_raise = 75

    # Success probabilities (stage transition rates)
    preseed_to_seed = 0.40
    seed_to_seriesa = 0.30
    seriesa_to_seriesb = 0.40
    seriesb_to_seriesc = 0.50
    seriesc_to_exit = 0.60

    # Exit outcomes (in millions)
    # These represent the RETURN to early investors after dilution
    decacorn_exit = 5000  # $10B+ exit, early investor 10x return
    unicorn_exit = 1000  # $1-10B exit, early investor 10x return
    successful_exit = 200  # $200M-1B exit, good return
    modest_exit = 30  # $50-200M exit, modest return
    acquihire = 5  # Soft landing, minimal return

    # Exit distribution (given Series C completion)
    decacorn_prob = 0.05
    unicorn_prob = 0.15
    successful_prob = 0.30
    modest_prob = 0.35
    acquihire_prob = 0.15

    graph_dict = {
        # Single terminal node (required by petersburg)
        0: {"payoff": 0, "after": []},
        # Terminal failure nodes (all point to node 0)
        1: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed pre-seed
        2: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed seed
        3: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed Series A
        4: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed Series B
        5: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Failed Series C
        # Exit outcome nodes (all point to node 0)
        6: {
            "payoff": decacorn_exit,
            "after": [{"node_id": 0, "cost": 0, "weight": 1.0}],
        },  # Decacorn
        7: {"payoff": unicorn_exit, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},  # Unicorn
        8: {
            "payoff": successful_exit,
            "after": [{"node_id": 0, "cost": 0, "weight": 1.0}],
        },  # Successful exit
        9: {
            "payoff": modest_exit,
            "after": [{"node_id": 0, "cost": 0, "weight": 1.0}],
        },  # Modest exit
        10: {
            "payoff": acquihire,
            "after": [{"node_id": 0, "cost": 0, "weight": 1.0}],
        },  # Acqui-hire
        # Exit distribution node (after Series C)
        11: {
            "payoff": 0,
            "after": [
                {"node_id": 6, "cost": 0, "weight": decacorn_prob},
                {"node_id": 7, "cost": 0, "weight": unicorn_prob},
                {"node_id": 8, "cost": 0, "weight": successful_prob},
                {"node_id": 9, "cost": 0, "weight": modest_prob},
                {"node_id": 10, "cost": 0, "weight": acquihire_prob},
            ],
        },
        # Series C outcomes
        12: {
            "payoff": 0,
            "after": [
                {"node_id": 11, "cost": 0, "weight": seriesc_to_exit},
                {"node_id": 5, "cost": 0, "weight": 1 - seriesc_to_exit},
            ],
        },
        # Series B outcomes
        13: {
            "payoff": 0,
            "after": [
                {"node_id": 12, "cost": seriesc_raise, "weight": seriesb_to_seriesc},
                {"node_id": 4, "cost": 0, "weight": 1 - seriesb_to_seriesc},
            ],
        },
        # Series A outcomes
        14: {
            "payoff": 0,
            "after": [
                {"node_id": 13, "cost": seriesb_raise, "weight": seriesa_to_seriesb},
                {"node_id": 3, "cost": 0, "weight": 1 - seriesa_to_seriesb},
            ],
        },
        # Seed outcomes
        15: {
            "payoff": 0,
            "after": [
                {"node_id": 14, "cost": seriesa_raise, "weight": seed_to_seriesa},
                {"node_id": 2, "cost": 0, "weight": 1 - seed_to_seriesa},
            ],
        },
        # Pre-seed outcomes
        16: {
            "payoff": 0,
            "after": [
                {"node_id": 15, "cost": seed_raise, "weight": preseed_to_seed},
                {"node_id": 1, "cost": 0, "weight": 1 - preseed_to_seed},
            ],
        },
        # Starting node (decision to start startup)
        17: {
            "payoff": 0,
            "after": [
                {"node_id": 16, "cost": preseed_raise, "weight": 1.0},
            ],
        },
    }

    g.from_dict(graph_dict)

    return g


def run_startup_simulation(num_trials=100000):
    """
    Simulate the startup funding journey.
    """
    print("=" * 80)
    print("STARTUP FUNDING JOURNEY ANALYSIS")
    print("=" * 80)
    print()

    g = build_startup_journey_graph()

    # Simulate many startup journeys
    outcomes = []
    for _ in range(num_trials):
        outcome = g.get_outcome()
        outcomes.append(outcome)

    outcomes = np.array(outcomes)

    print(f"Simulation Results ({num_trials:,} startups)")
    print("-" * 80)
    print()

    # Expected value from investor perspective
    expected_value = np.mean(outcomes)
    print(f"Expected Value per Investment: ${expected_value:.2f}M")
    print()

    # Outcome distribution
    print("Outcome Distribution:")
    failures = np.sum(outcomes <= 0)
    failure_rate = (failures / num_trials) * 100
    print(f"  Total Failures: {failures:,} ({failure_rate:.2f}%)")

    exits = np.sum(outcomes > 0)
    exit_rate = (exits / num_trials) * 100
    print(f"  Successful Exits: {exits:,} ({exit_rate:.2f}%)")
    print()

    # Break down by exit type
    decacorns = np.sum(outcomes > 3000)
    unicorns = np.sum((outcomes > 500) & (outcomes <= 3000))
    successful = np.sum((outcomes > 100) & (outcomes <= 500))
    modest = np.sum((outcomes > 10) & (outcomes <= 100))
    acquihires = np.sum((outcomes > 0) & (outcomes <= 10))

    print("  Exit Type Breakdown:")
    print(f"    Decacorn (>$3B return): {decacorns:,} ({decacorns/num_trials*100:.3f}%)")
    print(f"    Unicorn ($500M-3B): {unicorns:,} ({unicorns/num_trials*100:.3f}%)")
    print(f"    Successful ($100-500M): {successful:,} ({successful/num_trials*100:.2f}%)")
    print(f"    Modest ($10-100M): {modest:,} ({modest/num_trials*100:.2f}%)")
    print(f"    Acqui-hire (<$10M): {acquihires:,} ({acquihires/num_trials*100:.2f}%)")
    print()

    # Risk metrics
    print("Risk Metrics:")
    print(f"  Median Outcome: ${np.median(outcomes):.2f}M")
    print(f"  75th Percentile: ${np.percentile(outcomes, 75):.2f}M")
    print(f"  90th Percentile: ${np.percentile(outcomes, 90):.2f}M")
    print(f"  99th Percentile: ${np.percentile(outcomes, 99):.2f}M")
    print(f"  Standard Deviation: ${np.std(outcomes):.2f}M")
    print()

    # Power law demonstration
    top_10_pct = np.percentile(outcomes, 90)
    top_10_pct_value = np.sum(outcomes[outcomes >= top_10_pct])
    total_value = np.sum(outcomes[outcomes > 0])
    top_10_contribution = (top_10_pct_value / total_value * 100) if total_value > 0 else 0

    print("Power Law Dynamics:")
    print(f"  Top 10% of exits contribute {top_10_contribution:.1f}% of total value")
    print("  This demonstrates the classic VC power law")
    print()

    return outcomes


def portfolio_analysis():
    """
    Analyze VC portfolio construction strategy.
    """
    print("=" * 80)
    print("VENTURE CAPITAL PORTFOLIO ANALYSIS")
    print("=" * 80)
    print()

    # Calculate probabilities
    preseed_to_seed = 0.40
    seed_to_seriesa = 0.30
    seriesa_to_seriesb = 0.40
    seriesb_to_seriesc = 0.50
    seriesc_to_exit = 0.60

    overall_exit_prob = (
        preseed_to_seed
        * seed_to_seriesa
        * seriesa_to_seriesb
        * seriesb_to_seriesc
        * seriesc_to_exit
    )

    unicorn_plus_prob = overall_exit_prob * 0.20  # Unicorn or better

    print("Key Probabilities:")
    print(f"  Any successful exit: {overall_exit_prob*100:.2f}%")
    print(f"  Unicorn+ exit: {unicorn_plus_prob*100:.4f}%")
    print()

    # Portfolio strategy
    print("Portfolio Strategy for a $100M VC Fund:")
    print()

    fund_size = 100  # $100M
    target_multiple = 3  # 3x return = $300M
    avg_check_size = 10  # $10M per company

    portfolio_size = int(fund_size / avg_check_size)

    print(f"  Fund Size: ${fund_size}M")
    print(f"  Target Return: {target_multiple}x (${fund_size * target_multiple}M)")
    print(f"  Average Check Size: ${avg_check_size}M")
    print(f"  Portfolio Size: {portfolio_size} companies")
    print()

    # Expected outcomes
    expected_exits = portfolio_size * overall_exit_prob
    expected_unicorns = portfolio_size * unicorn_plus_prob

    print(f"  Expected Successful Exits: {expected_exits:.1f}")
    print(f"  Expected Unicorn+ Exits: {expected_unicorns:.2f}")
    print()

    # What do we need to return the fund?
    print("  Return Scenarios:")

    # Scenario 1: One big winner
    big_winner_return = 1000  # $1B return on $10M investment
    remainder_needed = (fund_size * target_multiple) - big_winner_return
    print(f"    If we get 1 unicorn returning ${big_winner_return}M:")
    print(f"      Still need ${remainder_needed}M from other {portfolio_size-1} investments")
    print(f"      Avg return needed per other company: ${remainder_needed/(portfolio_size-1):.1f}M")
    print()

    # Scenario 2: Power law
    print("    Typical VC Power Law Distribution:")
    print("      Top 1 company: Returns 100x ($1,000M)")
    print("      Top 2-3 companies: Return 10x each ($100M each)")
    print("      Companies 4-7: Return 3x each ($30M each)")
    print("      Companies 8-10: Return 1x (break even)")
    print("      Total: $1,320M on $100M (13.2x fund)")
    print()


def inversion_analysis():
    """
    Inversion: What would have to be true for different scenarios?
    """
    print("=" * 80)
    print("INVERSION ANALYSIS")
    print("=" * 80)
    print()

    print("Question: What needs to be true for a startup to achieve a $1B exit?")
    print()

    # Work backwards from $1B valuation
    print("Working Backwards from $1B Exit:")
    print("  At Series C (exit time):")
    print("    - Annual Revenue: $100M+")
    print("    - Growth Rate: 50%+ YoY")
    print("    - Gross Margins: 70%+")
    print("    - Market Position: Top 3 in category")
    print()

    print("  At Series B (18 months prior):")
    print("    - Annual Revenue: $40M")
    print("    - Growth Rate: 100%+ YoY")
    print("    - Clear path to $100M revenue")
    print("    - Proven unit economics")
    print()

    print("  At Series A (36 months prior):")
    print("    - Annual Revenue: $5M")
    print("    - Growth Rate: 200%+ YoY")
    print("    - Strong product-market fit")
    print("    - LTV/CAC ratio > 3")
    print()

    print("  At Seed (48 months prior):")
    print("    - Monthly Revenue: $100K")
    print("    - Consistent month-over-month growth")
    print("    - 100+ paying customers")
    print("    - TAM > $1B")
    print()

    print()
    print("Question: How does improving early-stage success rates impact outcomes?")
    print()

    # Sensitivity analysis
    base_case_prob = 0.40 * 0.30 * 0.40 * 0.50 * 0.60
    print(f"  Base Case (overall exit probability): {base_case_prob*100:.2f}%")
    print()

    # Improve seed -> Series A (the "Valley of Death")
    improved_prob = 0.40 * 0.45 * 0.40 * 0.50 * 0.60  # Improve from 30% to 45%
    improvement = ((improved_prob / base_case_prob) - 1) * 100
    print("  If Seed→Series A improves from 30% to 45%:")
    print(f"    New overall probability: {improved_prob*100:.2f}%")
    print(f"    Improvement: +{improvement:.1f}%")
    print("    Insight: Better product-market fit validation at seed stage")
    print()

    # Improve all stages by 10%
    all_improved = 0.50 * 0.40 * 0.50 * 0.60 * 0.70
    improvement = ((all_improved / base_case_prob) - 1) * 100
    print("  If all stages improve by 10 percentage points:")
    print(f"    New overall probability: {all_improved*100:.2f}%")
    print(f"    Improvement: +{improvement:.1f}%")
    print("    Insight: Compound improvements have exponential impact")
    print()


def sensitivity_analysis_seed_to_seriesa():
    """
    Perform sensitivity analysis: how does Seed→Series A success rate affect outcomes?

    This is the "Valley of Death" - the most critical transition point.
    """
    print("=" * 80)
    print("SENSITIVITY ANALYSIS: Seed → Series A Success Rate")
    print("=" * 80)
    print()
    print("Question: How sensitive are outcomes to the 'Valley of Death'?")
    print()

    # Test range of Seed → Series A success rates
    seed_to_a_rates = np.arange(0.20, 0.55, 0.05)
    results = {"seed_to_a_rate": [], "expected_value": [], "exit_rate": [], "unicorn_rate": []}

    print("Running simulations across Seed→Series A success rates...")
    print()

    for rate in seed_to_a_rates:
        # Costs
        preseed_raise = 0.3
        seed_raise = 1.5
        seriesa_raise = 10
        seriesb_raise = 30
        seriesc_raise = 75

        # Success probabilities
        preseed_to_seed = 0.40
        seed_to_seriesa = rate  # VARYING THIS
        seriesa_to_seriesb = 0.40
        seriesb_to_seriesc = 0.50
        seriesc_to_exit = 0.60

        # Exit outcomes
        decacorn_exit = 5000
        unicorn_exit = 1000
        successful_exit = 200
        modest_exit = 30
        acquihire = 5

        decacorn_prob = 0.05
        unicorn_prob = 0.15
        successful_prob = 0.30
        modest_prob = 0.35
        acquihire_prob = 0.15

        g = Graph()
        graph_dict = {
            0: {"payoff": 0, "after": []},
            1: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            2: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            3: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            4: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            5: {"payoff": 0, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            6: {"payoff": decacorn_exit, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            7: {"payoff": unicorn_exit, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            8: {"payoff": successful_exit, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            9: {"payoff": modest_exit, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            10: {"payoff": acquihire, "after": [{"node_id": 0, "cost": 0, "weight": 1.0}]},
            11: {
                "payoff": 0,
                "after": [
                    {"node_id": 6, "cost": 0, "weight": decacorn_prob},
                    {"node_id": 7, "cost": 0, "weight": unicorn_prob},
                    {"node_id": 8, "cost": 0, "weight": successful_prob},
                    {"node_id": 9, "cost": 0, "weight": modest_prob},
                    {"node_id": 10, "cost": 0, "weight": acquihire_prob},
                ],
            },
            12: {
                "payoff": 0,
                "after": [
                    {"node_id": 11, "cost": 0, "weight": seriesc_to_exit},
                    {"node_id": 5, "cost": 0, "weight": 1 - seriesc_to_exit},
                ],
            },
            13: {
                "payoff": 0,
                "after": [
                    {"node_id": 12, "cost": seriesc_raise, "weight": seriesb_to_seriesc},
                    {"node_id": 4, "cost": 0, "weight": 1 - seriesb_to_seriesc},
                ],
            },
            14: {
                "payoff": 0,
                "after": [
                    {"node_id": 13, "cost": seriesb_raise, "weight": seriesa_to_seriesb},
                    {"node_id": 3, "cost": 0, "weight": 1 - seriesa_to_seriesb},
                ],
            },
            15: {
                "payoff": 0,
                "after": [
                    {"node_id": 14, "cost": seriesa_raise, "weight": seed_to_seriesa},
                    {"node_id": 2, "cost": 0, "weight": 1 - seed_to_seriesa},
                ],
            },
            16: {
                "payoff": 0,
                "after": [
                    {"node_id": 15, "cost": seed_raise, "weight": preseed_to_seed},
                    {"node_id": 1, "cost": 0, "weight": 1 - preseed_to_seed},
                ],
            },
            17: {
                "payoff": 0,
                "after": [
                    {"node_id": 16, "cost": preseed_raise, "weight": 1.0},
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
        results["seed_to_a_rate"].append(rate)
        results["expected_value"].append(np.mean(outcomes))
        results["exit_rate"].append(np.sum(outcomes > 0) / len(outcomes) * 100)
        unicorns = np.sum(outcomes > 500)
        results["unicorn_rate"].append(unicorns / len(outcomes) * 100)

    # Print results table
    print("Results:")
    print("-" * 80)
    print(f"{'Seed→A Rate':<15} {'Expected Value':<20} {'Exit Rate':<20} {'Unicorn+ Rate':<15}")
    print("-" * 80)

    for i in range(len(results["seed_to_a_rate"])):
        print(
            f"{results['seed_to_a_rate'][i]*100:>12.0f}%  "
            f"${results['expected_value'][i]:>17.2f}M  "
            f"{results['exit_rate'][i]:>18.2f}%  "
            f"{results['unicorn_rate'][i]:>13.2f}%"
        )

    print()
    print("Key Findings:")
    ev_improvement = ((results["expected_value"][-1] / results["expected_value"][0]) - 1) * 100
    print(f"  - Improving Seed→A from 20% to 50% increases EV by {ev_improvement:.0f}%")

    baseline_ev = results["expected_value"][2]  # 30%
    improved_ev = results["expected_value"][4]  # 40%
    ev_gain = improved_ev - baseline_ev
    print(f"  - A 10-point improvement (30%→40%) adds ~${ev_gain:.0f}M in EV per startup")
    print("  - This is the 'Valley of Death' - most critical transition point")

    print()
    print("ASCII Chart: Expected Value vs Seed→Series A Success Rate")
    print("-" * 80)

    # Simple ASCII chart
    max_ev = max(results["expected_value"])
    min_ev = min(results["expected_value"])
    chart_width = 60

    for i in range(len(results["seed_to_a_rate"])):
        rate = results["seed_to_a_rate"][i]
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


if __name__ == "__main__":
    # Run simulation
    outcomes = run_startup_simulation(num_trials=100000)

    print()

    # Portfolio analysis
    portfolio_analysis()

    print()

    # Inversion analysis
    inversion_analysis()

    # Sensitivity analysis
    sensitivity_analysis_seed_to_seriesa()

    print()
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("1. Startup outcomes follow an extreme power law: 90%+ fail, but")
    print("   the top 1% can return 100-1000x.")
    print()
    print("2. VC portfolio strategy is essential: need 20-30 bets to capture")
    print("   the 1-2 big winners that return the entire fund.")
    print()
    print("3. The 'Valley of Death' between Seed and Series A is the highest")
    print("   mortality point - improving success here has huge impact.")
    print()
    print("4. Each funding stage is both validation and commitment: passing")
    print("   one stage means you get to attempt the next, harder challenge.")
    print()
    print("5. Early-stage improvements compound: a 10% improvement in each")
    print("   stage doubles your overall probability of success.")
    print()
    print("6. Inversion is powerful: work backward from desired exit to")
    print("   determine required milestones at each stage.")
    print()
    print("7. SENSITIVITY: The Seed→Series A transition is highly elastic.")
    print("   Improving this 'Valley of Death' has exponential impact on outcomes.")
    print()
