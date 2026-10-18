"""Sequential continue-or-abandon walkthrough: optimal policy vs always-continue.

A project first reveals whether conditions are favorable (3 in 4) or unfavorable (1 in 4).
The state is observed BEFORE the decision, so the choice may differ per state: abandon for a
salvage payoff of 10, or continue by paying 20 for a chance at a payoff of 100 (success
3 in 4 when favorable, 1 in 4 when unfavorable). Net payoff is payoffs minus costs. Every
number is exact and identical on every run. Needs only the `examples` extra.
"""

from petersburg import Graph

FAVORABLE, UNFAVORABLE = 2, 3
FAVORABLE_ABANDON, FAVORABLE_CONTINUE = 4, 5
UNFAVORABLE_ABANDON, UNFAVORABLE_CONTINUE = 8, 9


def build_project_graph():
    """Observed state (nodes 2, 3), then a decision node in each state; unresolved."""
    return Graph().from_dict(
        {
            1: {"after": []},
            FAVORABLE: {"decision": True, "after": [{"node_id": 1, "weight": 3}]},
            UNFAVORABLE: {"decision": True, "after": [{"node_id": 1, "weight": 1}]},
            FAVORABLE_ABANDON: {"payoff": 10, "after": [{"node_id": FAVORABLE}]},
            FAVORABLE_CONTINUE: {"after": [{"node_id": FAVORABLE, "cost": 20}]},
            6: {"payoff": 100, "after": [{"node_id": FAVORABLE_CONTINUE, "weight": 3}]},
            7: {"after": [{"node_id": FAVORABLE_CONTINUE, "weight": 1}]},
            UNFAVORABLE_ABANDON: {"payoff": 10, "after": [{"node_id": UNFAVORABLE}]},
            UNFAVORABLE_CONTINUE: {"after": [{"node_id": UNFAVORABLE, "cost": 20}]},
            10: {"payoff": 100, "after": [{"node_id": UNFAVORABLE_CONTINUE, "weight": 1}]},
            11: {"after": [{"node_id": UNFAVORABLE_CONTINUE, "weight": 3}]},
        }
    )


def always_continue_policy():
    """A complete hand-written policy: continue in both states."""
    return {FAVORABLE: FAVORABLE_CONTINUE, UNFAVORABLE: UNFAVORABLE_CONTINUE}


def evidence(graph, policy):
    """Exact evidence for ``graph`` played under ``policy`` (unresolved graph untouched)."""
    resolved = graph.apply_policy(policy)
    distribution = resolved.outcome_distribution()
    return {
        "policy": policy,
        "expected_value": resolved.expected_value(),
        "distribution": distribution,
        "p_loss": sum(p for outcome, p in distribution.items() if outcome < 0),
    }


def analyze(graph):
    """Optimal policy and always-continue, each with exact EV, PMF and P(loss)."""
    best = graph.optimal_policy()
    return {
        "optimal_value": best["value"],
        "optimal": evidence(graph, best["policy"]),
        "always_continue": evidence(graph, always_continue_policy()),
    }


def main():
    facts = analyze(build_project_graph())
    print(f"optimal_policy() value: {facts['optimal_value']:.2f}")
    for name in ("optimal", "always_continue"):
        row = facts[name]
        print(f"\n{name}: choices {row['policy']}")
        print(f"  expected net payoff: {row['expected_value']:.4f}")
        print(f"  outcome PMF: {dict(sorted(row['distribution'].items()))}")
        print(f"  P(loss): {row['p_loss']:.4f}")
    print(
        "\nThe state is observed before each decision, so the optimal policy continues when\n"
        "favorable (EV 55 > 10) and abandons when unfavorable (continuing nets 5 < 10). The\n"
        "objective is expected net payoff, not minimum loss probability; here the optimal\n"
        "policy also loses less often, but that is a property of these numbers, not a goal.\n"
        "This is not EVPI and not a risk-optimal policy: the state is revealed, not future draws."
    )


if __name__ == "__main__":
    main()
