"""Exact value of perfect information from one joint scenario/payoff table.

Two equally likely scenarios decide what each action pays (action A pays 30 / 0, action B
pays 0 / 20). Two graphs are built from that SAME table and solved exactly with
``optimal_policy()``, ``apply_policy()`` and ``expected_value()``:

* no information: the action is chosen BEFORE the scenario is resolved, so one action
  must serve every scenario;
* perfect information: chance resolves the scenario first and each scenario has its own
  decision node, so the choice may depend on the observed scenario.

Gross EVPI is the difference. The information cost is subtracted once, outside the
graphs, independent of which action is selected, to give the net improvement.

Limitation: perfect information here reveals the entire joint scenario before acting. It is
an upper bound on what research can be worth, not a model of a noisy study, partial
information or continuous payoff uncertainty. Needs only the `examples` extra.
"""

import math
from numbers import Real

from petersburg import Graph
from petersburg.exceptions import ValidationError

DEFAULT_INFORMATION_COST = 5

# Joint table: scenario -> (probability, {action: payoff}).
SCENARIOS = {
    "scenario 1": (0.5, {"A": 30, "B": 0}),
    "scenario 2": (0.5, {"A": 0, "B": 20}),
}
# Same shape, but A pays at least as much as B in every scenario.
DOMINATED_SCENARIOS = {
    "scenario 1": (0.5, {"A": 30, "B": 10}),
    "scenario 2": (0.5, {"A": 20, "B": 0}),
}


def _actions(scenarios):
    if not scenarios:
        raise ValidationError("scenarios must contain at least one scenario")
    first = next(iter(scenarios.values()))[1]
    actions = list(first)
    for name, (probability, payoffs) in scenarios.items():
        if set(payoffs) != set(actions):
            raise ValidationError(f"{name} must define payoffs for exactly the actions {actions}")
        if not (isinstance(probability, Real) and math.isfinite(probability) and probability > 0):
            raise ValidationError(f"{name} probability must be positive and finite")
    return actions


def build_no_information_graph(scenarios):
    """Decision node first; each action then faces the unresolved scenario chance."""
    actions = _actions(scenarios)
    spec = {"start": {"decision": True, "after": []}}
    for action in actions:
        spec[f"act {action}"] = {"after": [{"node_id": "start"}]}
        for name, (probability, payoffs) in scenarios.items():
            spec[f"{action} | {name}"] = {
                "payoff": payoffs[action],
                "after": [{"node_id": f"act {action}", "weight": probability}],
            }
    return Graph().from_dict(spec)


def build_perfect_information_graph(scenarios):
    """Chance resolves the scenario first; each scenario has its own decision node."""
    actions = _actions(scenarios)
    spec = {"start": {"after": []}}
    for name, (probability, payoffs) in scenarios.items():
        spec[f"observe {name}"] = {
            "decision": True,
            "after": [{"node_id": "start", "weight": probability}],
        }
        for action in actions:
            spec[f"{action} | {name}"] = {
                "payoff": payoffs[action],
                "after": [{"node_id": f"observe {name}"}],
            }
    return Graph().from_dict(spec)


def _validate_cost(information_cost):
    if (
        isinstance(information_cost, bool)
        or not isinstance(information_cost, Real)
        or not math.isfinite(information_cost)
        or information_cost < 0
    ):
        raise ValidationError(
            f"information_cost must be a nonnegative finite number, got {information_cost!r}"
        )


def _solve(graph):
    best = graph.optimal_policy()
    resolved = graph.apply_policy(best["policy"])
    return {
        "optimal_value": best["value"],
        "policy": best["policy"],
        "expected_value": resolved.expected_value(),
    }


def compare_information(scenarios=SCENARIOS, information_cost=DEFAULT_INFORMATION_COST):
    """Exact no-information vs perfect-information comparison for one joint table."""
    _validate_cost(information_cost)
    without = _solve(build_no_information_graph(scenarios))
    with_info = _solve(build_perfect_information_graph(scenarios))
    gross = with_info["expected_value"] - without["expected_value"]
    return {
        "no_information": without,
        "perfect_information": with_info,
        "information_cost": information_cost,
        "gross_evpi": gross,
        "net_improvement": gross - information_cost,
    }


def main():
    result = compare_information()
    for key, label in (
        ("no_information", "No information (act before the scenario resolves)"),
        ("perfect_information", "Perfect information (act after observing the scenario)"),
    ):
        row = result[key]
        print(f"{label}:")
        print(f"  expected value: {row['expected_value']:.2f}")
        print(f"  choices: {row['policy']}")
    print(f"\nGross EVPI: {result['gross_evpi']:.2f}")
    print(f"Information cost (charged once): {result['information_cost']:.2f}")
    print(f"Net improvement: {result['net_improvement']:.2f}")
    print(
        "\nPerfect information reveals the entire joint scenario before acting, so gross EVPI is\n"
        "an upper bound on the value of research. It is not a model of a noisy study, partial\n"
        "information or continuous payoff uncertainty. Net improvement is gross EVPI minus the\n"
        "information cost, which does not depend on the action chosen."
    )


if __name__ == "__main__":
    main()
