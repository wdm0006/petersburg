"""Exact value of perfect information from one joint scenario/payoff table.

Two equally likely scenarios decide what each action pays (action A pays 30 / 0, action B
pays 0 / 20). ``petersburg.value_of_information`` builds two graphs from that SAME table and
solves them exactly with ``optimal_policy()``:

* no information: the action is chosen BEFORE the scenario is resolved, so one action
  must serve every scenario;
* perfect information: chance resolves the scenario first and each scenario has its own
  decision node, so the choice may depend on the observed scenario.

Gross value is the difference (EVPI). The information cost is subtracted once, outside the
graphs, independent of which action is selected, to give the net value.

Limitation: perfect information here reveals the entire joint scenario before acting. It is
an upper bound on what research can be worth, not a model of a noisy study or continuous
payoff uncertainty; pass ``observation=`` to model a study that reveals only a group of
scenarios. Needs only the `examples` extra.
"""

from petersburg import value_of_information

DEFAULT_INFORMATION_COST = 5
ACTIONS = ["A", "B"]

SCENARIOS = {
    "scenario 1": {"probability": 0.5, "payoffs": {"A": 30, "B": 0}},
    "scenario 2": {"probability": 0.5, "payoffs": {"A": 0, "B": 20}},
}
# Same shape, but A pays at least as much as B in every scenario.
DOMINATED_SCENARIOS = {
    "scenario 1": {"probability": 0.5, "payoffs": {"A": 30, "B": 10}},
    "scenario 2": {"probability": 0.5, "payoffs": {"A": 20, "B": 0}},
}


def compare_information(
    scenarios=SCENARIOS, information_cost=DEFAULT_INFORMATION_COST, observation=None
):
    """Exact no-information vs (perfect, by default) information comparison."""
    return value_of_information(scenarios, ACTIONS, observation, information_cost)


def main():
    result = compare_information()
    print("No information (act before the scenario resolves):")
    print(f"  expected value: {result['value_without']:.2f}")
    print(f"  choice: {result['policy_without']}")
    print("Perfect information (act after observing the scenario):")
    print(f"  expected value: {result['value_with']:.2f}")
    print(f"  choices: {result['policy_with']}")
    print(f"\nGross EVPI: {result['gross_value']:.2f}")
    print(f"Information cost (charged once): {DEFAULT_INFORMATION_COST:.2f}")
    print(f"Net improvement: {result['net_value']:.2f}")
    print(
        "\nPerfect information reveals the entire joint scenario before acting, so gross EVPI is\n"
        "an upper bound on the value of research. It is not a model of a noisy study or\n"
        "continuous payoff uncertainty. Net improvement is gross EVPI minus the\n"
        "information cost, which does not depend on the action chosen."
    )


if __name__ == "__main__":
    main()
