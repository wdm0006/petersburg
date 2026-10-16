import unittest

from petersburg import Graph, Node, SpecValidationError, ValidationError


def _two_stage_spec():
    """Greedy first choice (node 2, payoff 8) loses to node 3, whose own stage-2 choice is node 6."""
    return {
        1: {"decision": True, "after": []},
        2: {"payoff": 8, "after": [{"node_id": 1}]},
        3: {"payoff": 2, "decision": True, "after": [{"node_id": 1, "cost": 1}]},
        4: {"payoff": 0, "after": [{"node_id": 2}]},
        5: {"payoff": -8, "after": [{"node_id": 2}]},
        6: {"payoff": 8, "after": [{"node_id": 3, "cost": 4}]},
        7: {"payoff": 0, "after": [{"node_id": 3}]},
        8: {"payoff": 8, "after": [{"node_id": 6}]},
        9: {"payoff": 0, "after": [{"node_id": 6}]},
    }


def _graph(spec=None):
    return Graph().from_dict(spec or _two_stage_spec())


class _Classifier:
    def predict_proba(self, features):
        return [[0.5, 0.5]]


class TestOptimalPolicy(unittest.TestCase):
    def test_value_and_policy_beat_greedy_first_choice(self):
        result = _graph().optimal_policy()
        self.assertEqual(result, {"value": 9.0, "policy": {1: 3, 3: 6}})

    def test_every_alternative_policy_is_no_better(self):
        g = _graph()
        best = g.optimal_policy()["value"]
        for first in (2, 3):
            for second in (6, 7):
                applied = g.apply_policy({1: first, 3: second})
                self.assertLessEqual(applied.expected_value(), best)
        self.assertEqual(g.apply_policy({1: 2, 3: 7}).expected_value(), 4.0)

    def test_applied_policy_matches_value_and_supports_exact_evaluators(self):
        g = _graph()
        applied = g.apply_policy(g.optimal_policy()["policy"])
        self.assertEqual(applied.expected_value(), 9.0)
        self.assertEqual(applied.variance(), 16.0)
        self.assertEqual(applied.outcome_distribution(), {5: 0.5, 13: 0.5})
        self.assertEqual(applied.expected_options(), {2: 4.0, 3: 9.0})
        applied.exact_sensitivity()
        applied.get_outcome()

    def test_apply_policy_returns_new_graph_and_leaves_original_unresolved(self):
        g = _graph()
        applied = g.apply_policy(g.optimal_policy()["policy"])
        self.assertIsNot(applied, g)
        self.assertTrue(g.start_node.is_decision)
        self.assertFalse(applied.start_node.is_decision)
        with self.assertRaises(ValidationError):
            g.expected_value()

    def test_policy_only_lists_reachable_decision_nodes(self):
        g = _graph(
            {
                1: {"after": []},
                2: {"decision": True, "after": [{"node_id": 1, "weight": 1}]},
                3: {"decision": True, "after": [{"node_id": 1, "weight": 0}]},
                4: {"payoff": 1, "after": [{"node_id": 2}]},
                5: {"payoff": 2, "after": [{"node_id": 2}]},
                6: {"payoff": 7, "after": [{"node_id": 3}]},
            }
        )
        result = g.optimal_policy()
        self.assertEqual(result, {"value": 2.0, "policy": {2: 5, 3: 6}})

    def test_terminal_decision_node_has_no_choice(self):
        g = _graph({1: {"decision": True, "payoff": 3, "after": []}})
        self.assertEqual(g.optimal_policy(), {"value": 3.0, "policy": {}})
        self.assertEqual(g.apply_policy({}).expected_value(), 3.0)

    def test_parallel_edges_are_keyed_by_occurrence(self):
        def spec(first_cost, second_cost):
            return {
                1: {"decision": True, "after": []},
                2: {
                    "payoff": 100,
                    "after": [
                        {"node_id": 1, "cost": first_cost},
                        {"node_id": 1, "cost": second_cost},
                    ],
                },
            }

        cheap_first = _graph(spec(10, 80)).optimal_policy()
        self.assertEqual(cheap_first, {"value": 90.0, "policy": {1: 2}})
        cheap_second = _graph(spec(80, 10))
        result = cheap_second.optimal_policy()
        self.assertEqual(result, {"value": 90.0, "policy": {1: (2, 1)}})
        self.assertEqual(cheap_second.apply_policy(result["policy"]).expected_value(), 90.0)
        self.assertEqual(cheap_second.apply_policy({1: 2}).expected_value(), 20.0)

    def test_ties_break_to_earliest_outcome(self):
        g = _graph(
            {
                1: {"decision": True, "after": []},
                2: {"payoff": 5, "after": [{"node_id": 1}]},
                3: {"payoff": 5, "after": [{"node_id": 1}]},
                4: {"payoff": 5, "after": [{"node_id": 1}]},
            }
        )
        self.assertEqual(g.optimal_policy()["policy"], {1: 2})

    def test_decision_edge_weights_are_ignored(self):
        spec = _two_stage_spec()
        spec[2]["after"][0]["weight"] = 1000
        spec[3]["after"][0]["weight"] = 0
        self.assertEqual(_graph(spec).optimal_policy()["policy"], {1: 3, 3: 6})

    def test_classifier_weights_rejected(self):
        g = _graph()
        with self.assertRaises(ValidationError):
            g.start_node.add_outcome(Node(99), classifier=_Classifier())
        chance = _graph({1: {"after": []}, 2: {"after": [{"node_id": 1}]}})
        chance.start_node.add_outcome(Node(3), classifier=_Classifier())
        with self.assertRaises(ValidationError):
            chance.optimal_policy()
        chance.start_node.is_decision = True
        with self.assertRaises(ValidationError):
            chance.optimal_policy()

    def test_unbuilt_graph_rejected(self):
        with self.assertRaises(ValidationError):
            Graph().optimal_policy()
        with self.assertRaises(ValidationError):
            Graph().apply_policy({})


class TestApplyPolicyValidation(unittest.TestCase):
    def test_rejects_bad_policies(self):
        g = _graph()
        for policy in (
            {1: 3},
            {1: 3, 3: 6, 2: 4},
            {1: 3, 3: 2},
            {1: 9, 3: 6},
            [],
        ):
            with self.subTest(policy=policy), self.assertRaises(ValidationError):
                g.apply_policy(policy)


class TestUnresolvedDecisionNodes(unittest.TestCase):
    def test_flag_is_stored_on_nodes(self):
        g = _graph()
        decisions = {n.node_id for n in g.node_list() if n.is_decision}
        self.assertEqual(decisions, {1, 3})

    def test_evaluators_and_simulators_reject_unresolved_graph(self):
        g = _graph()
        calls = {
            "expected_value": g.expected_value,
            "expected_options": g.expected_options,
            "get_outcome": g.get_outcome,
            "get_outcome_node": g.get_outcome_node,
            "get_options": g.get_options,
            "variance": g.variance,
            "std": g.std,
            "outcome_distribution": g.outcome_distribution,
            "option_distributions": g.option_distributions,
            "exact_sensitivity": g.exact_sensitivity,
            "analyze_sensitivity": g.analyze_sensitivity,
            "identify_critical_parameters": g.identify_critical_parameters,
        }
        for name, call in calls.items():
            with self.subTest(name), self.assertRaisesRegex(ValidationError, "optimal_policy"):
                call()

    def test_graph_without_decisions_is_unaffected(self):
        g = _graph({1: {"after": []}, 2: {"payoff": 4, "after": [{"node_id": 1}]}})
        self.assertEqual(g.expected_value(), 4.0)
        self.assertEqual(g.optimal_policy(), {"value": 4.0, "policy": {}})


class TestDecisionSpecValidation(unittest.TestCase):
    def test_non_bool_decision_rejected_and_model_preserved(self):
        g = _graph({1: {"after": []}})
        original = g.start_node
        for value in (1, "yes", None, 0.0):
            spec = {1: {"decision": value, "after": []}}
            with self.subTest(value), self.assertRaises(SpecValidationError):
                g.from_dict(spec)
        self.assertIs(g.start_node, original)


class TestDecisionSerialization(unittest.TestCase):
    def test_dict_and_json_round_trip(self):
        g = _graph()
        spec = g.to_dict()
        self.assertIs(spec[1]["decision"], True)
        self.assertIs(spec[3]["decision"], True)
        self.assertNotIn("decision", spec[2])
        self.assertEqual(Graph().from_dict(spec).to_dict(), spec)
        rebuilt = Graph().from_json(g.to_json())
        self.assertEqual(rebuilt.to_dict(), spec)
        self.assertEqual(rebuilt.optimal_policy(), g.optimal_policy())

    def test_graph_without_decisions_serializes_without_the_key(self):
        g = _graph({1: {"after": []}, 2: {"payoff": 1, "after": [{"node_id": 1}]}})
        self.assertEqual(
            g.to_dict(),
            {
                1: {"type": "fixed", "payoff": 0, "after": []},
                2: {
                    "type": "fixed",
                    "payoff": 1,
                    "after": [{"node_id": 1, "cost": 0, "weight": 1}],
                },
            },
        )
        self.assertNotIn("decision", g.to_json())


if __name__ == "__main__":
    unittest.main()
