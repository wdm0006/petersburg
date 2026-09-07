"""
Tests for the petersburg exception hierarchy: every public validation path raises a
PetersburgError subclass, and the historical built-in contracts (spec problems caught
as AttributeError, other validation caught as ValueError) keep working unchanged.
"""

import unittest

import numpy as np

from petersburg import (
    FrequencyEstimator,
    Graph,
    MixedModeEstimator,
    Node,
    PetersburgError,
    SpecValidationError,
    ValidationError,
)

__author__ = "willmcginnis"


def _two_node_spec():
    return {
        1: {"payoff": 5, "after": []},
        2: {"payoff": 10, "after": [{"node_id": 1, "cost": 1}]},
    }


def _estimator_data():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([[0, 1], [1, 2], [0, 1], [1, 2], [0, 1], [1, 2]])
    return X, y


def _fitted_frequency_estimator(**kwargs):
    X, y = _estimator_data()
    est = FrequencyEstimator(**kwargs)
    est.fit(X, y)
    return est


def _fitted_mixed_mode_estimator():
    X, y = _estimator_data()
    est = MixedModeEstimator()
    est.fit(X, y)
    return est


class TestExceptionHierarchy(unittest.TestCase):
    """Structure of the hierarchy itself."""

    def test_validation_error_subclasses_petersburg_error_and_value_error(self):
        self.assertTrue(issubclass(ValidationError, PetersburgError))
        self.assertTrue(issubclass(ValidationError, ValueError))

    def test_spec_validation_error_subclasses_validation_error_and_attribute_error(self):
        self.assertTrue(issubclass(SpecValidationError, ValidationError))
        self.assertTrue(issubclass(SpecValidationError, PetersburgError))
        self.assertTrue(issubclass(SpecValidationError, ValueError))
        self.assertTrue(issubclass(SpecValidationError, AttributeError))

    def test_petersburg_error_is_not_a_value_error(self):
        self.assertFalse(issubclass(PetersburgError, ValueError))
        self.assertFalse(issubclass(PetersburgError, AttributeError))

    def test_validation_error_is_caught_as_value_error(self):
        caught = None
        try:
            raise ValidationError("boom")
        except ValueError as err:  # the compat contract: still a ValueError
            caught = err
        self.assertIsInstance(caught, PetersburgError)

    def test_spec_validation_error_is_caught_as_attribute_error(self):
        caught = None
        try:
            raise SpecValidationError("boom")
        except AttributeError as err:  # the compat contract: still an AttributeError
            caught = err
        self.assertIsInstance(caught, ValidationError)
        self.assertIsInstance(caught, ValueError)

    def test_spec_validation_error_is_not_a_bare_attribute_error(self):
        self.assertIsNot(type(SpecValidationError("boom")), AttributeError)

    def test_exceptions_are_exported_from_the_package(self):
        import petersburg

        for name in ("PetersburgError", "ValidationError", "SpecValidationError"):
            self.assertIs(getattr(petersburg, name), globals()[name])
            self.assertIn(name, petersburg.__all__)


class TestGraphSpecValidation(unittest.TestCase):
    """Graph.from_dict spec validation raises SpecValidationError, not bare AttributeError."""

    def setUp(self):
        self.graph = Graph()

    def _assert_spec_error(self, bad_spec, *message_parts):
        with self.assertRaises(PetersburgError) as ctx:
            self.graph.from_dict(bad_spec)
        self.assertIsNot(type(ctx.exception), AttributeError)
        for part in message_parts:
            self.assertIn(part, str(ctx.exception))
        return ctx.exception

    def test_empty_dict_has_no_starting_node(self):
        self._assert_spec_error({})

    def test_multiple_starting_nodes_rejected(self):
        self._assert_spec_error({1: {"after": []}, 2: {"after": []}})

    def test_unknown_node_type_rejected_with_accepted_list(self):
        self._assert_spec_error(
            {1: {"type": "poisson", "after": []}}, "poisson", "gaussian", "uniform"
        )

    def test_unknown_predecessor_reference_rejected(self):
        self._assert_spec_error(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 0, "after": [{"node_id": 9}]},
            },
            "9",
        )

    def test_self_cycle_rejected(self):
        self._assert_spec_error({1: {"payoff": 0, "after": [{"node_id": 1}]}})

    def test_disconnected_cycle_rejected(self):
        self._assert_spec_error(
            {
                1: {"payoff": 0, "after": []},
                2: {"payoff": 0, "after": [{"node_id": 3}]},
                3: {"payoff": 0, "after": [{"node_id": 2}]},
            }
        )

    def test_non_dict_specification_rejected(self):
        self._assert_spec_error([{"node_id": 1}])

    def test_non_dict_node_spec_rejected(self):
        self._assert_spec_error({1: "payoff only"})

    def test_failures_are_still_caught_as_value_error(self):
        try:
            self.graph.from_dict({})
        except ValueError:
            pass
        else:
            self.fail("spec validation was not catchable as ValueError")

    def test_failures_are_still_caught_as_attribute_error(self):
        try:
            self.graph.from_dict({})
        except AttributeError:
            pass
        else:
            self.fail("spec validation lost its historical AttributeError catchability")

    def test_failed_rebuild_preserves_existing_graph(self):
        self.graph.from_dict(_two_node_spec())
        original_start = self.graph.start_node
        self._assert_spec_error({1: {"after": []}, 2: {"after": []}})
        self.assertIs(self.graph.start_node, original_start)


class TestAdjacencyMatrixValidation(unittest.TestCase):
    """from_adj_matrix matrix validation raises ValidationError (still a ValueError)."""

    def setUp(self):
        self.graph = Graph()

    def test_non_square_matrix_rejected(self):
        with self.assertRaises(ValidationError):
            self.graph.from_adj_matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_complex_entries_rejected(self):
        with self.assertRaises(ValidationError):
            self.graph.from_adj_matrix(np.array([[1 + 2j, 0], [0, 1 - 2j]]))

    def test_infinite_entries_rejected(self):
        with self.assertRaises(ValidationError):
            self.graph.from_adj_matrix(np.array([[np.inf, 0], [0, 1.0]]))

    def test_negative_entries_rejected(self):
        with self.assertRaises(ValidationError):
            self.graph.from_adj_matrix(np.array([[-1.0, 0], [0, 1.0]]))

    def test_failures_are_still_caught_as_value_error(self):
        try:
            self.graph.from_adj_matrix(np.zeros((3, 2)))
        except ValueError:
            pass
        else:
            self.fail("matrix validation was not catchable as ValueError")


class TestSampleCountAndSensitivityValidation(unittest.TestCase):
    """Simulation-argument validation raises ValidationError (still a ValueError)."""

    def setUp(self):
        self.graph = Graph().from_dict(_two_node_spec())

    def test_get_options_rejects_zero_iters(self):
        with self.assertRaises(ValidationError):
            self.graph.get_options(iters=0)

    def test_analyze_sensitivity_rejects_unknown_parameter_type(self):
        with self.assertRaises(ValidationError):
            self.graph.analyze_sensitivity(parameter_type="bogus")

    def test_analyze_sensitivity_rejects_non_positive_perturbation(self):
        with self.assertRaises(ValidationError):
            self.graph.analyze_sensitivity(perturbation=0, num_simulations=2)

    def test_analyze_sensitivity_rejects_perturbation_of_one(self):
        with self.assertRaises(ValidationError):
            self.graph.analyze_sensitivity(perturbation=1.0, num_simulations=2)

    def test_identify_critical_parameters_rejects_bad_sample_count(self):
        with self.assertRaises(ValidationError):
            self.graph.identify_critical_parameters(num_simulations=-3)

    def test_failures_are_still_caught_as_value_error(self):
        with self.assertRaises(ValueError):
            self.graph.get_options(iters=0)


class TestNodeValidation(unittest.TestCase):
    """Node-level validation raises ValidationError (still a ValueError)."""

    def test_scaled_payoff_rejects_zero_factor(self):
        with self.assertRaises(ValidationError):
            with Node(1).scaled_payoff(0):
                pass

    def test_scaled_payoff_rejects_negative_factor(self):
        with self.assertRaises(ValidationError):
            with Node(1).scaled_payoff(-0.5):
                pass

    def test_negative_transition_weight_rejected_at_simulation(self):
        node = Node(1)
        node.add_outcome(Node(2), weight=-1)
        with self.assertRaises(ValidationError):
            node.get_outcome()

    def test_all_zero_transition_weights_rejected_at_simulation(self):
        node = Node(1)
        node.add_outcome(Node(2), weight=0)
        node.add_outcome(Node(3), weight=0)
        with self.assertRaises(ValidationError):
            node.get_outcome()

    def test_failures_are_still_caught_as_value_error(self):
        with self.assertRaises(ValueError):
            with Node(1).scaled_payoff(0):
                pass


class TestEstimatorInputValidation(unittest.TestCase):
    """Estimator input validation raises ValidationError (still a ValueError)."""

    def test_fit_rejects_one_dimensional_target(self):
        X, _ = _estimator_data()
        with self.assertRaises(ValidationError):
            FrequencyEstimator().fit(X, np.array([0, 1, 0, 1, 0, 1]))

    def test_fit_rejects_empty_target(self):
        X, _ = _estimator_data()
        with self.assertRaises(ValidationError):
            FrequencyEstimator().fit(X, np.empty((0, 2)))

    def test_fit_rejects_single_column_target(self):
        X, _ = _estimator_data()
        with self.assertRaises(ValidationError):
            FrequencyEstimator().fit(X, np.array([[0], [1], [0], [1], [0], [1]]))

    def test_fit_rejects_one_dimensional_feature_matrix(self):
        _, y = _estimator_data()
        with self.assertRaises(ValidationError):
            _fitted_mixed_mode_estimator().fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), y)

    def test_mixed_mode_fit_rejects_row_count_mismatch(self):
        X, y = _estimator_data()
        with self.assertRaises(ValidationError):
            MixedModeEstimator().fit(X, y[:-1])

    def test_predict_rejects_one_dimensional_feature_matrix(self):
        est = _fitted_frequency_estimator()
        with self.assertRaises(ValidationError):
            est.predict(np.array([1.0, 2.0]))

    def test_predict_rejects_non_positive_num_simulations(self):
        X, _ = _estimator_data()
        est = _fitted_frequency_estimator(num_simulations=0)
        with self.assertRaises(ValidationError):
            est.predict(X)

    def test_partial_fit_rejects_unknown_category(self):
        est = _fitted_frequency_estimator()
        X, _ = _estimator_data()
        with self.assertRaises(ValidationError):
            est.partial_fit(X[:1], np.array([[0, 7]]))

    def test_score_rejects_one_dimensional_target(self):
        est = _fitted_frequency_estimator()
        X, _ = _estimator_data()
        with self.assertRaises(ValidationError):
            est.score(X, np.array([0, 1, 0, 1, 0, 1]))

    def test_failures_are_still_caught_as_value_error(self):
        X, _ = _estimator_data()
        with self.assertRaises(ValueError):
            FrequencyEstimator().fit(X, np.array([0, 1, 0, 1, 0, 1]))


if __name__ == "__main__":
    unittest.main()
