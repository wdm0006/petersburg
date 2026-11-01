"""Exact value of information over finite scenario tables."""

import random
import unittest

from petersburg import ValidationError, value_of_information
from petersburg.graph import MAX_INFORMATION_ACTIONS, MAX_INFORMATION_SCENARIOS

ACTIONS = ["A", "B"]


def _table(*rows):
    return {
        f"s{i}": {"probability": p, "payoffs": dict(zip(ACTIONS, pay))}
        for i, (p, pay) in enumerate(rows, 1)
    }


CROSSING = _table((0.5, (30, 0)), (0.5, (0, 20)))
# four equally likely scenarios: {s1, s2} favour A, {s3, s4} favour B
FOUR = _table((0.25, (10, 0)), (0.25, (6, 0)), (0.25, (0, 10)), (0.25, (0, 6)))
# s1, s3 favour A; s2, s4 favour B; only s4 is worth separating from the rest
MIXED = _table((0.25, (10, 0)), (0.25, (0, 8)), (0.25, (6, 0)), (0.25, (0, 2)))


class TestFixture(unittest.TestCase):
    def test_crossing_values(self):
        r = value_of_information(CROSSING, ACTIONS, cost=5)
        self.assertEqual(r["value_without"], 15.0)
        self.assertEqual(r["value_with"], 25.0)
        self.assertEqual(r["gross_value"], 10.0)
        self.assertEqual(r["net_value"], 5.0)
        self.assertEqual(r["policy_without"], "A")
        self.assertEqual(r["policy_with"], {("s1",): "A", ("s2",): "B"})

    def test_none_and_singletons_agree(self):
        full = value_of_information(CROSSING, ACTIONS, cost=5)
        singles = value_of_information(CROSSING, ACTIONS, [["s1"], ["s2"]], cost=5)
        self.assertEqual(full, singles)

    def test_single_group_is_no_information(self):
        r = value_of_information(CROSSING, ACTIONS, [["s1", "s2"]])
        self.assertEqual(r["gross_value"], 0.0)
        self.assertEqual(r["value_with"], r["value_without"])
        self.assertEqual(r["policy_with"], {("s1", "s2"): "A"})

    def test_coarse_partition_is_strictly_between(self):
        full = value_of_information(FOUR, ACTIONS)
        coarse = value_of_information(FOUR, ACTIONS, [["s1", "s2"], ["s3", "s4"]])
        none = value_of_information(FOUR, ACTIONS, [["s1", "s2", "s3", "s4"]])
        self.assertEqual(none["gross_value"], 0.0)
        self.assertEqual(coarse["gross_value"], 4.0)
        self.assertEqual(full["gross_value"], 4.0)
        # a partition that mixes favourable and unfavourable scenarios is strictly partial
        full = value_of_information(MIXED, ACTIONS)
        part = value_of_information(MIXED, ACTIONS, [["s1", "s2", "s3"], ["s4"]])
        self.assertEqual(full["value_without"], 4.0)
        self.assertEqual(full["gross_value"], 2.5)
        self.assertEqual(part["gross_value"], 0.5)
        self.assertTrue(0 < part["gross_value"] < full["gross_value"])

    def test_zero_probability_group_omitted(self):
        t = _table((1.0, (3, 1)), (0.0, (0, 9)))
        r = value_of_information(t, ACTIONS)
        self.assertEqual(r["gross_value"], 0.0)
        self.assertEqual(r["policy_with"], {("s1",): "A"})

    def test_gross_never_negative_on_random_tables(self):
        for seed in range(8):
            rng = random.Random(seed)
            n = rng.randint(2, 6)
            raw = [rng.random() + 0.01 for _ in range(n)]
            table = {
                f"s{i}": {
                    "probability": w / sum(raw),
                    "payoffs": {a: rng.uniform(-50, 50) for a in ACTIONS + ["C"]},
                }
                for i, w in enumerate(raw)
            }
            names = list(table)
            groups = [names[: n // 2], names[n // 2 :]]
            full = value_of_information(table, ACTIONS + ["C"])
            part = value_of_information(table, ACTIONS + ["C"], groups)
            self.assertGreaterEqual(part["gross_value"], 0.0)
            self.assertGreaterEqual(full["gross_value"], part["gross_value"] - 1e-9)


class TestValidation(unittest.TestCase):
    def bad(self, *args, **kwargs):
        with self.assertRaises(ValidationError) as ctx:
            value_of_information(*args, **kwargs)
        return str(ctx.exception)

    def test_costs(self):
        for cost in (-1, float("nan"), float("inf"), True, "5", None):
            self.assertIn("cost", self.bad(CROSSING, ACTIONS, cost=cost))

    def test_probabilities(self):
        self.assertIn("sum to 1", self.bad(_table((0.5, (1, 2)), (0.6, (1, 2))), ACTIONS))
        self.assertIn("probability", self.bad(_table((-0.5, (1, 2)), (1.5, (1, 2))), ACTIONS))
        self.assertIn(
            "probability", self.bad(_table((float("nan"), (1, 2)), (1.0, (1, 2))), ACTIONS)
        )
        self.assertIn("non-empty", self.bad({}, ACTIONS))

    def test_ragged_and_bad_payoffs(self):
        ragged = _table((0.5, (1, 2)), (0.5, (1, 2)))
        del ragged["s2"]["payoffs"]["B"]
        self.assertIn("exactly the actions", self.bad(ragged, ACTIONS))
        self.assertIn("finite", self.bad(_table((1.0, (float("inf"), 2))), ACTIONS))
        self.assertIn("at least one action", self.bad(CROSSING, []))
        self.assertIn("duplicates", self.bad(CROSSING, ["A", "A"]))

    def test_partitions(self):
        self.assertIn("more than once", self.bad(CROSSING, ACTIONS, [["s1"], ["s1", "s2"]]))
        self.assertIn("does not cover", self.bad(CROSSING, ACTIONS, [["s1"]]))
        self.assertIn("unknown scenario", self.bad(CROSSING, ACTIONS, [["s1", "s2", "zz"]]))
        self.assertIn("empty", self.bad(CROSSING, ACTIONS, [["s1", "s2"], []]))

    def test_size_bounds(self):
        n = MAX_INFORMATION_SCENARIOS + 1
        big = {i: {"probability": 1 / n, "payoffs": {"A": 0}} for i in range(n)}
        self.assertIn("scenarios", self.bad(big, ["A"]))
        many = [f"a{i}" for i in range(MAX_INFORMATION_ACTIONS + 1)]
        table = {"s": {"probability": 1.0, "payoffs": dict.fromkeys(many, 0)}}
        self.assertIn("actions", self.bad(table, many))


if __name__ == "__main__":
    unittest.main()
