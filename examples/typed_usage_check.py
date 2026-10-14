"""Snippet type-checked by CI against the installed wheel to prove py.typed is honoured."""

from petersburg import Graph

g = Graph(random_state=1).from_dict(
    {
        1: {"after": []},
        2: {"after": [{"node_id": 1, "cost": 1, "weight": 1}], "payoff": 3},
    }
)
ev: float = g.expected_value()
reveal: dict = g.to_dict()
print(ev, reveal)
