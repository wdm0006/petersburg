"""
.. module:: graph
   :platform: Unix, Windows
   :synopsis:

.. moduleauthor:: Will McGinnis <will@pedalwrencher.com>


"""

import numpy as np

from petersburg.nodes import (
    GaussianNode,
    LogNormalNode,
    Node,
    PowerLawNode,
    UniformNode,
    is_numeric_weight,
)

__author__ = "willmcginnis"


class Graph:
    """
    A graph holds a heirarchy of nodes and edges with payoffs and costs.

    Example:

    >>> from petersburg import Graph
    >>> g = Graph()
    >>> g.from_dict({
    >>>      1: {'payoff': 0, 'after': []},
    >>>      2: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 10}]},
    >>>      3: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 10}]},
    >>>      4: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 10}]},
    >>>      5: {'payoff': 0, 'after': [{'node_id': 2, 'cost': 5}, {'node_id': 3, 'cost': 10}]},
    >>>      6: {'payoff': 0, 'after': [{'node_id': 2, 'cost': 5}, {'node_id': 4, 'cost': 10}]},
    >>>      7: {'payoff': 10, 'after': [{'node_id': 5, 'cost': 0}]},
    >>>      8: {'payoff': 3, 'after': [{'node_id': 5, 'cost': 0}]},
    >>>      9: {'payoff': 10, 'after': [{'node_id': 6, 'cost': 0}]},
    >>>      10: {'payoff': 3, 'after': [{'node_id': 6, 'cost': 0}]},
    >>> })

    Which represents a decision between 3 options with differing costs and outcomes. The starting point (of which there
    can only be one, is represented by an empty list in the 'after' key of a node.

    """

    def __init__(self):
        self.start_node = None

    def from_dict(self, d):
        """
        Assembles a graph from a dictionary of nodes and their dependencies. Assumes, directed acyclic and with one
        starting node.  Each node has a payoff, each edge has a cost, and each edge has a weight which corresponds to
        likelyhood of being traversed.

        Nodes can be specified with different types using the 'type' key:
        - 'fixed' or omitted: Node with fixed payoff (default)
        - 'uniform': UniformNode with 'min_payoff' and 'max_payoff'
        - 'gaussian': GaussianNode with 'mean' and 'std'
        - 'lognormal': LogNormalNode with 'mu' and 'sigma'
        - 'powerlaw': PowerLawNode with 'scale' and 'alpha'

        :param d:
        :return:
        """

        # here we iterate through the dictionary and instantiate all of the node objects specified. One of them (and
        # exactly one of them), should be after nothing, and is the starting node, the rest go into a node list. Edges
        # are handled later.
        node = None
        node_list = {}
        for key in d:
            # Create the appropriate node type based on the 'type' key
            node_spec = d[key]
            node_type = node_spec.get("type", "fixed").lower()

            if node_type == "uniform":
                new_node = UniformNode(
                    key, node_spec.get("min_payoff", 0), node_spec.get("max_payoff", 0)
                )
            elif node_type == "gaussian":
                new_node = GaussianNode(key, node_spec.get("mean", 0), node_spec.get("std", 1))
            elif node_type == "lognormal":
                new_node = LogNormalNode(key, node_spec.get("mu", 0), node_spec.get("sigma", 1))
            elif node_type == "powerlaw":
                new_node = PowerLawNode(key, node_spec.get("scale", 1), node_spec.get("alpha", 2))
            else:  # 'fixed' or any other value defaults to Node
                new_node = Node(key, payoff=node_spec.get("payoff", 0))

            if not node_spec["after"]:
                if node is not None:
                    raise AttributeError("Graph cannot have more than one starting node.")
                node = new_node
                node_list.update({key: new_node})
            else:
                node_list.update({key: new_node})

        if node is None:
            raise AttributeError("Dict must contain a starting node (empty list for after key)")

        # the start node is the entry point for basically everything we will do later on. It's the only part we actually
        # need to keep around in the instance here, because all other nodes will be under it via reference once we
        # process the edges.
        self.start_node = node

        # now that we have a node list, we want to iterate through all of the other nodes, and then through the after
        # list specified for each, and add the connections that create the graph.
        for key in d:
            for edge in d[key]["after"]:
                if edge["node_id"] not in node_list:
                    raise AttributeError(
                        f"Node {key} lists unknown predecessor node_id {edge['node_id']} "
                        f"in its 'after' list"
                    )
                node_list[edge["node_id"]].add_outcome(
                    node_list[key], cost=edge.get("cost", 0), weight=edge.get("weight", 1)
                )

        return self

    def from_adj_matrix(self, A, labels=None, clf_matrix=None):
        """
        Takes in a numpy adjacency matrix and forms a petersburg graph from it.

        The matrix is of type [row -> col]: a nonzero, non-NaN entry A[r, c] creates an edge
        from node r to node c, whose weight is A[r, c] normalized by the sum of row r. A
        synthetic root node with id -1 is injected and becomes the graph's start node,
        connecting to every node that has no predecessors.

        So ``A[0, 1] = 1`` builds the chain -1 -> 0 -> 1.

        :param A:
        :return:
        """

        if labels is None:
            labels = [(1, 1) for _ in range(A.shape[0])]
            labels[0] = (0, 0)

        if A.shape[0] != A.shape[1]:
            raise ValueError("Adjanceny Matrix must be square")

        dict_spec = {}
        for c_idx in range(A.shape[1]):
            after = []
            for r_idx in range(A.shape[0]):
                if A[r_idx, c_idx] != 0.0 and not np.isnan(A[r_idx, c_idx]):
                    row_sum = np.nansum(A[r_idx, :])
                    if row_sum != 0.0:
                        weight = A[r_idx, c_idx] / row_sum
                    else:
                        weight = 0.0
                    try:
                        clf = clf_matrix[r_idx][c_idx]
                        if clf is not None:
                            after.append(
                                {
                                    "node_id": r_idx,
                                    "cost": 0,
                                    "weight": clf,
                                    "_weight": weight,
                                    "_cnt": A[r_idx, c_idx],
                                }
                            )
                        else:
                            after.append(
                                {
                                    "node_id": r_idx,
                                    "cost": 0,
                                    "weight": weight,
                                    "_weight": weight,
                                    "_cnt": A[r_idx, c_idx],
                                }
                            )
                    except (IndexError, TypeError):
                        after.append(
                            {
                                "node_id": r_idx,
                                "cost": 0,
                                "weight": weight,
                                "_weight": weight,
                                "_cnt": A[r_idx, c_idx],
                            }
                        )

            if len(after) > 0 or labels[c_idx][0] == 0:
                dict_spec[c_idx] = {"payoff": 0, "after": after}

        # add in root node (super hacky)
        dict_spec[-1] = {"after": [], "payoff": 0}
        for k in dict_spec.keys():
            if k != -1 and len(dict_spec[k].get("after", [])) == 0:
                # the weight is the sum of all _cnt values that follow this node
                weight = 0
                for node in dict_spec.keys():
                    for a in dict_spec[node].get("after", []):
                        if a.get("node_id", None) == k:
                            weight += a.get("_cnt", 0)

                dict_spec[k]["after"] = [{"node_id": -1, "weight": weight, "cost": 0}]

        return self.from_dict(dict_spec)

    def get_outcome(self, iters=None, ruin=False, starting_bank=0, feature_vector=None):
        """
        Starting with the starting node, the graph is walked once, and the profit is returned, run multiple times to get
        an expected value estimate.

        :return:
        """
        if iters is None:
            payoff, cost = self.start_node.get_outcome(feature_vector)
            return payoff - cost
        else:
            bank = starting_bank
            for _ in range(iters):
                payoff, cost = self.start_node.get_outcome(feature_vector)
                bank = bank + payoff - cost
                if ruin:
                    if bank <= 0:
                        return 0
            return bank

    def get_outcome_node(self, feature_vector=None):
        """
        Starting with the starting node, the graph is walked once, and the ID of the final node reached is returned

        :return:
        """

        node_id = self.start_node.get_outcome_node(feature_vector)

        return node_id

    def get_options(self, iters=100, extended_stats=False, feature_vector=None):
        """
        Starts with each of the outcomes from the starting node seperately, to get the expected values (using iters
        iterations) for each of the initial options. Returns a dictionary of node_id: expected profit pairs.

        :param iters:
        :param extended_stats:
        :param feature_vector: Features passed to classifier-weighted edges. Required when
            the graph uses classifiers for edge weights.
        :return:
        """

        choice = {}
        for outcome in self.start_node.outcomes:
            out = []
            for _ in range(iters):
                payoff, cost = outcome[0].get_outcome(feature_vector=feature_vector)
                out.append(payoff - cost - outcome[0].cost)
            if not extended_stats:
                choice.update({outcome[0].to_node.node_id: float(sum(out)) / len(out)})
            else:
                choice.update(
                    {
                        outcome[0].to_node.node_id: {
                            "mean": float(sum(out)) / len(out),
                            "max": max(out),
                            "min": min(out),
                            "count": len(out),
                        }
                    }
                )
        return choice

    def to_tree(self):
        """

        :return:
        """

        return self.start_node.to_tree()

    def to_networkx(self):
        """

        :return:
        """
        try:
            import networkx as nx
        except ImportError as err:
            raise ImportError("the to networkx function requires networkx") from err

        g = nx.DiGraph()

        # first make a node id: obj mapping and add nodes to the graph
        node_to_node_id = {node: node.node_id for node in self.node_list()}
        nodes = list(node_to_node_id.values())
        g.add_nodes_from(nodes)

        # now iterate through and add in our edges using that mapping
        edges = list(self.edge_list())
        for edge in edges:
            from_node_id = node_to_node_id.get(edge.from_node)
            to_node_id = node_to_node_id.get(edge.to_node)
            cost = edge.cost
            g.add_edge(from_node_id, to_node_id, weight=cost)

        return g

    def edge_list(self):
        return self.start_node.get_edges(set())

    def node_list(self):
        return self.start_node.get_nodes(set())

    def to_mermaid(self, orientation="LR", max_nodes=50):
        """
        Export the graph to Mermaid diagram syntax.

        Nodes are emitted with the start node first, then in ascending node id order, so
        repeated exports of the same graph produce byte-for-byte identical text and
        `max_nodes` truncation always keeps the start node.

        :param orientation: Graph orientation ('LR' for left-right, 'TD' for top-down)
        :param max_nodes: Maximum number of nodes to include (for large graphs)
        :return: String containing Mermaid diagram syntax
        """
        lines = [f"graph {orientation}"]

        nodes = self._mermaid_node_order()

        # Limit nodes if graph is too large
        if len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]

        included = set(nodes)
        edges = sorted(
            (e for e in self.edge_list() if e.from_node in included and e.to_node in included),
            key=self._mermaid_edge_sort_key,
        )

        # Create node ID to display mapping
        node_to_id = {node: node.node_id for node in nodes}

        # Add node definitions with payoffs
        for node in nodes:
            node_id = node_to_id[node]

            # Check if this is the start node
            if node == self.start_node:
                lines.append(f"    {node_id}((Start))")
            elif node.payoff != 0:
                label = f"Node {node_id}<br/>Payoff: ${node.payoff}"
                lines.append(f'    {node_id}["{label}"]')
            elif len(node.outcomes) == 0:
                # Terminal/leaf node
                lines.append(f"    {node_id}[End]")
            else:
                lines.append(f"    {node_id}[Node {node_id}]")

        # Add edges with costs and weights
        for edge in edges:
            from_id = node_to_id.get(edge.from_node)
            to_id = node_to_id.get(edge.to_node)

            if from_id is None or to_id is None:
                continue

            # Build edge label
            label_parts = []
            if edge.cost != 0:
                label_parts.append(f"Cost: ${edge.cost}")

            # Try to get weight from edge (it's stored in the from_node's outcomes)
            weight = None
            for outcome_edge, w in edge.from_node.outcomes:
                if outcome_edge == edge:
                    if is_numeric_weight(w):
                        weight = w
                    break

            if weight is not None and weight != 1.0:
                label_parts.append(f"P: {weight:.2f}")

            if label_parts:
                label = " | ".join(label_parts)
                lines.append(f"    {from_id} -->|{label}| {to_id}")
            else:
                lines.append(f"    {from_id} --> {to_id}")

        # Add styling
        lines.append("")
        lines.append("    classDef terminal fill:#e1f5e1")
        lines.append("    classDef payoff fill:#fff4e1")

        # Mark terminal nodes (those included in the export with no outgoing outcomes)
        for node in nodes:
            if len(node.outcomes) == 0:
                lines.append(f"    class {node_to_id[node]} terminal")

        # Mark nodes with payoffs
        for node in nodes:
            if node.payoff > 0:
                lines.append(f"    class {node_to_id[node]} payoff")

        return "\n".join(lines)

    @staticmethod
    def _edge_sort_key(edge):
        return (str(edge.from_node.node_id), str(edge.to_node.node_id), str(edge.cost))

    @staticmethod
    def _node_sort_key(node):
        return str(node.node_id)

    def _mermaid_node_order(self):
        """
        Nodes in export order: the start node first, then the rest by ascending node id.

        :return: List of nodes in a stable, process-independent order
        """
        rest = sorted(
            (n for n in self.node_list() if n is not self.start_node), key=self._node_sort_key
        )
        return [self.start_node] + rest

    @staticmethod
    def _mermaid_edge_sort_key(edge):
        """
        Sort key for an edge, tie-broken by its position in its source node's outcomes so
        that parallel edges sharing endpoints and cost still order deterministically.

        :param edge: The edge to key
        :return: Tuple usable as a total order over a graph's edges
        """
        position = next(
            (
                idx
                for idx, (outcome_edge, _) in enumerate(edge.from_node.outcomes)
                if outcome_edge is edge
            ),
            -1,
        )
        return Graph._edge_sort_key(edge) + (position,)

    @staticmethod
    def _numeric_weight(edge):
        """
        Locate an edge's numeric weight in its source node's outcomes.

        :param edge: The edge to look up
        :return: (index, weight) tuple, or None if the edge has no numeric weight
        """
        for idx, (outcome_edge, weight) in enumerate(edge.from_node.outcomes):
            if outcome_edge == edge and is_numeric_weight(weight):
                return idx, weight
        return None

    @staticmethod
    def _elasticity(sensitivity, baseline_ev):
        """
        Express a sensitivity as a fraction of the baseline expected value.

        ``sensitivity`` is a mean of absolute deviations, so elasticity is a magnitude:
        the baseline's absolute value is used and a negative baseline does not flip the sign.

        :param sensitivity: Mean absolute change in expected value
        :param baseline_ev: Unperturbed expected value
        :return: sensitivity / abs(baseline_ev), or 0 when the baseline is zero
        """
        return (sensitivity / abs(baseline_ev)) if baseline_ev != 0 else 0

    def analyze_sensitivity(
        self, parameter_type="edge_weights", num_simulations=1000, perturbation=0.1, max_params=10
    ):
        """
        Automatically analyze sensitivity of outcomes to graph parameters.

        This method identifies which parameters (edge weights, costs, or payoffs) have
        the most impact on expected outcomes.

        Candidate parameters are ordered deterministically (by node id, then edge cost)
        before ``max_params`` is applied, so repeated runs on the same graph analyze the
        same parameters. The return value reports both how many candidates were eligible
        and how many were actually analyzed.

        :param parameter_type: Type of parameter to analyze ('edge_weights', 'costs', or 'payoffs')
        :param num_simulations: Number of Monte Carlo simulations per parameter value
        :param perturbation: How much to vary parameters, strictly between 0 and 1 (e.g., 0.1 = ±10%)
        :param max_params: Maximum number of parameters to analyze, or None for no limit
        :return: Dictionary with sensitivity results sorted by impact
        :raises ValueError: If perturbation is not strictly between 0 and 1
        """
        import numpy as np

        if not 0 < perturbation < 1:
            raise ValueError(f"perturbation must be strictly between 0 and 1, got {perturbation}")

        # Get baseline expected value
        baseline_outcomes = []
        for _ in range(num_simulations):
            baseline_outcomes.append(self.get_outcome())
        baseline_ev = np.mean(baseline_outcomes)

        sensitivity_results = []
        candidate_count = 0

        if parameter_type == "edge_weights":
            # Analyze each edge's weight sensitivity; edges without a usable numeric
            # weight are not candidates at all, so they never consume a max_params slot
            candidates = []
            for edge in sorted(self.edge_list(), key=self._edge_sort_key):
                found = self._numeric_weight(edge)
                if found is not None and found[1] != 0:
                    candidates.append((edge, found[0], found[1]))
            candidate_count = len(candidates)

            for edge, edge_index, original_weight in candidates[:max_params]:
                # Test increased weight
                edge.from_node.outcomes[edge_index] = (edge, original_weight * (1 + perturbation))
                increased_outcomes = []
                for _ in range(num_simulations):
                    increased_outcomes.append(self.get_outcome())
                increased_ev = np.mean(increased_outcomes)

                # Test decreased weight
                edge.from_node.outcomes[edge_index] = (
                    edge,
                    original_weight * (1 - perturbation),
                )
                decreased_outcomes = []
                for _ in range(num_simulations):
                    decreased_outcomes.append(self.get_outcome())
                decreased_ev = np.mean(decreased_outcomes)

                # Restore original weight
                edge.from_node.outcomes[edge_index] = (edge, original_weight)

                # Calculate sensitivity (average absolute change in EV)
                sensitivity = (
                    abs(increased_ev - baseline_ev) + abs(decreased_ev - baseline_ev)
                ) / 2

                sensitivity_results.append(
                    {
                        "parameter": f"Edge {edge.from_node.node_id}→{edge.to_node.node_id} weight",
                        "edge": edge,
                        "original_value": original_weight,
                        "baseline_ev": baseline_ev,
                        "increased_ev": increased_ev,
                        "decreased_ev": decreased_ev,
                        "sensitivity": sensitivity,
                        "elasticity": self._elasticity(sensitivity, baseline_ev),
                    }
                )

        elif parameter_type == "costs":
            # Analyze edge cost sensitivity; zero-cost edges are not candidates
            edges = sorted(self.edge_list(), key=self._edge_sort_key)
            candidates = [e for e in edges if e.cost != 0]
            candidate_count = len(candidates)

            for edge in candidates[:max_params]:
                original_cost = edge.cost

                # Test increased cost
                edge.cost = original_cost * (1 + perturbation)
                increased_outcomes = []
                for _ in range(num_simulations):
                    increased_outcomes.append(self.get_outcome())
                increased_ev = np.mean(increased_outcomes)

                # Test decreased cost
                edge.cost = original_cost * (1 - perturbation)
                decreased_outcomes = []
                for _ in range(num_simulations):
                    decreased_outcomes.append(self.get_outcome())
                decreased_ev = np.mean(decreased_outcomes)

                # Restore original cost
                edge.cost = original_cost

                sensitivity = (
                    abs(increased_ev - baseline_ev) + abs(decreased_ev - baseline_ev)
                ) / 2

                sensitivity_results.append(
                    {
                        "parameter": f"Edge {edge.from_node.node_id}→{edge.to_node.node_id} cost",
                        "edge": edge,
                        "original_value": original_cost,
                        "baseline_ev": baseline_ev,
                        "increased_ev": increased_ev,
                        "decreased_ev": decreased_ev,
                        "sensitivity": sensitivity,
                        "elasticity": self._elasticity(sensitivity, baseline_ev),
                    }
                )

        elif parameter_type == "payoffs":
            # Analyze node payoff sensitivity
            nodes = sorted(self.node_list(), key=self._node_sort_key)
            candidates = [n for n in nodes if n.payoff != 0]
            candidate_count = len(candidates)

            for node in candidates[:max_params]:
                original_payoff = node.payoff

                # Test increased payoff
                with node.scaled_payoff(1 + perturbation):
                    increased_outcomes = []
                    for _ in range(num_simulations):
                        increased_outcomes.append(self.get_outcome())
                    increased_ev = np.mean(increased_outcomes)

                # Test decreased payoff
                with node.scaled_payoff(1 - perturbation):
                    decreased_outcomes = []
                    for _ in range(num_simulations):
                        decreased_outcomes.append(self.get_outcome())
                    decreased_ev = np.mean(decreased_outcomes)

                sensitivity = (
                    abs(increased_ev - baseline_ev) + abs(decreased_ev - baseline_ev)
                ) / 2

                sensitivity_results.append(
                    {
                        "parameter": f"Node {node.node_id} payoff",
                        "node": node,
                        "original_value": original_payoff,
                        "baseline_ev": baseline_ev,
                        "increased_ev": increased_ev,
                        "decreased_ev": decreased_ev,
                        "sensitivity": sensitivity,
                        "elasticity": self._elasticity(sensitivity, baseline_ev),
                    }
                )

        # Sort by sensitivity (highest impact first)
        sensitivity_results.sort(key=lambda x: x["sensitivity"], reverse=True)

        return {
            "baseline_ev": baseline_ev,
            "parameter_type": parameter_type,
            "perturbation": perturbation,
            "max_params": max_params,
            "candidate_parameters": candidate_count,
            "parameters_analyzed": len(sensitivity_results),
            "results": sensitivity_results,
        }

    def identify_critical_parameters(
        self, num_simulations=1000, perturbation=0.1, top_n=5, max_params=10
    ):
        """
        Identify the most critical parameters in the graph across all parameter types.

        This is a convenience method that analyzes weights, costs, and payoffs, then
        returns the top N most impactful parameters regardless of type.

        :param num_simulations: Number of Monte Carlo simulations per parameter
        :param perturbation: How much to vary parameters (e.g., 0.1 = ±10%)
        :param top_n: Number of top parameters to return
        :param max_params: Maximum number of parameters analyzed per parameter type,
            or None for no limit
        :return: Dictionary with analysis summary and top parameters
        """
        all_results = []
        total_candidates = 0

        # Analyze all parameter types
        for param_type in ["edge_weights", "costs", "payoffs"]:
            analysis = self.analyze_sensitivity(
                parameter_type=param_type,
                num_simulations=num_simulations,
                perturbation=perturbation,
                max_params=max_params,
            )
            all_results.extend(analysis["results"])
            total_candidates += analysis["candidate_parameters"]

        # Sort all parameters by sensitivity
        all_results.sort(key=lambda x: x["sensitivity"], reverse=True)

        # Get top N
        top_parameters = all_results[:top_n]

        return {
            "baseline_ev": all_results[0]["baseline_ev"] if all_results else 0,
            "max_params": max_params,
            "total_candidate_parameters": total_candidates,
            "total_parameters_analyzed": len(all_results),
            "top_parameters": top_parameters,
        }

    def print_sensitivity_report(
        self, num_simulations=1000, perturbation=0.1, top_n=5, max_params=10
    ):
        """
        Print a formatted sensitivity analysis report.

        :param num_simulations: Number of Monte Carlo simulations per parameter
        :param perturbation: How much to vary parameters (e.g., 0.1 = ±10%)
        :param top_n: Number of top parameters to display
        :param max_params: Maximum number of parameters analyzed per parameter type,
            or None for no limit
        """
        print("=" * 80)
        print("AUTOMATIC SENSITIVITY ANALYSIS")
        print("=" * 80)
        print()

        results = self.identify_critical_parameters(
            num_simulations=num_simulations,
            perturbation=perturbation,
            top_n=top_n,
            max_params=max_params,
        )

        analyzed = results["total_parameters_analyzed"]
        candidates = results["total_candidate_parameters"]
        print(f"Baseline Expected Value: ${results['baseline_ev']:.2f}")
        print(f"Parameters Analyzed: {analyzed} of {candidates}")
        if analyzed < candidates:
            print(
                f"  NOTE: {candidates - analyzed} parameter(s) excluded by "
                f"max_params={max_params} (per parameter type). "
                f"Raise max_params to analyze them."
            )
        print(f"Perturbation: ±{perturbation*100:.0f}%")
        print(f"Simulations per parameter: {num_simulations:,}")
        print()

        print(f"Top {top_n} Most Sensitive Parameters:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Parameter':<35} {'Sensitivity':<15} {'Elasticity':<12}")
        print("-" * 80)

        for i, param in enumerate(results["top_parameters"], 1):
            sensitivity_str = f"${param['sensitivity']:.2f}"
            elasticity_str = f"{param['elasticity']*100:.1f}%"
            print(f"{i:<6} {param['parameter']:<35} {sensitivity_str:<15} {elasticity_str:<12}")

        print("-" * 80)
        print()

        if results["top_parameters"]:
            top_param = results["top_parameters"][0]
            print("Key Finding:")
            print(f"  The most sensitive parameter is: {top_param['parameter']}")
            print(
                f"  A {perturbation*100:.0f}% change in this parameter changes EV by ~${top_param['sensitivity']:.2f}"
            )
            print(
                f"  This represents a {top_param['elasticity']*100:.1f}% change in expected value"
            )
            print()

            # Provide actionable insight
            if "weight" in top_param["parameter"].lower():
                print("  → Focus on improving the probability of this transition")
            elif "cost" in top_param["parameter"].lower():
                print("  → Focus on reducing the cost of this step")
            elif "payoff" in top_param["parameter"].lower():
                print("  → Focus on increasing the value of this outcome")
            print()

    def plot(self, filename):
        """
        :return:
        """

        g = self.to_networkx()
        self.graph_draw(g, filename)

    @staticmethod
    def graph_draw(g, filename):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            import networkx as nx  # noqa: F401
            import pygraphviz  # noqa: F401
        except ImportError as err:
            raise ImportError("the plot function requires networkx and pygraphviz") from err

        # pure graphviz
        # A = nx.to_agraph(g)
        # A.layout(
        #     'dot',
        #     args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8'
        # )
        # A.draw(filename)

        # bastardization
        plt.figure()
        pos = nx.nx_agraph.graphviz_layout(
            g, prog="dot", args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8'
        )
        nx.draw(g, pos=pos)
        # edge_labels = nx.get_edge_attributes(g, 'weight')
        # print(edge_labels)
        # nx.draw_networkx_edge_labels(g, pos, labels=edge_labels)
        plt.savefig(filename)
        plt.close()
