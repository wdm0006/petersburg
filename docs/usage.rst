Usage
=====

Building a graph
----------------

Graphs are built from a dictionary specification with :meth:`~petersburg.graph.Graph.from_dict`.
Keys are node IDs; each node's ``after`` list is its **predecessors**, and the node with an
empty ``after`` list is where walks start:

.. code-block:: python

    from petersburg import Graph

    spec = {
        0: {'payoff': 0, 'after': []},  # Start node — the one with an empty 'after' list
        1: {'payoff': 100, 'after': [{'node_id': 0, 'cost': 10, 'weight': 0.3}]},  # Success branch (30%)
        2: {'payoff': -50, 'after': [{'node_id': 0, 'cost': 5, 'weight': 0.7}]},   # Failure branch (70%)
        3: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 0}, {'node_id': 2, 'cost': 0}]},  # Terminal node
    }
    g = Graph().from_dict(spec)

At each node the walk picks among the edges leaving it with probability proportional to each
edge's ``weight`` (normalized to sum to 1). A weight on a node's single outgoing edge has no
effect — that edge is always taken.

Simulating
----------

:meth:`~petersburg.graph.Graph.get_outcome` runs one Monte Carlo walk and returns the net
outcome (node payoffs minus edge costs). Pass an integer ``random_state`` to the ``Graph``
constructor (or an existing ``numpy.random.Generator``) for reproducible simulations:

.. code-block:: python

    outcomes = [g.get_outcome() for _ in range(10_000)]
    print(sum(outcomes) / len(outcomes))  # ≈ -11.5 for the graph above

Uncertain payoffs
-----------------

Distribution node types replace a fixed ``payoff`` with a sampled one, selected via the
``type`` key: ``uniform`` (``min_payoff``/``max_payoff``), ``gaussian`` (``mean``/``std``),
``lognormal`` (``mu``/``sigma``), and ``powerlaw`` (``scale``/``alpha``). See
:class:`~petersburg.nodes.UniformNode`, :class:`~petersburg.nodes.GaussianNode`,
:class:`~petersburg.nodes.LogNormalNode`, and :class:`~petersburg.nodes.PowerLawNode`.

Comparing initial options
-------------------------

:meth:`~petersburg.graph.Graph.get_options` simulates the walk once per outgoing edge of
the start node and compares the options by expected value. ``distribution=True`` augments
each option with statistics of its simulated outcome distribution; ``alpha`` (default
``0.05``) is the quantile level for ``var_alpha``/``cvar_alpha``, and
``return_samples=True`` (requires ``distribution=True``) also returns the raw per-walk
samples:

.. code-block:: python

    options = g.get_options(iters=10_000)
    # {1: 90.0, 2: -55.0}

    options = g.get_options(iters=10_000, distribution=True, alpha=0.05)
    # each option carries mean/max/min/count plus:
    #   std, percentiles (p5/p25/p50/p75/p95), p_loss, var_alpha, cvar_alpha

``cvar_alpha <= var_alpha`` always holds, because expected shortfall averages the outcomes
at or below the quantile.

Sensitivity analysis
--------------------

:meth:`~petersburg.graph.Graph.analyze_sensitivity`,
:meth:`~petersburg.graph.Graph.identify_critical_parameters`, and
:meth:`~petersburg.graph.Graph.print_sensitivity_report` perturb edge weights, edge costs,
and node payoffs automatically and rank the parameters by impact on expected value:

.. code-block:: python

    g.print_sensitivity_report(num_simulations=1000, perturbation=0.1, top_n=5)

With the ``visualization`` extra installed,
:meth:`~petersburg.graph.Graph.plot_sensitivity` returns a matplotlib Figure with
parameters ranked by sensitivity, largest at the top. It accepts either report
format, or computes a combined report when omitted. It never opens a window.
Keyword arguments style the bars; ``filename`` saves the figure.

.. code-block:: python

    report = g.identify_critical_parameters(top_n=10)
    figure = g.plot_sensitivity(report, top_n=5, filename="sensitivity.png")

For graphs with numeric weights,
:meth:`~petersburg.graph.Graph.exact_sensitivity` ranks the same parameters by
re-evaluating the exact expected value at ``±perturbation`` with no sampling, so its
numbers are identical across calls and seeds. Its report (uncapped, ``analysis_seed`` of
``None``) can be passed to :meth:`~petersburg.graph.Graph.plot_sensitivity`; classifier
weights raise ``ValidationError``.

Each parameter's baseline and both of its perturbation arms replay one shared stream of
random draws (common random numbers), so a reported ``sensitivity`` measures the
parameter's effect rather than the sum of three independent sampling errors. Costs and
payoffs gain an exact consequence: ``sensitivity <= perturbation * abs(value)``. Edge
weights have no comparable bound — perturbing a weight changes which edge is selected, at
which point the paired streams legitimately diverge — so weight rankings still vary
somewhat by seed on a heavy-tailed graph.

The seed for that shared stream is reported as ``analysis_seed``. A graph built with a
``random_state`` derives it from its own generator without consuming it, so seeded
analyses repeat exactly; an unseeded graph draws a fresh one each call. Supplying a
pre-computed ``baseline_ev`` requires supplying the ``analysis_seed`` it was drawn under,
since a baseline from another stream would leave the arms unpaired.

Error handling
--------------

Every error the public API raises subclasses :class:`~petersburg.exceptions.PetersburgError`:

- :class:`~petersburg.exceptions.ValidationError` (also a ``ValueError``) — invalid input
  values: adjacency matrices, sensitivity arguments, estimator targets and feature
  matrices, node payoffs, transition weights, and invalid argument combinations.
- :class:`~petersburg.exceptions.SpecValidationError` (a ``ValidationError`` that is also
  an ``AttributeError``) — invalid graph specifications: unrecognized node types, unknown
  node references, cycles, missing or multiple starting nodes, non-dict specifications.

Existing ``except ValueError`` and ``except AttributeError`` handlers keep working; new
code should catch ``PetersburgError`` or ``ValidationError``.

Exact expected value
--------------------

:meth:`~petersburg.graph.Graph.expected_value` returns the exact expected net outcome of
``get_outcome()`` without simulation, for graphs whose transition weights are all numeric
(classifier weights raise ``ValidationError``).

:meth:`~petersburg.graph.Graph.variance` and :meth:`~petersburg.graph.Graph.std` compute
exact population spread of the net outcome. They require numeric weights and finite
moments; power-law nodes require ``alpha > 2``. Zero-weight branches are skipped.

Exact fixed-payoff outcome probabilities
----------------------------------------

:meth:`~petersburg.graph.Graph.outcome_distribution` returns an ascending-outcome-ordered
mapping of net payoff to probability without sampling. It requires fixed payoff nodes,
finite real payoffs and costs, and numeric transition weights throughout the reachable
model, including zero-weight branches. Zero-weight edges propagate no mass.

``max_outcomes=10000`` bounds distinct outcomes at every node during accumulation;
exceeding it raises ``ValidationError`` without a partial or approximate result. Only
exactly equal numeric outcomes merge; floating-point representations that differ remain
separate atoms, with no rounding or binning.

.. code-block:: python

    g = Graph().from_dict({
        1: {"payoff": 5, "after": []},
        2: {"payoff": 10, "after": [{"node_id": 1, "cost": 2, "weight": 3}]},
        3: {"payoff": -4, "after": [{"node_id": 1, "cost": 3, "weight": 1}]},
    })
    masses = g.outcome_distribution()  # {-2: 0.25, 13: 0.75}
    p_loss = sum(p for payoff, p in masses.items() if payoff < 0)  # 0.25

:meth:`~petersburg.graph.Graph.option_distributions` returns the same kind of distribution
for each initial option, keyed like ``get_options()`` and ``expected_options()``. Each first
edge is forced regardless of its start weight; the start payoff and edge cost are included.
Restrictions and the ``max_outcomes`` cap match ``outcome_distribution``.

.. code-block:: python

    g.option_distributions()  # {2: {13: 1.0}, 3: {-2: 1.0}}

:meth:`~petersburg.graph.Graph.option_statistics` reduces those distributions, in one call, to
``mean``, ``std`` (population), ``p_loss``, ``var_alpha`` and ``cvar_alpha`` per option.
``var_alpha`` is the smallest outcome whose cumulative probability reaches ``alpha`` (a
discrete quantile, unlike the interpolated percentiles of ``get_options(distribution=True)``);
``cvar_alpha`` is the probability-weighted mean of all outcomes at or below it, including the
whole boundary atom.

.. code-block:: python

    g.option_statistics(alpha=0.5)[3]
    # {'mean': -2.0, 'std': 0.0, 'p_loss': 1.0, 'var_alpha': -2.0, 'cvar_alpha': -2.0}

Decision nodes
--------------

A node with ``"decision": True`` lets the decision-maker choose its outgoing edge; its edge
weights are ignored. :meth:`~petersburg.graph.Graph.optimal_policy` returns the exact optimal
value and the choice at every reachable decision node (keyed like ``get_options()``), and
:meth:`~petersburg.graph.Graph.apply_policy` returns a new graph with those choices fixed, so
the exact evaluators and simulators work on it. Calling them on a graph that still contains
decision nodes raises :class:`~petersburg.exceptions.ValidationError`.

.. code-block:: python

    g = Graph().from_dict({
        1: {"decision": True, "after": []},
        2: {"payoff": 8, "after": [{"node_id": 1}]},
        3: {"payoff": 2, "decision": True, "after": [{"node_id": 1, "cost": 1}]},
        4: {"payoff": 16, "after": [{"node_id": 3, "cost": 4}]},
        5: {"payoff": 0, "after": [{"node_id": 3}]},
    })
    best = g.optimal_policy()  # {"value": 13.0, "policy": {1: 3, 3: 4}}
    resolved = g.apply_policy(best["policy"])  # a new graph with those choices fixed
    resolved.expected_value()  # 13.0; variance(), exact_sensitivity() and simulation work too

Serialization
-------------

:meth:`~petersburg.graph.Graph.to_dict` produces exactly the dictionary
:meth:`~petersburg.graph.Graph.from_dict` consumes, so graphs with numeric weights
round-trip losslessly. Edges weighted by estimator objects (trained classifiers) are not
serializable and raise ``ValidationError`` on serialize.

.. code-block:: python

    reloaded = Graph().from_dict(g.to_dict())
    assert reloaded.to_dict() == g.to_dict()

:meth:`~petersburg.graph.Graph.to_json` and :meth:`~petersburg.graph.Graph.from_json` do the
same through a JSON string (a ``petersburg-graph`` version 1 document with a node list), so
integer and string ids keep their type. Reading and writing files is left to the caller.

.. code-block:: python

    text = g.to_json(indent=2)
    assert Graph().from_json(text).to_dict() == g.to_dict()

The draft 2020-12 `JSON Schema for version 1
<https://github.com/wdm0006/petersburg/blob/master/petersburg/schemas/petersburg-graph-v1.schema.json>`_
is shipped as ``petersburg/schemas/petersburg-graph-v1.schema.json`` in the package.
It describes the explicit serializer output: ``format``, ``version``, and ``nodes``;
each node requires ``id``, ``type``, its type-specific payoff parameters, and ``after``.
Predecessor records require ``node_id``, ``cost``, and ``weight``; parallel edges are allowed.
The schema checks structure only. Acyclicity, a single start, unique ids, valid references,
finite values, and distribution constraints remain loader/node validation responsibilities.
Always use ``from_json`` to reconstruct a validated graph after a schema check.

External tools can use the schema file without importing Petersburg. Python callers can
locate the installed copy with the standard library:

.. code-block:: python

    from importlib.resources import files

    schema_text = files("petersburg").joinpath(
        "schemas/petersburg-graph-v1.schema.json"
    ).read_text(encoding="utf-8")

Prediction
----------

:class:`~petersburg.estimators.FrequencyEstimator` and
:class:`~petersburg.estimators.MixedModeEstimator` are scikit-learn style estimators that
learn a decision graph from historical paths (``y`` has one column per decision layer) and
predict outcomes through it. Both expose ``classes_`` and ``n_features_in_`` after ``fit``,
``predict()`` returning one fitted terminal label per row, ``predict_proba()`` with rows
summing to 1, and ``partial_fit()`` for incremental updates. Hyperparameters live in the
constructor, so ``get_params()``/``set_params()``/``clone()`` and tools like
``GridSearchCV`` work as expected:

.. code-block:: python

    from petersburg import MixedModeEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    est = MixedModeEstimator(min_samples=100, clf=LogisticRegression())
    est.fit(X, y)

    proba = est.predict_proba(X)  # shape (n_samples, n_classes); rows sum to 1
    pred = est.predict(X)         # shape (n_samples,)

    search = GridSearchCV(
        est, {"min_samples": [10, 50, 100], "clf__C": [0.1, 1.0, 10.0]}, cv=3
    )

Visualization
-------------

:meth:`~petersburg.graph.Graph.to_mermaid` exports the graph as a Mermaid diagram with no
extra dependencies. :meth:`~petersburg.graph.Graph.plot` renders an image and requires the
``[graphviz]`` extra (pygraphviz) plus a system Graphviz install, and networkx/matplotlib
from the ``[visualization]`` extra:

.. code-block:: python

    print(g.to_mermaid())
    g.plot("graph.png")  # needs pip install petersburg[graphviz] + system Graphviz
