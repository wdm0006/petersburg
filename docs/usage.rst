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

Serialization
-------------

:meth:`~petersburg.graph.Graph.to_dict` produces exactly the dictionary
:meth:`~petersburg.graph.Graph.from_dict` consumes, so graphs with numeric weights
round-trip losslessly. Edges weighted by estimator objects (trained classifiers) are not
serializable and raise ``ValidationError`` on serialize.

.. code-block:: python

    reloaded = Graph().from_dict(g.to_dict())
    assert reloaded.to_dict() == g.to_dict()

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
