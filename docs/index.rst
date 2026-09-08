.. petersburg documentation master file

petersburg documentation
========================

A framework for analyzing probabilistic decision processes as directed graphs with
automatic sensitivity analysis.

petersburg models a decision problem as a directed acyclic graph whose nodes carry
payoffs and whose edges carry costs and transition weights. Monte Carlo simulation
over the graph yields expected outcomes; sensitivity analysis identifies which
parameters matter most; scikit-learn style estimators can learn the graph from
historical paths.

.. toctree::
   :maxdepth: 2

   usage
   api

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
