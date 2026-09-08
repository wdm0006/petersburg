"""
.. module:: petersburg
   :platform: Unix, Windows
   :synopsis:

.. moduleauthor:: Will McGinnis <will@pedalwrencher.com>


"""

import importlib.metadata

from petersburg.edges import Edge
from petersburg.estimators import FrequencyEstimator, MixedModeEstimator
from petersburg.exceptions import PetersburgError, SpecValidationError, ValidationError
from petersburg.graph import Graph
from petersburg.nodes import GaussianNode, LogNormalNode, Node, PowerLawNode, UniformNode

try:
    __version__ = importlib.metadata.version("petersburg")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    # Package not installed (e.g. imported from a raw source checkout). Kept here rather
    # than in a __version__.py module, whose name a string attribute would shadow.
    __version__ = "unknown"

__all__ = [
    "__version__",
    "Node",
    "UniformNode",
    "GaussianNode",
    "LogNormalNode",
    "PowerLawNode",
    "MixedModeEstimator",
    "Graph",
    "Edge",
    "FrequencyEstimator",
    "PetersburgError",
    "ValidationError",
    "SpecValidationError",
]
