"""
.. module:: exceptions
   :platform: Unix, Windows
   :synopsis:

.. moduleauthor:: Will McGinnis <will@pedalwrencher.com>


"""

__author__ = "willmcginnis"


class PetersburgError(Exception):
    """
    Base class for every error petersburg's public API raises on invalid use.

    Catch this one type to handle any petersburg validation failure without
    over-matching unrelated errors raised by caller code.
    """


class ValidationError(PetersburgError, ValueError):
    """
    Raised when an input value is invalid: adjacency matrices, sensitivity
    arguments, estimator targets and feature matrices, node payoffs, and
    transition weights.

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers keep
    working unchanged; new code can catch ``PetersburgError`` (or this class)
    instead.
    """


class SpecValidationError(ValidationError, AttributeError):
    """
    Raised when a graph specification is invalid: unrecognized node types,
    references to unknown nodes, cycles, missing or multiple starting nodes,
    or non-dict specifications.

    Graph specifications historically raised a bare ``AttributeError``. This
    subclass carries the petersburg hierarchy while remaining an
    ``AttributeError``, so callers (and tests) that caught the old error keep
    working unchanged. New code should catch ``PetersburgError`` or
    ``ValidationError`` rather than ``AttributeError``.
    """
