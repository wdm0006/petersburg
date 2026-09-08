"""Sphinx configuration for the petersburg documentation."""

import os
import sys

# Work from a raw source checkout too (autodoc imports the package).
sys.path.insert(0, os.path.abspath(".."))

project = "petersburg"
author = "Will McGinnis"
copyright = "2026, Will McGinnis"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

autodoc_member_order = "bysource"

# scikit-learn's inherited metadata-routing docstrings (pulled in through the estimator
# base classes) cross-reference terms and labels that exist only in sklearn's own docs.
# These subtypes are otherwise unused by this scaffold.
suppress_warnings = ["ref.term", "ref.ref"]

html_theme = "sphinx_rtd_theme"
