import inspect
import logging
import os
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Tuple, Union
from urllib.error import URLError
from urllib.request import urlretrieve

import sphinx_autodoc_typehints
from docutils import nodes
from jinja2.defaults import DEFAULT_FILTERS
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.domains.python import PyObject, PyTypedField
from sphinx.environment import BuildEnvironment
from sphinx.ext import autosummary

import matplotlib

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, os.path.abspath("_ext"))


# remove PyCharm’s old six module

if "six" in sys.modules:
    print(*sys.path, sep="\n")
    for pypath in list(sys.path):
        if any(p in pypath for p in ["PyCharm", "pycharm"]) and "helpers" in pypath:
            sys.path.remove(pypath)
    del sys.modules["six"]

matplotlib.use("agg")

logger = logging.getLogger(__name__)


# -- General configuration ------------------------------------------------

needs_sphinx = "1.7"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "edit_on_github",
]


# Generate the API documentation when building
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False
napoleon_custom_sections = [("Params", "Parameters")]

intersphinx_mapping = dict(
    python=("https://docs.python.org/3", None),
    anndata=("https://anndata.readthedocs.io/en/latest/", None),
    scanpy=("https://scanpy.readthedocs.io/en/latest/", None),
    cellrank=("https://cellrank.readthedocs.io/en/latest/", None),
)


# General information about the project.
project = "CausalEGM"
author = "Qiao Liu"
title = "A general causal inference framework by encoding generative modeling"
copyright = f"{datetime.now():%Y}, {author}"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = dict(navigation_depth=1, titles_only=True)
github_repo = "CausalEGM"

source_suffix = [".rst", ".ipynb"]
master_doc = "index"
