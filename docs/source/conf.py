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


logger = logging.getLogger(__name__)


# -- General configuration ------------------------------------------------

needs_sphinx = "1.7"

extensions = [
    "sphinx.ext.autodoc",
]




# General information about the project.
project = "CausalEGM"
author = "Qiao Liu"
title = "A general causal inference framework by encoding generative modeling"
copyright = f"{datetime.now():%Y}, {author}"




html_theme = "sphinx_rtd_theme"
