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

templates_path = ["_templates"]
source_suffix = [".rst", ".ipynb"]
master_doc = "index"

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
github_nb_repo = "scvelo_notebooks"
html_static_path = ["_static"]


# TODO: Add docstrings
def setup(app):
    """TODO."""
    app.add_css_file("custom.css")


# -- Options for other output ------------------------------------------

htmlhelp_basename = "causalegmdoc"
title_doc = f"{project} documentation"

latex_documents = [(master_doc, f"{project}.tex", title_doc, author, "manual")]
man_pages = [(master_doc, project, title_doc, [author], 1)]
texinfo_documents = [
    (master_doc, project, title_doc, author, project, title, "Miscellaneous")
]


# -- GitHub URLs for class and method pages ------------------------------------------

# TODO: Finish docstrings
def get_obj_module(qualname):
    """Get a module/class/attribute and its original module by qualname."""
    modname = qualname
    classname = None
    attrname = None
    while modname not in sys.modules:
        attrname = classname
        modname, classname = modname.rsplit(".", 1)

    # retrieve object and find original module name
    if classname:
        cls = getattr(sys.modules[modname], classname)
        modname = cls.__module__
        obj = getattr(cls, attrname) if attrname else cls
    else:
        obj = None

    return obj, sys.modules[modname]


# TODO: Finish docstrings
def get_linenos(obj):
    """Get an object’s line numbers."""
    try:
        lines, start = inspect.getsourcelines(obj)
    except TypeError:
        return None, None
    else:
        return start, start + len(lines) - 1


# set project_dir: project/docs/source/conf.py/../../.. → project/
project_dir = Path(__file__).parent.parent.parent
github_url_scvelo = "https://github.com/theislab/scvelo/tree/master"
github_url_read_loom = "https://github.com/theislab/anndata/tree/master/anndata"
github_url_read = "https://github.com/theislab/scanpy/tree/master"
github_url_scanpy = "https://github.com/theislab/scanpy/tree/master/scanpy"


# TODO: Finish docstrings
def modurl(qualname):
    """Get the full GitHub URL for some object’s qualname."""
    obj, module = get_obj_module(qualname)
    github_url = github_url_scvelo
    try:
        path = PurePosixPath(Path(module.__file__).resolve().relative_to(project_dir))
    except ValueError:
        # trying to document something from another package
        github_url = (
            github_url_read_loom
            if "read_loom" in qualname
            else github_url_read
            if "read" in qualname
            else github_url_scanpy
        )
        path = "/".join(module.__file__.split("/")[-2:])
    start, end = get_linenos(obj)
    fragment = f"#L{start}-L{end}" if start and end else ""
    return f"{github_url}/{path}{fragment}"


# TODO: Add docstrings
def api_image(qualname: str) -> Optional[str]:
    """TODO."""
    path = Path(__file__).parent / f"{qualname}.png"
    print(path, path.is_file())
    return (
        f".. image:: {path.name}\n   :width: 200\n   :align: right"
        if path.is_file()
        else ""
    )


# modify the default filters

DEFAULT_FILTERS.update(modurl=modurl, api_image=api_image)

# -- Override some classnames in autodoc --------------------------------------------

qualname_overrides = {
    "anndata.base.AnnData": "anndata.AnnData",
    "scvelo.pl.scatter": "scvelo.plotting.scatter",
}

fa_orig = sphinx_autodoc_typehints.format_annotation


# TODO: Add docstrings
def format_annotation(annotation):
    """TODO."""
    if getattr(annotation, "__origin__", None) is Union or hasattr(
        annotation, "__union_params__"
    ):
        params = getattr(annotation, "__union_params__", None) or getattr(
            annotation, "__args__", None
        )
        return ", ".join(map(format_annotation, params))
    if getattr(annotation, "__origin__", None) is Mapping:
        return ":class:`~typing.Mapping`"
    if inspect.isclass(annotation):
        full_name = f"{annotation.__module__}.{annotation.__qualname__}"
        override = qualname_overrides.get(full_name)
        if override is not None:
            return f":py:class:`~{qualname_overrides[full_name]}`"
    return fa_orig(annotation)


sphinx_autodoc_typehints.format_annotation = format_annotation


# -- Prettier Param docs --------------------------------------------


# TODO: Add docstrings
class PrettyTypedField(PyTypedField):
    """TODO."""

    list_type = nodes.definition_list

    # TODO: Add docstrings
    def make_field(
        self,
        types: Dict[str, List[nodes.Node]],
        domain: str,
        items: Tuple[str, List[nodes.inline]],
        env: BuildEnvironment = None,
    ) -> nodes.field:
        """TODO."""

        def makerefs(rolename, name, node):
            return self.make_xrefs(rolename, domain, name, node, env=env)

        def handle_item(
            fieldarg: str, content: List[nodes.inline]
        ) -> nodes.definition_list_item:
            head = nodes.term()
            head += makerefs(self.rolename, fieldarg, addnodes.literal_strong)
            fieldtype = types.pop(fieldarg, None)
            if fieldtype is not None:
                head += nodes.Text(" : ")
                if len(fieldtype) == 1 and isinstance(fieldtype[0], nodes.Text):
                    (text_node,) = fieldtype  # type: nodes.Text
                    head += makerefs(
                        self.typerolename, text_node.astext(), addnodes.literal_emphasis
                    )
                else:
                    head += fieldtype

            body_content = nodes.paragraph("", "", *content)
            body = nodes.definition("", body_content)

            return nodes.definition_list_item("", head, body)

        fieldname = nodes.field_name("", self.label)
        if len(items) == 1 and self.can_collapse:
            fieldarg, content = items[0]
            bodynode = handle_item(fieldarg, content)
        else:
            bodynode = self.list_type()
            for fieldarg, content in items:
                bodynode += handle_item(fieldarg, content)
        fieldbody = nodes.field_body("", bodynode)
        return nodes.field("", fieldname, fieldbody)


# replace matching field types with ours
PyObject.doc_field_types = [
    PrettyTypedField(
        ft.name,
        names=ft.names,
        typenames=ft.typenames,
        label=ft.label,
        rolename=ft.rolename,
        typerolename=ft.typerolename,
        can_collapse=ft.can_collapse,
    )
    if isinstance(ft, PyTypedField)
    else ft
    for ft in PyObject.doc_field_types
]