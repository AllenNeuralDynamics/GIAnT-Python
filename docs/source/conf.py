"""Configuration file for the Sphinx documentation builder."""

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date

# -- Path Setup --------------------------------------------------------------
from os.path import abspath, dirname
from pathlib import Path

from giant_python import __version__ as package_version

INSTITUTE_NAME = "Allen Institute for Neural Dynamics"

current_year = date.today().year

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = Path(dirname(dirname(dirname(abspath(__file__))))).name
copyright = f"{current_year}, {INSTITUTE_NAME}"
author = INSTITUTE_NAME
release = package_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]
templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}

napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "Path": "pathlib.Path",
    "ndarray": "numpy.ndarray",
    "TrialTable": "giant_python.models.trial_table.TrialTable",
    "Slap2Info": "giant_python.models.trial_table.Slap2Info",
    "BandSiloParams": "giant_python.models.params.BandSiloParams",
    "ExecutionOptions": "giant_python.models.params.ExecutionOptions",
    "AnnotationOptions": "giant_python.models.params.AnnotationOptions",
    "ExperimentSummary": (
        "giant_python.models.experiment_summary.ExperimentSummary"
    ),
}


def _resolve_public_type_alias(app, env, node, contnode):
    """Resolve h5py's runtime class name through its public documentation."""
    if (
        node.get("refdomain") == "py"
        and node.get("reftarget") == "h5py._hl.group.Group"
    ):
        node["reftarget"] = "h5py.Group"


def setup(app):
    """Normalize private type names before intersphinx resolves references."""
    app.connect("missing-reference", _resolve_public_type_alias, priority=400)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "light_logo": "light-logo.svg",
    "dark_logo": "dark-logo.svg",
}

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False
