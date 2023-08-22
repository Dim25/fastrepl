"""Configuration for sphinx."""
# Configuration file for the Sphinx documentation builder.
# P
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import importlib.metadata

version = importlib.metadata.version("fastrepl")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "♾️ FastREPL"
copyright = "2023, yujonglee"
author = "yujonglee"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "m2r2",
    "myst_nb",
    "sphinxcontrib.autodoc_pydantic",
]

myst_heading_anchors = 4
# suppress_warnings = ["myst.header"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = project + " " + version
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
    "css/algolia.css",
    "https://cdn.jsdelivr.net/npm/@docsearch/css@3",
]
html_js_files = [
    (
        "https://cdn.jsdelivr.net/npm/@docsearch/js@3.3.3/dist/umd/index.js",
        {"defer": "defer"},
    ),
    ("js/algolia.js", {"defer": "defer"}),
]

nb_execution_mode = "off"
