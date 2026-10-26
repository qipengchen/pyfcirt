"""Sphinx configuration; import the installed package for API documentation."""

from importlib.metadata import version as package_version

project = "PyFCIRT"
author = "PyFCIRT contributors"
release = package_version("pyfcirt")
version = release
extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.intersphinx"]
html_theme = "alabaster"
exclude_patterns = ["_build", ".ipynb_checkpoints", "**/.ipynb_checkpoints"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
nb_execution_mode = "off"
nb_execution_raise_on_error = True
nb_execution_timeout = 600
myst_heading_anchors = 3
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
