"""Sphinx configuration; import the installed package for API documentation."""

from importlib.metadata import version as package_version

project = "PyFCIRT"
author = "PyFCIRT contributors"
release = package_version("pyfcirt")
version = release
extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["pyfcirt.css"]
html_logo = "_static/pyfcirt-logo.svg"
html_favicon = "_static/pyfcirt-logo.svg"
html_context = {"default_mode": "light"}
# Landing pages use the full content area; guides retain native section navigation.
html_sidebars = {
    "index": [],
    "examples": [],
    "guide/getting_started": [],
}
html_theme_options = {
    "logo": {"text": "PyFCIRT"},
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/qipengchen/pyfcirt",
            "icon": "fa-brands fa-github",
        }
    ],
    "navigation_depth": 3,
    "collapse_navigation": False,
    "show_toc_level": 2,
    "show_nav_level": 1,
    "header_links_before_dropdown": 4,
    "navbar_align": "right",
    "article_header_start": [],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],
}
html_title = "PyFCIRT"
exclude_patterns = [
    "_build",
    ".ipynb_checkpoints",
    "**/.ipynb_checkpoints",
    "contributing.md",
    "architecture.md",
    "requirements.md",
    "testing.md",
    "recovery.md",
    "tutorial_migration.md",
    "roadmap.md",
    "reference/new_feature_requirements.md",
    "reference/simulation_requirements.md",
    "examples/analysis_previews.md",
    "tutorials/tirt_groups.ipynb",
    "models/extensions.md",
    "models/statistical_audit.md",
    "models/thurstonian_roadmap.md",
]
autodoc_member_order = "bysource"
# Physical-parameter pages must not inherit generic PyMC distribution prose.
autodoc_inherit_docstrings = False
# Reference pages are authored once in Markdown; autosummary supplies their
# linked indices without generating a second .rst stub for the same document.
autosummary_generate = False
autodoc_typehints = "signature"
python_maximum_signature_line_length = 88
# Read real signatures; dataclass-generated docstrings quote postponed annotations.
autodoc_docstring_signature = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = False
napoleon_type_aliases = {
    "optional": "``optional``",
    "mapping": "``mapping``",
    "Label": "``str | int``",
    "pd.DataFrame": "pandas.DataFrame",
    "xr.DataTree": "xarray.DataTree",
    "xr.Dataset": "xarray.Dataset",
    "xr.DataArray": "xarray.DataArray",
    "Path": "pathlib.Path",
}
nb_execution_mode = "off"
nb_execution_raise_on_error = True
nb_output_stderr = "remove"
nb_execution_timeout = 600
myst_heading_anchors = 4
myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "pytensor": ("https://pytensor.readthedocs.io/en/latest/", None),
}
intersphinx_timeout = 10


def canonical_type_references(app, doctree):
    """Resolve known annotation aliases to documented public classes.

    Autodoc can expand xarray annotations to implementation-module names. Keep
    their links on the public API; unresolved canonical targets remain errors.
    """
    from docutils import nodes
    from sphinx import addnodes

    aliases = {
        "pd.DataFrame": "pandas.DataFrame",
        "xr.DataTree": "xarray.DataTree",
        "xr.Dataset": "xarray.Dataset",
        "xr.DataArray": "xarray.DataArray",
        "Path": "pathlib.Path",
        "pm.Model": "pymc.model.core.Model",
        "TensorVariable": "pytensor.tensor.TensorVariable",
        "pytensor.tensor.variable.TensorVariable": "pytensor.tensor.TensorVariable",
        "RegressionOptions": "pyfcirt.RegressionOptions",
        "xarray.core.dataset.Dataset": "xarray.Dataset",
        "xarray.core.dataarray.DataArray": "xarray.DataArray",
        "xarray.core.datatree.DataTree": "xarray.DataTree",
    }
    literals = {
        "optional": "optional",
        "mapping": "mapping",
        "Label": "str | int",
    }
    for node in doctree.findall(addnodes.pending_xref):
        if node.get("refdomain") != "py":
            continue
        target = node.get("reftarget")
        if (
            target == "sample"
            and node.get("reftype") == "func"
            and node.get("py:module") == "pyfcirt.inference"
        ):
            node.replace_self(nodes.literal("", node.astext()))
        elif target in literals:
            node.replace_self(nodes.literal("", literals[target]))
        elif target in aliases:
            node["reftarget"] = aliases[target]


def keep_reference_signature(app, what, name, obj, options, lines):
    """Use source signatures and the API pages' edited prose."""
    if app.env.docname.startswith("reference/generated/"):
        lines.clear()


def dataclass_factory_signature(app, what, name, obj, options, signature, return_annotation):
    """Keep generated factory defaults parseable by Sphinx's native signature parser."""
    from dataclasses import is_dataclass

    if what == "class" and is_dataclass(obj) and signature and "<factory>" in signature:
        # The actual factory defaults are documented in each parameter section.
        return signature.replace("<factory>", "..."), return_annotation
    return None


def setup(app):
    app.connect("doctree-read", canonical_type_references)
    app.connect("autodoc-process-docstring", keep_reference_signature)
    app.connect("autodoc-process-signature", dataclass_factory_signature)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
