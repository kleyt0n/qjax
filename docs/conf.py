"""Sphinx configuration for the qjax documentation."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the package importable for autodoc (it is also installed in the venv).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qjax  # noqa: E402

# -- Project information -----------------------------------------------------
project = "qjax"
author = "Kleyton Costa"
copyright = f"{datetime.now(tz=timezone.utc):%Y}, {author}"
release = qjax.__version__
version = qjax.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST (Markdown) ---------------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath", "deflist", "colon_fence"]
myst_heading_anchors = 3

# -- Autodoc / Napoleon ------------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output (Furo) ------------------------------------------------------
html_theme = "furo"
html_title = f"qjax {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    # Pastel-orange accent taken from the qjax logo (#ffbf80). On the light
    # theme we use the logo's deeper orange (#f5a55a) so links stay legible on
    # white; the pastel #ffbf80 reads well on the dark theme.
    "light_css_variables": {
        "color-brand-primary": "#f5a55a",
        "color-brand-content": "#f5a55a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ffbf80",
        "color-brand-content": "#ffbf80",
    },
}
