# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))
sys.path.insert(0, os.path.abspath('_ext'))
from digeo import __version__ as digeo_version

project = 'DiGeo'
copyright = '2026, DiGeo Developers'
author = 'DiGeo Developers'
release = digeo_version

autodoc_mock_imports = ["digeo.ops.cuda"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',   # Google/NumPy docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_remove_toctrees',
    "sphinx_design",
    'roles',
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
}

autosummary_generate = True
autosummary_imported_members = True
autoclass_content = "init"
autodoc_typehints = "description"
templates_path = ['_templates']
exclude_patterns = []

html_sidebars = {
    "install": [],
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_title = "DiGeo Documentation"

html_theme_options = {
    "sidebar_includehidden": True,
    "use_edit_page_button": True,
    "external_links": [],
    "icon_links_label": "Icon Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/circle-group/DiGeo",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "show_prev_next": False,
    "search_bar_text": "Search the docs ...",
    "navigation_with_keys": False,
    "collapse_navigation": False,
    "navigation_depth": 2,
    "show_nav_level": 1,
    "show_toc_level": 1,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "header_dropdown_text": "More",
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "logo": {
        "image_light": "digeo.svg",
        "image_dark": "digeo.svg",
    },
    "surface_warnings": True,
    # -- Template placement in theme layouts ----------------------------------
    "navbar_start": ["navbar-logo"],
    # Note that the alignment of navbar_center is controlled by navbar_align
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # navbar_persistent is persistent right (even when on mobiles)
    "navbar_persistent": ["search-button"],
    "article_header_start": ["breadcrumbs"],
    "article_header_end": [],
    "article_footer_items": ["prev-next"],
    "content_footer_items": [],
    # Use html_sidebars that map page patterns to list of sidebar templates
    "primary_sidebar_end": [],
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": [],
    "secondary_sidebar_items": {
        "**": [
            "page-toc",
            "sourcelink",
        ],
    },
    "show_version_warning_banner": True,
    "announcement": None,
}

html_static_path = ['_static']
html_css_files = ['theme-overrides.css']
html_favicon = '_static/icon.svg'
