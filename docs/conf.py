import os
import sys

# Path setup: add src directory to sys.path
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'FROG'
copyright = '2026, Andrea Pellegrini, Paolo Righi'
author = 'Andrea Pellegrini, Paolo Righi'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_parser',
    'sphinx_rtd_theme',
    'sphinx_design',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
                    # hand-written fragments included by the generated CLI pages
                    'cli/_parts/**', 'cli/_examples/**']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

root_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': True,
}

# -- MyST Parser configuration -----------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "substitution",
    "deflist",
    "tasklist",
]

# 'fieldlist' exists only in myst-parser >= 0.18; ReadTheDocs installs >= 2.0 (docs/requirements.txt).
# An older local install (e.g. Homebrew's myst-parser 0.15) still builds, with a clear warning.
try:
    import myst_parser as _myst
    _myst_version = tuple(int(x) for x in _myst.__version__.split(".")[:2])
except Exception:
    _myst_version = (0, 0)
if _myst_version >= (0, 18):
    myst_enable_extensions.append("fieldlist")
if _myst_version < (2, 0):
    import warnings as _warnings
    _warnings.warn(f"myst-parser {_myst.__version__} is older than the 2.0 required by docs/requirements.txt; "
                   "build with: python -m pip install -r docs/requirements.txt", stacklevel=1)
myst_heading_anchors = 3

# Mock imports for heavy optional dependencies during build
autodoc_mock_imports = [
    'torch',
    'aimnet',
    'mace',
    'tblite',
    'xtb',
]
