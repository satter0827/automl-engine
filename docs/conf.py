from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

project = "automl_engine"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

html_theme = "sphinx_rtd_theme"

autosummary_generate = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

templates_path = ["_templates"]
html_static_path = ["_static"]
exclude_patterns = ["_build"]
