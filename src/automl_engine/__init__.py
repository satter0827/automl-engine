"""
AutoML Engine のトップレベルパッケージ.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__: list[str] = ["common", "ml"]


try:
    __version__ = version("automl-engine")
except PackageNotFoundError:
    __version__ = "0.0.0"
