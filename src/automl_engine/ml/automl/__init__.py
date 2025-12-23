"""
AutoML 実行パッケージ。
タスク種別ごとの探索・学習・選定を統括するエントリポイントを提供する。
"""

from __future__ import annotations

from . import template

__all__: list[str] = ["template"]
