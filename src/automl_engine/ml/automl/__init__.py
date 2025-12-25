"""
AutoML 実行パッケージ。
タスク種別ごとの探索・学習・選定を統括するエントリポイントを提供する。
"""

from __future__ import annotations

from . import supervised

__all__: list[str] = ["supervised"]
