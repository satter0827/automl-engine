"""
教師あり学習の後方互換性テスト.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from automl_engine.ml.training.supervised import run_supervised


def test_run_supervised_backward_compatibility() -> None:
    """
    既存のAPIとの後方互換性を確認するテスト.
    新しいパラメータを指定せずに呼び出して、既存の動作が維持されていることを確認.
    """
    # データの生成
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )

    # アルゴリズム定義
    algorithms = {
        "LogisticRegression": {
            "estimator_cls": lambda: LogisticRegression(max_iter=1000, random_state=42)
        },
    }

    # 評価指標定義
    metrics = {
        "accuracy": {
            "scorer": {
                "score_func": accuracy_score,
                "greater_is_better": True,
            }
        }
    }

    # CV 定義
    cv = KFold(n_splits=2, shuffle=True, random_state=42)

    # 既存のAPIで実行（新しいパラメータを指定しない）
    estimators, results = run_supervised(
        X=X,
        y=y,
        algorithms=algorithms,
        metrics=metrics,
        primary_metric_key="accuracy",
        cv=cv,
        search_method="grid",
    )

    # 結果の検証
    assert len(estimators) == 1
    assert "LogisticRegression" in estimators
    assert len(results) == 1
    assert "LogisticRegression" in results
    assert "search_method" in results["LogisticRegression"]
    assert results["LogisticRegression"]["search_method"] == "grid"
    assert "cv_scores_mean" in results["LogisticRegression"]
    assert "accuracy" in results["LogisticRegression"]["cv_scores_mean"]
