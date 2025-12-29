"""
教師あり学習の search_method 引数のテスト.
"""

from __future__ import annotations

from typing import Any

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from automl_engine.ml.training.supervised import run_supervised


@pytest.fixture()
def sample_data() -> tuple[Any, Any]:
    """
    テスト用のデータセットを生成.

    Returns:
        特徴量とターゲットのタプル.
    """
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    return X, y


def test_search_method_grid(sample_data: tuple[Any, Any]) -> None:
    """
    search_method="grid" のテスト.

    Args:
        sample_data: テスト用データ.
    """
    X, y = sample_data

    # アルゴリズム定義（constants/algorithms.py の構造に従う）
    algorithms = {
        "logreg": {
            "estimator_cls": LogisticRegression,
            "init_params": {
                "max_iter": 1000,
                "random_state": 42,
            },
            "search_space": {
                "grid": {
                    "C": [0.1, 1.0],
                    "penalty": ["l2"],
                },
                "optuna": {},
            },
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

    # 実行
    estimators, results = run_supervised(
        X=X,
        y=y,
        algorithms=algorithms,
        metrics=metrics,
        primary_metric_key="accuracy",
        cv=cv,
        search_method="grid",
        n_jobs=1,
        n_jobs_cv=1,
        parallel_backend="joblib",
    )

    # 結果の検証
    assert len(estimators) == 1
    assert "logreg" in estimators
    assert "logreg" in results
    assert results["logreg"]["search_method"] == "grid"
    assert "best_params" in results["logreg"]
    assert "cv_scores_mean" in results["logreg"]
    assert "accuracy" in results["logreg"]["cv_scores_mean"]


def test_search_method_optuna(sample_data: tuple[Any, Any]) -> None:
    """
    search_method="optuna" のテスト.

    Args:
        sample_data: テスト用データ.
    """
    X, y = sample_data

    # アルゴリズム定義
    algorithms = {
        "rf": {
            "estimator_cls": RandomForestClassifier,
            "init_params": {
                "random_state": 42,
                "n_jobs": 1,
            },
            "search_space": {
                "grid": {},
                "optuna": {
                    "n_estimators": {"type": "int", "low": 10, "high": 50},
                    "max_depth": {"type": "int_or_none", "low": 2, "high": 10},
                },
            },
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

    # 実行
    estimators, results = run_supervised(
        X=X,
        y=y,
        algorithms=algorithms,
        metrics=metrics,
        primary_metric_key="accuracy",
        cv=cv,
        search_method="optuna",
        optuna_trials=3,
        n_jobs=1,
        n_jobs_cv=1,
        parallel_backend="joblib",
    )

    # 結果の検証
    assert len(estimators) == 1
    assert "rf" in estimators
    assert "rf" in results
    assert results["rf"]["search_method"] == "optuna"
    assert "best_params" in results["rf"]
    assert "best_value" in results["rf"]
    assert "cv_scores_mean" in results["rf"]


def test_search_method_none(sample_data: tuple[Any, Any]) -> None:
    """
    search_method=None のテスト（デフォルトパラメータのみで実行）.

    Args:
        sample_data: テスト用データ.
    """
    X, y = sample_data

    # アルゴリズム定義
    algorithms = {
        "logreg": {
            "estimator_cls": LogisticRegression,
            "init_params": {
                "max_iter": 1000,
                "random_state": 42,
                "C": 1.0,
            },
            "search_space": {
                "grid": {
                    "C": [0.1, 1.0, 10.0],
                },
                "optuna": {},
            },
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

    # 実行
    estimators, results = run_supervised(
        X=X,
        y=y,
        algorithms=algorithms,
        metrics=metrics,
        primary_metric_key="accuracy",
        cv=cv,
        search_method=None,
        n_jobs=1,
        n_jobs_cv=1,
        parallel_backend="joblib",
    )

    # 結果の検証
    assert len(estimators) == 1
    assert "logreg" in estimators
    assert "logreg" in results
    assert results["logreg"]["search_method"] == "default"
    assert "cv_scores_mean" in results["logreg"]
    assert "accuracy" in results["logreg"]["cv_scores_mean"]
    # search_method=None の場合は best_params がないことを確認
    assert "best_params" not in results["logreg"]


def test_multiple_algorithms_with_different_search_spaces(
    sample_data: tuple[Any, Any],
) -> None:
    """
    複数アルゴリズムで異なる search_space を持つ場合のテスト.

    Args:
        sample_data: テスト用データ.
    """
    X, y = sample_data

    # アルゴリズム定義
    algorithms = {
        "logreg": {
            "estimator_cls": LogisticRegression,
            "init_params": {
                "max_iter": 1000,
                "random_state": 42,
            },
            "search_space": {
                "grid": {
                    "C": [0.1, 1.0],
                },
                "optuna": {},
            },
        },
        "rf": {
            "estimator_cls": RandomForestClassifier,
            "init_params": {
                "random_state": 42,
                "n_jobs": 1,
            },
            "search_space": {
                "grid": {
                    "n_estimators": [10, 20],
                },
                "optuna": {},
            },
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

    # 実行
    estimators, results = run_supervised(
        X=X,
        y=y,
        algorithms=algorithms,
        metrics=metrics,
        primary_metric_key="accuracy",
        cv=cv,
        search_method="grid",
        n_jobs=2,
        n_jobs_cv=1,
        parallel_backend="joblib",
    )

    # 結果の検証
    assert len(estimators) == 2
    assert "logreg" in estimators
    assert "rf" in estimators
    assert results["logreg"]["search_method"] == "grid"
    assert results["rf"]["search_method"] == "grid"
