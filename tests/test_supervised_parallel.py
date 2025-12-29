"""
教師あり学習の並列実行機能のテスト.
"""

from __future__ import annotations

from typing import Any

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

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


def test_run_supervised_parallel_joblib(sample_data: tuple[Any, Any]) -> None:
    """
    joblib バックエンドでの並列実行テスト.

    Args:
        sample_data: テスト用データ.
    """
    X, y = sample_data

    # アルゴリズム定義
    algorithms = {
        "LogisticRegression": {
            "estimator_cls": lambda: LogisticRegression(max_iter=1000, random_state=42)
        },
        "RandomForest": {
            "estimator_cls": lambda: RandomForestClassifier(n_estimators=10, random_state=42)
        },
        "DecisionTree": {
            "estimator_cls": lambda: DecisionTreeClassifier(random_state=42)
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
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    # 並列実行
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
    assert len(estimators) == 3
    assert len(results) == 3
    assert "LogisticRegression" in estimators
    assert "RandomForest" in estimators
    assert "DecisionTree" in estimators

    # 各結果に必要な情報が含まれているか確認
    for key in algorithms.keys():
        assert key in results
        assert "search_method" in results[key]
        assert "cv_scores_mean" in results[key]
        assert "accuracy" in results[key]["cv_scores_mean"]


def test_run_supervised_parallel_concurrent(sample_data: tuple[Any, Any]) -> None:
    """
    concurrent.futures バックエンドでの並列実行テスト.

    Args:
        sample_data: テスト用データ.
    """
    X, y = sample_data

    # アルゴリズム定義
    algorithms = {
        "LogisticRegression": {
            "estimator_cls": lambda: LogisticRegression(max_iter=1000, random_state=42)
        },
        "RandomForest": {
            "estimator_cls": lambda: RandomForestClassifier(n_estimators=10, random_state=42)
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

    # 並列実行
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
        parallel_backend="concurrent",
    )

    # 結果の検証
    assert len(estimators) == 2
    assert len(results) == 2


def test_run_supervised_n_jobs_auto(sample_data: tuple[Any, Any]) -> None:
    """
    n_jobs=-1（全コア使用）での並列実行テスト.

    Args:
        sample_data: テスト用データ.
    """
    X, y = sample_data

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

    # 並列実行（n_jobs=-1 でデフォルト）
    estimators, results = run_supervised(
        X=X,
        y=y,
        algorithms=algorithms,
        metrics=metrics,
        primary_metric_key="accuracy",
        cv=cv,
        search_method="grid",
        parallel_backend="joblib",
    )

    # 結果の検証
    assert len(estimators) == 1
    assert "LogisticRegression" in estimators


def test_run_supervised_n_jobs_cv_auto(sample_data: tuple[Any, Any]) -> None:
    """
    n_jobs_cv=None（自動計算）での並列実行テスト.

    Args:
        sample_data: テスト用データ.
    """
    X, y = sample_data

    # アルゴリズム定義
    algorithms = {
        "DecisionTree": {
            "estimator_cls": lambda: DecisionTreeClassifier(random_state=42)
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

    # 並列実行（n_jobs_cv=None で自動計算）
    estimators, results = run_supervised(
        X=X,
        y=y,
        algorithms=algorithms,
        metrics=metrics,
        primary_metric_key="accuracy",
        cv=cv,
        search_method="grid",
        n_jobs=2,
        n_jobs_cv=None,
        parallel_backend="joblib",
    )

    # 結果の検証
    assert len(estimators) == 1


def test_run_supervised_with_timeout(sample_data: tuple[Any, Any]) -> None:
    """
    タイムアウト付き並列実行テスト.

    Args:
        sample_data: テスト用データ.
    """
    X, y = sample_data

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

    # 並列実行（長めのタイムアウト）
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
        algorithm_timeout=60,
    )

    # 結果の検証
    assert len(estimators) == 1
