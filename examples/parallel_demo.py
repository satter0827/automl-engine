"""
並列実行のデモスクリプト.

このスクリプトは、run_supervisedメソッドの並列実行機能を示すサンプルです。
"""

import logging

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from automl_engine.ml.training.supervised import run_supervised

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main() -> None:
    """メイン関数."""
    print("=" * 80)
    print("run_supervised 並列実行デモ")
    print("=" * 80)

    # データの生成
    print("\n1. データセットを生成中...")
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )
    print(f"   データサイズ: X={X.shape}, y={y.shape}")

    # アルゴリズム定義
    print("\n2. アルゴリズムを定義中...")
    algorithms = {
        "LogisticRegression": {
            "estimator_cls": lambda: LogisticRegression(max_iter=1000, random_state=42)
        },
        "RandomForest": {
            "estimator_cls": lambda: RandomForestClassifier(
                n_estimators=50, random_state=42
            )
        },
        "DecisionTree": {
            "estimator_cls": lambda: DecisionTreeClassifier(random_state=42)
        },
    }
    print(f"   アルゴリズム数: {len(algorithms)}")
    print(f"   アルゴリズム: {list(algorithms.keys())}")

    # 評価指標定義
    print("\n3. 評価指標を定義中...")
    metrics = {
        "accuracy": {
            "scorer": {
                "score_func": accuracy_score,
                "greater_is_better": True,
            }
        },
        "f1": {
            "scorer": {
                "score_func": f1_score,
                "greater_is_better": True,
                "kwargs": {"average": "macro"},
            }
        },
    }
    print(f"   評価指標: {list(metrics.keys())}")

    # CV 定義
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 並列実行（joblib バックエンド）
    print("\n4. 並列実行中（joblib バックエンド）...")
    print("   n_jobs=2（アルゴリズム並列数）")
    print("   n_jobs_cv=1（CV並列数）")
    print("-" * 80)

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

    # 結果の表示
    print("\n5. 結果:")
    print("-" * 80)
    for algo_name, result in results.items():
        print(f"\n   アルゴリズム: {algo_name}")
        print(f"   探索手法: {result['search_method']}")
        if "cv_scores_mean" in result:
            print("   CV スコア (平均):")
            for metric_name, score in result["cv_scores_mean"].items():
                print(f"      {metric_name}: {score:.4f}")

    print("\n" + "=" * 80)
    print("デモ完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
