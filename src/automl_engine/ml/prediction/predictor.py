"""
教師あり学習モデルの推論処理を提供するモジュール.
"""

from __future__ import annotations

from typing import Any, Optional

from sklearn.base import BaseEstimator

_PRIMARY_SCORE_KEY = "primary_score_mean"
_BEST_SCORE_KEY = "best_score"
_CV_SCORES_KEY = "cv_scores_mean"


def _extract_primary_score(
    info: dict[str, Any],
    primary_metric_key: str,
) -> Optional[float]:
    """
    評価情報から主評価指標のスコアを抽出する.

    Args:
        info: 評価情報.
        primary_metric_key: 主評価指標キー.

    Returns:
        Optional[float]: 抽出できたスコア。存在しない場合は None.
    """
    if _PRIMARY_SCORE_KEY in info:
        score = info[_PRIMARY_SCORE_KEY]
        if isinstance(score, (int, float)):
            return float(score)

    cv_scores = info.get(_CV_SCORES_KEY)
    if isinstance(cv_scores, dict) and primary_metric_key in cv_scores:
        score = cv_scores[primary_metric_key]
        if isinstance(score, (int, float)):
            return float(score)

    if _BEST_SCORE_KEY in info:
        score = info[_BEST_SCORE_KEY]
        if isinstance(score, (int, float)):
            return float(score)

    return None


def _select_best_model_key(
    estimators: dict[str, BaseEstimator],
    model_info: dict[str, dict[str, Any]],
    primary_metric_key: str,
) -> str:
    """
    最良モデルのキーを決定する.

    Args:
        estimators: 学習済み推定器.
        model_info: 評価情報.
        primary_metric_key: 主評価指標キー.

    Returns:
        str: 最良モデルのキー.

    Raises:
        RuntimeError: モデルが存在しない場合.
    """
    if not estimators:
        raise RuntimeError("学習済みモデルが存在しません。")

    scored_keys: list[tuple[str, float]] = []
    for key, info in model_info.items():
        if key not in estimators:
            continue
        score = _extract_primary_score(info, primary_metric_key)
        if score is not None:
            scored_keys.append((key, score))

    if scored_keys:
        scored_keys.sort(key=lambda item: item[1], reverse=True)
        return scored_keys[0][0]

    return sorted(estimators.keys())[0]


def predict_supervised(
    X: Any,
    *,
    estimators: dict[str, BaseEstimator],
    model_info: dict[str, dict[str, Any]],
    primary_metric_key: str,
    algorithm: Optional[str] = None,
) -> Any:
    """
    教師あり学習モデルで予測を実行する.

    Args:
        X: 予測対象データ.
        estimators: 学習済み推定器.
        model_info: 評価情報.
        primary_metric_key: 主評価指標キー.
        algorithm: 使用するアルゴリズムキー（省略時は最良モデル）.

    Returns:
        Any: 予測結果.

    Raises:
        RuntimeError: 学習済みモデルが存在しない場合.
        ValueError: 指定アルゴリズムが存在しない場合.
    """
    if not estimators:
        raise RuntimeError("学習済みモデルが存在しません。")

    if algorithm is not None:
        if algorithm not in estimators:
            raise ValueError(f"指定されたアルゴリズムが存在しません: {algorithm}")
        estimator = estimators[algorithm]
    else:
        best_key = _select_best_model_key(
            estimators=estimators,
            model_info=model_info,
            primary_metric_key=primary_metric_key,
        )
        estimator = estimators[best_key]

    return estimator.predict(X)
