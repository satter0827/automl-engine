"""
教師あり学習の学習・評価・探索を実行するモジュール.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import optuna
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.pipeline import Pipeline


def run_supervised(
    X: Any,
    y: Any,
    *,
    algorithms: dict[str, Callable],
    metrics: dict[str, Callable],
    primary_metric_key: str,
    preprocess: Optional[Pipeline | ColumnTransformer] = None,
    cv: BaseCrossValidator,
    search_method: Literal["grid", "optuna"] = "grid",
    optuna_trials: int = 50,
    optuna_timeout: Optional[int] = None,
    sample_weight: Optional[Any] = None,
    groups: Optional[Any] = None,
) -> tuple[dict[str, BaseEstimator], dict[str, dict[str, Any]]]:
    """
    教師あり学習の学習・評価・探索を実行する.

    Args:
        X: 特徴量データ.
        y: 目的変数.
        algorithms: 推定器生成 Callable 群.
        metrics: 評価指標 Callable 群.
        primary_metric_key: 最適化対象の評価指標キー.
        preprocess: 前処理パイプライン.
        cv: クロスバリデーション分割器.
        search_method: 探索手法（grid / optuna）.
        optuna_trials: Optuna の試行回数.
        optuna_timeout: Optuna のタイムアウト秒数.
        sample_weight: サンプル重み.
        groups: CV 用グループ.

    Returns:
        学習済み推定器と評価結果の辞書.
    """
    # 評価指標の scorers を作成
    scorers = {k: make_scorer(f) for k, f in metrics.items()}

    # 各アルゴリズムで学習・評価・探索を実行
    estimators: dict[str, BaseEstimator] = {}
    results: dict[str, dict[str, Any]] = {}

    # アルゴリズムごとに処理を実行
    for key, factory in algorithms.items():
        if search_method == "grid":
            # グリッドサーチで学習・評価を実行
            est, info = _run_grid(
                factory=factory,
                scorers=scorers,
                primary_metric_key=primary_metric_key,
                preprocess=preprocess,
                cv=cv,
                X=X,
                y=y,
                sample_weight=sample_weight,
                groups=groups,
            )
        elif search_method == "optuna":
            # Optuna で学習・評価・探索を実行
            est, info = _run_optuna(
                factory=factory,
                scorers=scorers,
                primary_metric_key=primary_metric_key,
                preprocess=preprocess,
                cv=cv,
                X=X,
                y=y,
                optuna_trials=optuna_trials,
                optuna_timeout=optuna_timeout,
                sample_weight=sample_weight,
                groups=groups,
            )
        else:
            # パラメタチューニングを行わずに学習・評価を実行
            pass

        # 結果を格納
        estimators[key] = est
        results[key] = info

    # 結果を返す
    return estimators, results


def _run_grid(
    *,
    factory: Callable,
    scorers: dict[str, Any],
    primary_metric_key: str,
    preprocess: Optional[Pipeline | ColumnTransformer],
    cv: BaseCrossValidator,
    X: Any,
    y: Any,
    sample_weight: Optional[Any],
    groups: Optional[Any],
) -> tuple[BaseEstimator, dict[str, Any]]:
    """
    Grid（固定パラメータ）で学習・評価を行う.

    Args:
        factory: 推定器生成 Callable.
        scorers: 評価指標 scorers.
        primary_metric_key: 主評価指標キー.
        preprocess: 前処理.
        cv: クロスバリデーション.
        X: 特徴量.
        y: 目的変数.
        sample_weight: サンプル重み.
        groups: CV 用グループ.

    Returns:
        学習済み推定器と評価情報.
    """
    # 推定器の生成
    estimator = factory()
    model = _compose(preprocess, estimator)
    fit_params = _fit_params(model, sample_weight)

    # クロスバリデーションで評価
    cv_out = cross_validate(
        estimator=model,
        X=X,
        y=y,
        scoring=scorers,
        cv=cv,
        n_jobs=-1,
        groups=groups,
        fit_params=fit_params,
        return_train_score=False,
    )

    # モデルの学習
    model.fit(X, y, groups=groups, **fit_params)

    # 結果の返却
    return model, {
        "search_method": "grid",
        "cv_scores_mean": {k: float(cv_out[f"test_{k}"].mean()) for k in scorers},
        "primary_score_mean": float(cv_out[f"test_{primary_metric_key}"].mean()),
    }


def _run_optuna(
    *,
    factory: Callable,
    scorers: dict[str, Any],
    primary_metric_key: str,
    preprocess: Optional[Pipeline | ColumnTransformer],
    cv: BaseCrossValidator,
    X: Any,
    y: Any,
    optuna_trials: int,
    optuna_timeout: Optional[int],
    sample_weight: Optional[Any],
    groups: Optional[Any],
) -> tuple[BaseEstimator, dict[str, Any]]:
    """
    Optuna によるハイパーパラメータ探索を行う.

    Args:
        factory: trial を受け取る推定器生成 Callable.
        scorers: 評価指標 scorers.
        primary_metric_key: 主評価指標キー.
        preprocess: 前処理.
        cv: クロスバリデーション.
        X: 特徴量.
        y: 目的変数.
        optuna_trials: 試行回数.
        optuna_timeout: タイムアウト秒数.
        sample_weight: サンプル重み.
        groups: CV 用グループ.

    Returns:
        学習済み推定器と評価情報.
    """

    def objective(trial: Any) -> float:
        """
        Optuna 目的関数.

        Args:
            trial: Optuna Trial オブジェクト.

        Returns:
            float: 主要評価指標の平均スコア.
        """
        # 推定器の生成
        est = factory(trial)
        model = _compose(preprocess, est)
        fit_params = _fit_params(model, sample_weight)

        # クロスバリデーションで評価
        out = cross_validate(
            estimator=model,
            X=X,
            y=y,
            scoring={primary_metric_key: scorers[primary_metric_key]},
            cv=cv,
            n_jobs=-1,
            groups=groups,
            fit_params=fit_params,
            return_train_score=False,
        )

        # 主要評価指標の平均スコアを返す
        return float(out[f"test_{primary_metric_key}"].mean())

    # Optuna スタディの作成と最適化の実行
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optuna_trials, timeout=optuna_timeout)

    # 最良モデルの学習と評価
    best_estimator = factory(study.best_trial)
    best_model = _compose(preprocess, best_estimator)
    best_fit_params = _fit_params(best_model, sample_weight)

    # 最良モデルを学習
    best_model.fit(X, y, groups=groups, **best_fit_params)

    # 最良モデルのクロスバリデーション評価
    cv_out = cross_validate(
        estimator=best_model,
        X=X,
        y=y,
        scoring=scorers,
        cv=cv,
        n_jobs=-1,
        groups=groups,
        fit_params=best_fit_params,
        return_train_score=False,
    )

    # 結果の返却
    return best_model, {
        "search_method": "optuna",
        "best_value": float(study.best_value),
        "best_params": dict(study.best_trial.params),
        "cv_scores_mean": {k: float(cv_out[f"test_{k}"].mean()) for k in scorers},
    }


def _compose(
    preprocess: Optional[Pipeline | ColumnTransformer],
    estimator: BaseEstimator,
) -> BaseEstimator:
    """
    前処理と推定器を結合する.

    Args:
        preprocess: 前処理パイプラインまたはカラム変換器.
        estimator: 推定器.

    Returns:
        結合されたパイプライン.
    """
    # None の場合、そのまま返す
    if preprocess is None:
        return estimator

    # Pipeline の場合、ステップを追加して返す
    if isinstance(preprocess, Pipeline):
        return Pipeline([*preprocess.steps, ("model", estimator)])

    # ColumnTransformer の場合、新しい Pipeline を作成して返す
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _fit_params(
    model: BaseEstimator,
    sample_weight: Optional[Any],
) -> dict[str, Any]:
    """
    sample_weight を fit 用パラメータに変換する.

    Args:
        model: 学習モデル.
        sample_weight: サンプル重み.

    Returns:
        fit 用パラメータ辞書.
    """
    # None の場合、空辞書を返す
    if sample_weight is None:
        return {}

    # Pipeline の場合、適切なキーで返す
    if isinstance(model, Pipeline):
        return {"model__sample_weight": sample_weight}

    # それ以外はそのまま返す
    return {"sample_weight": sample_weight}
