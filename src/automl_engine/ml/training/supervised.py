"""
教師あり学習の学習・評価・探索を実行するモジュール.
"""

from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Any, Literal, Optional

import joblib
import optuna
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# ロガーの初期化
logger = logging.getLogger(__name__)


def run_supervised(
    X: Any,
    y: Any,
    *,
    algorithms: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    primary_metric_key: str,
    preprocess: Optional[Pipeline | ColumnTransformer] = None,
    cv: BaseCrossValidator,
    search_method: Optional[Literal["grid", "optuna"]] = "grid",
    optuna_trials: int = 50,
    optuna_timeout: Optional[int] = None,
    sample_weight: Optional[Any] = None,
    groups: Optional[Any] = None,
    n_jobs: int = -1,
    n_jobs_cv: Optional[int] = None,
    parallel_backend: Literal["joblib", "concurrent", "ray"] = "joblib",
    algorithm_timeout: Optional[int] = None,
) -> tuple[dict[str, BaseEstimator], dict[str, dict[str, Any]]]:
    """
    教師あり学習の学習・評価・探索を実行する（並列化対応）.

    Args:
        X: 特徴量データ.
        y: 目的変数.
        algorithms: アルゴリズム定義辞書。各エントリは以下の構造を持つ:
            - estimator_cls: 推定器クラス（scikit-learn の BaseEstimator）
            - init_params: 推定器の初期化パラメータ
            - search_space: パラメータ探索空間（grid と optuna のキーを含む）
        metrics: 評価指標定義辞書。各エントリは以下の構造を持つ:
            - label: 評価指標のラベル
            - scorer: 評価指標の scorer（str、callable、または dict）
        primary_metric_key: 最適化対象の評価指標キー.
        preprocess: 前処理パイプライン.
        cv: クロスバリデーション分割器.
        search_method: 探索手法。以下のいずれか:
            - "grid": search_space["grid"] を使用して GridSearchCV を実行
            - "optuna": search_space["optuna"] を使用して Optuna で最適化
            - None: パラメータ探索を行わず、init_params のみでクロスバリデーション実行
        optuna_trials: Optuna の試行回数（search_method="optuna" の場合のみ使用）.
        optuna_timeout: Optuna のタイムアウト秒数（search_method="optuna" の場合のみ使用）.
        sample_weight: サンプル重み.
        groups: CV 用グループ.
        n_jobs: アルゴリズム並列数（-1で全コア使用）.
        n_jobs_cv: CV並列数（Noneで自動計算）.
        parallel_backend: 並列バックエンド（joblib / concurrent / ray）.
        algorithm_timeout: アルゴリズムごとのタイムアウト秒数.

    Returns:
        学習済み推定器と評価結果の辞書のタプル.
    """
    # 評価指標の scorers を作成
    scorers = _build_scoring(metrics)

    # CPU コア数の取得
    cpu_count = multiprocessing.cpu_count()

    # n_jobs の正規化（-1 は全コア使用）
    if n_jobs == -1:
        n_jobs_normalized = cpu_count
    else:
        n_jobs_normalized = min(n_jobs, cpu_count)

    # n_jobs_cv の自動計算
    if n_jobs_cv is None:
        # アルゴリズム並列数を考慮して CV 並列数を自動計算
        n_jobs_cv = max(1, cpu_count // n_jobs_normalized)
    else:
        n_jobs_cv = n_jobs_cv

    logger.info(
        f"並列実行開始: アルゴリズム数={len(algorithms)}, "
        f"n_jobs={n_jobs_normalized}, n_jobs_cv={n_jobs_cv}, "
        f"backend={parallel_backend}"
    )

    # 進捗情報の初期化
    progress_info: dict[str, Any] = {
        "total": len(algorithms),
        "completed": 0,
        "running": [],
        "failed": [],
        "results": {},
    }

    # 各アルゴリズムで学習・評価・探索を並列実行
    estimators: dict[str, BaseEstimator] = {}
    results: dict[str, dict[str, Any]] = {}

    # 並列バックエンドを選択して実行
    if parallel_backend == "joblib":
        estimators, results, progress_info = _run_parallel_joblib(
            algorithms=algorithms,
            scorers=scorers,
            primary_metric_key=primary_metric_key,
            preprocess=preprocess,
            cv=cv,
            X=X,
            y=y,
            search_method=search_method,
            optuna_trials=optuna_trials,
            optuna_timeout=optuna_timeout,
            sample_weight=sample_weight,
            groups=groups,
            n_jobs=n_jobs_normalized,
            n_jobs_cv=n_jobs_cv,
            algorithm_timeout=algorithm_timeout,
            progress_info=progress_info,
        )
    elif parallel_backend == "concurrent":
        estimators, results, progress_info = _run_parallel_concurrent(
            algorithms=algorithms,
            scorers=scorers,
            primary_metric_key=primary_metric_key,
            preprocess=preprocess,
            cv=cv,
            X=X,
            y=y,
            search_method=search_method,
            optuna_trials=optuna_trials,
            optuna_timeout=optuna_timeout,
            sample_weight=sample_weight,
            groups=groups,
            n_jobs=n_jobs_normalized,
            n_jobs_cv=n_jobs_cv,
            algorithm_timeout=algorithm_timeout,
            progress_info=progress_info,
        )
    elif parallel_backend == "ray":
        estimators, results, progress_info = _run_parallel_ray(
            algorithms=algorithms,
            scorers=scorers,
            primary_metric_key=primary_metric_key,
            preprocess=preprocess,
            cv=cv,
            X=X,
            y=y,
            search_method=search_method,
            optuna_trials=optuna_trials,
            optuna_timeout=optuna_timeout,
            sample_weight=sample_weight,
            groups=groups,
            n_jobs=n_jobs_normalized,
            n_jobs_cv=n_jobs_cv,
            algorithm_timeout=algorithm_timeout,
            progress_info=progress_info,
        )
    else:
        raise ValueError(f"未サポートのバックエンド: {parallel_backend}")

    # 全アルゴリズムが失敗した場合は例外を送出
    if not estimators:
        raise RuntimeError(
            f"全アルゴリズムが失敗しました。失敗アルゴリズム: {progress_info['failed']}"
        )

    logger.info(
        f"並列実行完了: 成功={len(estimators)}, 失敗={len(progress_info['failed'])}"
    )

    # 結果を返す
    return estimators, results


def _execute_algorithm(
    key: str,
    factory: dict[str, Any],
    scorers: dict[str, Any],
    primary_metric_key: str,
    preprocess: Optional[Pipeline | ColumnTransformer],
    cv: BaseCrossValidator,
    X: Any,
    y: Any,
    search_method: Optional[str],
    optuna_trials: int,
    optuna_timeout: Optional[int],
    sample_weight: Optional[Any],
    groups: Optional[Any],
    n_jobs_cv: int,
) -> tuple[str, BaseEstimator, dict[str, Any]]:
    """
    単一アルゴリズムの学習・評価を実行するヘルパー関数.

    Args:
        key: アルゴリズムキー.
        factory: 推定器定義を含む辞書（estimator_cls, init_params, search_space を含む）.
        scorers: 評価指標 scorers.
        primary_metric_key: 主評価指標キー.
        preprocess: 前処理.
        cv: クロスバリデーション.
        X: 特徴量.
        y: 目的変数.
        search_method: 探索手法（"grid", "optuna", None）.
        optuna_trials: Optuna 試行回数.
        optuna_timeout: Optuna タイムアウト秒数.
        sample_weight: サンプル重み.
        groups: CV 用グループ.
        n_jobs_cv: CV 並列数.

    Returns:
        アルゴリズムキー、学習済み推定器、評価情報のタプル.

    Raises:
        ValueError: 未サポートの探索手法が指定された場合.
    """
    try:
        logger.info(f"アルゴリズム '{key}' の実行開始 (n_jobs_cv={n_jobs_cv})")

        if search_method == "grid":
            # グリッドサーチで学習・評価を実行
            est, info = _run_grid(
                estimator_cls=factory["estimator_cls"],
                init_params=factory.get("init_params", {}),
                search_space=factory.get("search_space", {}).get("grid", {}),
                scorers=scorers,
                primary_metric_key=primary_metric_key,
                preprocess=preprocess,
                cv=cv,
                X=X,
                y=y,
                sample_weight=sample_weight,
                groups=groups,
                n_jobs_cv=n_jobs_cv,
            )
        elif search_method == "optuna":
            # Optuna で学習・評価・探索を実行
            est, info = _run_optuna(
                estimator_cls=factory["estimator_cls"],
                init_params=factory.get("init_params", {}),
                search_space=factory.get("search_space", {}).get("optuna", {}),
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
                n_jobs_cv=n_jobs_cv,
            )
        elif search_method is None:
            # デフォルトパラメータでクロスバリデーションを実行
            est, info = _run_default(
                estimator_cls=factory["estimator_cls"],
                init_params=factory.get("init_params", {}),
                scorers=scorers,
                primary_metric_key=primary_metric_key,
                preprocess=preprocess,
                cv=cv,
                X=X,
                y=y,
                sample_weight=sample_weight,
                groups=groups,
                n_jobs_cv=n_jobs_cv,
            )
        else:
            raise ValueError(f"未サポートの探索手法: {search_method}")

        logger.info(f"アルゴリズム '{key}' の実行完了")
        return key, est, info

    except Exception as e:
        logger.error(f"アルゴリズム '{key}' の実行中にエラーが発生: {str(e)}")
        raise


def _run_parallel_joblib(
    algorithms: dict[str, dict[str, Any]],
    scorers: dict[str, Any],
    primary_metric_key: str,
    preprocess: Optional[Pipeline | ColumnTransformer],
    cv: BaseCrossValidator,
    X: Any,
    y: Any,
    search_method: Optional[str],
    optuna_trials: int,
    optuna_timeout: Optional[int],
    sample_weight: Optional[Any],
    groups: Optional[Any],
    n_jobs: int,
    n_jobs_cv: int,
    algorithm_timeout: Optional[int],
    progress_info: dict[str, Any],
) -> tuple[dict[str, BaseEstimator], dict[str, dict[str, Any]], dict[str, Any]]:
    """
    joblib を使用した並列実行.

    Args:
        algorithms: アルゴリズム定義.
        scorers: 評価指標 scorers.
        primary_metric_key: 主評価指標キー.
        preprocess: 前処理.
        cv: クロスバリデーション.
        X: 特徴量.
        y: 目的変数.
        search_method: 探索手法.
        optuna_trials: Optuna 試行回数.
        optuna_timeout: Optuna タイムアウト秒数.
        sample_weight: サンプル重み.
        groups: CV 用グループ.
        n_jobs: 並列数.
        n_jobs_cv: CV 並列数.
        algorithm_timeout: アルゴリズムタイムアウト秒数.
        progress_info: 進捗情報.

    Returns:
        学習済み推定器、評価結果、進捗情報のタプル.
    """
    estimators: dict[str, BaseEstimator] = {}
    results: dict[str, dict[str, Any]] = {}

    # joblib で並列実行
    tasks = [
        joblib.delayed(_execute_algorithm)(
            key=key,
            factory=factory,
            scorers=scorers,
            primary_metric_key=primary_metric_key,
            preprocess=preprocess,
            cv=cv,
            X=X,
            y=y,
            search_method=search_method,
            optuna_trials=optuna_trials,
            optuna_timeout=optuna_timeout,
            sample_weight=sample_weight,
            groups=groups,
            n_jobs_cv=n_jobs_cv,
        )
        for key, factory in algorithms.items()
    ]

    # tqdm で進捗表示
    with tqdm(total=len(algorithms), desc="アルゴリズム実行", unit="algo") as pbar:
        # タイムアウトを考慮した並列実行
        parallel = joblib.Parallel(n_jobs=n_jobs, backend="loky", timeout=algorithm_timeout)

        try:
            # 完了した順に結果を取得
            for result in parallel(tasks):
                key, est, info = result
                estimators[key] = est
                results[key] = info
                progress_info["completed"] += 1
                progress_info["results"][key] = {"status": "success", "info": info}
                pbar.update(1)

        except Exception as e:
            # 一部失敗してもエラーログを記録して続行
            logger.error(f"並列実行中にエラーが発生: {str(e)}")

            # タスクを個別に実行して失敗したものを特定
            for key, factory in algorithms.items():
                if key in estimators:
                    continue

                try:
                    result = _execute_algorithm(
                        key=key,
                        factory=factory,
                        scorers=scorers,
                        primary_metric_key=primary_metric_key,
                        preprocess=preprocess,
                        cv=cv,
                        X=X,
                        y=y,
                        search_method=search_method,
                        optuna_trials=optuna_trials,
                        optuna_timeout=optuna_timeout,
                        sample_weight=sample_weight,
                        groups=groups,
                        n_jobs_cv=n_jobs_cv,
                    )
                    key_result, est, info = result
                    estimators[key_result] = est
                    results[key_result] = info
                    progress_info["completed"] += 1
                    progress_info["results"][key] = {"status": "success", "info": info}
                    pbar.update(1)

                except Exception as algo_error:
                    logger.error(f"アルゴリズム '{key}' が失敗: {str(algo_error)}")
                    progress_info["failed"].append(key)
                    progress_info["results"][key] = {
                        "status": "failed",
                        "error": str(algo_error),
                    }
                    pbar.update(1)

    return estimators, results, progress_info


def _run_parallel_concurrent(
    algorithms: dict[str, dict[str, Any]],
    scorers: dict[str, Any],
    primary_metric_key: str,
    preprocess: Optional[Pipeline | ColumnTransformer],
    cv: BaseCrossValidator,
    X: Any,
    y: Any,
    search_method: Optional[str],
    optuna_trials: int,
    optuna_timeout: Optional[int],
    sample_weight: Optional[Any],
    groups: Optional[Any],
    n_jobs: int,
    n_jobs_cv: int,
    algorithm_timeout: Optional[int],
    progress_info: dict[str, Any],
) -> tuple[dict[str, BaseEstimator], dict[str, dict[str, Any]], dict[str, Any]]:
    """
    concurrent.futures を使用した並列実行.

    Args:
        algorithms: アルゴリズム定義.
        scorers: 評価指標 scorers.
        primary_metric_key: 主評価指標キー.
        preprocess: 前処理.
        cv: クロスバリデーション.
        X: 特徴量.
        y: 目的変数.
        search_method: 探索手法.
        optuna_trials: Optuna 試行回数.
        optuna_timeout: Optuna タイムアウト秒数.
        sample_weight: サンプル重み.
        groups: CV 用グループ.
        n_jobs: 並列数.
        n_jobs_cv: CV 並列数.
        algorithm_timeout: アルゴリズムタイムアウト秒数.
        progress_info: 進捗情報.

    Returns:
        学習済み推定器、評価結果、進捗情報のタプル.
    """
    estimators: dict[str, BaseEstimator] = {}
    results: dict[str, dict[str, Any]] = {}

    # ThreadPoolExecutor を使用（lambda がピクル化できないため）
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # タスクを送信
        future_to_key = {
            executor.submit(
                _execute_algorithm,
                key=key,
                factory=factory,
                scorers=scorers,
                primary_metric_key=primary_metric_key,
                preprocess=preprocess,
                cv=cv,
                X=X,
                y=y,
                search_method=search_method,
                optuna_trials=optuna_trials,
                optuna_timeout=optuna_timeout,
                sample_weight=sample_weight,
                groups=groups,
                n_jobs_cv=n_jobs_cv,
            ): key
            for key, factory in algorithms.items()
        }

        # tqdm で進捗表示
        with tqdm(total=len(algorithms), desc="アルゴリズム実行", unit="algo") as pbar:
            # 完了した順に結果を取得
            for future in as_completed(future_to_key):
                key = future_to_key[future]

                try:
                    # タイムアウトを指定して結果を取得
                    key_result, est, info = future.result(timeout=algorithm_timeout)
                    estimators[key_result] = est
                    results[key_result] = info
                    progress_info["completed"] += 1
                    progress_info["results"][key] = {"status": "success", "info": info}
                    logger.info(f"アルゴリズム '{key}' が成功")

                except TimeoutError:
                    logger.error(f"アルゴリズム '{key}' がタイムアウト")
                    progress_info["failed"].append(key)
                    progress_info["results"][key] = {
                        "status": "failed",
                        "error": "タイムアウト",
                    }

                except Exception as e:
                    logger.error(f"アルゴリズム '{key}' が失敗: {str(e)}")
                    progress_info["failed"].append(key)
                    progress_info["results"][key] = {"status": "failed", "error": str(e)}

                finally:
                    pbar.update(1)

    return estimators, results, progress_info


def _run_parallel_ray(
    algorithms: dict[str, dict[str, Any]],
    scorers: dict[str, Any],
    primary_metric_key: str,
    preprocess: Optional[Pipeline | ColumnTransformer],
    cv: BaseCrossValidator,
    X: Any,
    y: Any,
    search_method: Optional[str],
    optuna_trials: int,
    optuna_timeout: Optional[int],
    sample_weight: Optional[Any],
    groups: Optional[Any],
    n_jobs: int,
    n_jobs_cv: int,
    algorithm_timeout: Optional[int],
    progress_info: dict[str, Any],
) -> tuple[dict[str, BaseEstimator], dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Ray を使用した並列実行（オプション）.

    Args:
        algorithms: アルゴリズム定義.
        scorers: 評価指標 scorers.
        primary_metric_key: 主評価指標キー.
        preprocess: 前処理.
        cv: クロスバリデーション.
        X: 特徴量.
        y: 目的変数.
        search_method: 探索手法.
        optuna_trials: Optuna 試行回数.
        optuna_timeout: Optuna タイムアウト秒数.
        sample_weight: サンプル重み.
        groups: CV 用グループ.
        n_jobs: 並列数.
        n_jobs_cv: CV 並列数.
        algorithm_timeout: アルゴリズムタイムアウト秒数.
        progress_info: 進捗情報.

    Returns:
        学習済み推定器、評価結果、進捗情報のタプル.

    Raises:
        ImportError: Ray がインストールされていない場合.
    """
    try:
        import ray
    except ImportError:
        raise ImportError(
            "Ray バックエンドを使用するには ray をインストールしてください: pip install ray"
        )

    # Ray の初期化（既に初期化済みでなければ）
    if not ray.is_initialized():
        ray.init(num_cpus=n_jobs, ignore_reinit_error=True)

    # リモート関数として登録
    @ray.remote
    def execute_remote(
        key: str,
        factory: dict[str, Any],
        scorers: dict[str, Any],
        primary_metric_key: str,
        preprocess: Optional[Pipeline | ColumnTransformer],
        cv: BaseCrossValidator,
        X: Any,
        y: Any,
        search_method: Optional[str],
        optuna_trials: int,
        optuna_timeout: Optional[int],
        sample_weight: Optional[Any],
        groups: Optional[Any],
        n_jobs_cv: int,
    ) -> tuple[str, BaseEstimator, dict[str, Any]]:
        return _execute_algorithm(
            key=key,
            factory=factory,
            scorers=scorers,
            primary_metric_key=primary_metric_key,
            preprocess=preprocess,
            cv=cv,
            X=X,
            y=y,
            search_method=search_method,
            optuna_trials=optuna_trials,
            optuna_timeout=optuna_timeout,
            sample_weight=sample_weight,
            groups=groups,
            n_jobs_cv=n_jobs_cv,
        )

    estimators: dict[str, BaseEstimator] = {}
    results: dict[str, dict[str, Any]] = {}

    # タスクを送信（キーとのマッピングを保持）
    future_to_key: dict[Any, str] = {}
    for key, factory in algorithms.items():
        future = execute_remote.remote(
            key=key,
            factory=factory,
            scorers=scorers,
            primary_metric_key=primary_metric_key,
            preprocess=preprocess,
            cv=cv,
            X=X,
            y=y,
            search_method=search_method,
            optuna_trials=optuna_trials,
            optuna_timeout=optuna_timeout,
            sample_weight=sample_weight,
            groups=groups,
            n_jobs_cv=n_jobs_cv,
        )
        future_to_key[future] = key

    # tqdm で進捗表示
    with tqdm(total=len(algorithms), desc="アルゴリズム実行", unit="algo") as pbar:
        # 完了した順に結果を取得
        futures = list(future_to_key.keys())
        while futures:
            # タイムアウト付きで待機
            ready, not_ready = ray.wait(
                futures, timeout=algorithm_timeout if algorithm_timeout else None
            )

            for future in ready:
                key = future_to_key.get(future, "unknown")
                try:
                    key_result, est, info = ray.get(future)
                    estimators[key_result] = est
                    results[key_result] = info
                    progress_info["completed"] += 1
                    progress_info["results"][key_result] = {"status": "success", "info": info}
                    logger.info(f"アルゴリズム '{key_result}' が成功")

                except Exception as e:
                    logger.error(f"アルゴリズム '{key}' が失敗: {str(e)}")
                    progress_info["failed"].append(key)
                    progress_info["results"][key] = {"status": "failed", "error": str(e)}

                finally:
                    pbar.update(1)

            futures = not_ready

    return estimators, results, progress_info


def _build_scoring(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    metrics 定義から sklearn の scoring 辞書を生成する.

    Args:
        metrics: 評価指標定義.

    Returns:
        scoring: sklearn 用 scoring 辞書.
    """
    scoring: dict[str, Any] = {}
    for key, spec in metrics.items():
        scorer = spec["scorer"]

        # 設計: dict なら make_scorer へ、そうでなければ scoring として素通し
        if isinstance(scorer, dict):
            score_func = scorer["score_func"]
            greater_is_better = scorer.get("greater_is_better", True)
            response_method = scorer.get("response_method")
            kwargs = scorer.get("kwargs", {})

            if response_method is None:
                scoring[key] = make_scorer(
                    score_func,
                    greater_is_better=greater_is_better,
                    **kwargs,
                )
            else:
                scoring[key] = make_scorer(
                    score_func,
                    greater_is_better=greater_is_better,
                    response_method=response_method,
                    **kwargs,
                )
        else:
            scoring[key] = scorer

    return scoring


def _run_grid(
    *,
    estimator_cls: type[BaseEstimator],
    init_params: dict[str, Any],
    search_space: dict[str, Any],
    scorers: dict[str, Any],
    primary_metric_key: str,
    preprocess: Optional[Pipeline | ColumnTransformer],
    cv: BaseCrossValidator,
    X: Any,
    y: Any,
    sample_weight: Optional[Any],
    groups: Optional[Any],
    n_jobs_cv: int = -1,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """
    GridSearchCV によるパラメータ探索を行う.

    Args:
        estimator_cls: 推定器クラス.
        init_params: 推定器の初期化パラメータ.
        search_space: グリッドサーチ用パラメータ空間.
        scorers: 評価指標 scorers.
        primary_metric_key: 主評価指標キー.
        preprocess: 前処理.
        cv: クロスバリデーション.
        X: 特徴量.
        y: 目的変数.
        sample_weight: サンプル重み.
        groups: CV 用グループ.
        n_jobs_cv: CV 並列数.

    Returns:
        学習済み推定器と評価情報.
    """
    # 推定器の生成（init_params を使用）
    estimator = estimator_cls(**init_params)
    model = _compose(preprocess, estimator)
    fit_params = _fit_params(model, sample_weight)

    # search_space が空の場合はパラメータ探索なしで学習
    if not search_space:
        params = dict(fit_params)
        if groups is not None:
            params["groups"] = groups

        cv_out = cross_validate(
            estimator=model,
            X=X,
            y=y,
            scoring=scorers,
            cv=cv,
            n_jobs=n_jobs_cv,
            params=params,
            return_train_score=False,
        )

        # モデルの学習
        model.fit(X, y, **fit_params)

        # 結果の返却
        return model, {
            "search_method": "grid",
            "cv_scores_mean": {k: float(cv_out[f"test_{k}"].mean()) for k in scorers},
            "primary_score_mean": float(cv_out[f"test_{primary_metric_key}"].mean()),
        }

    # GridSearchCV でパラメータ探索
    # Pipeline の場合、パラメータ名に "model__" を付ける
    if isinstance(model, Pipeline):
        param_grid = {f"model__{k}": v for k, v in search_space.items()}
    else:
        param_grid = search_space

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorers,
        refit=primary_metric_key,
        cv=cv,
        n_jobs=n_jobs_cv,
        return_train_score=False,
    )

    # GridSearchCV の fit
    if groups is not None:
        grid_search.fit(X, y, groups=groups, **fit_params)
    else:
        grid_search.fit(X, y, **fit_params)

    # 最良モデルを取得
    best_model = grid_search.best_estimator_

    # 結果の返却
    cv_scores_mean = {
        k: float(grid_search.cv_results_[f"mean_test_{k}"][grid_search.best_index_]) for k in scorers
    }
    return best_model, {
        "search_method": "grid",
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "cv_scores_mean": cv_scores_mean,
        "primary_score_mean": float(grid_search.best_score_),
    }


def _run_optuna(
    *,
    estimator_cls: type[BaseEstimator],
    init_params: dict[str, Any],
    search_space: dict[str, Any],
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
    n_jobs_cv: int = -1,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """
    Optuna によるハイパーパラメータ探索を行う.

    Args:
        estimator_cls: 推定器クラス.
        init_params: 推定器の初期化パラメータ.
        search_space: Optuna 用パラメータ空間定義.
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
        n_jobs_cv: CV 並列数.

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
        # パラメータのサンプリング
        trial_params = dict(init_params)
        for param_name, param_spec in search_space.items():
            if isinstance(param_spec, dict) and "type" in param_spec:
                param_type = param_spec["type"]
                if param_type == "loguniform":
                    trial_params[param_name] = trial.suggest_float(
                        param_name, param_spec["low"], param_spec["high"], log=True
                    )
                elif param_type == "uniform":
                    trial_params[param_name] = trial.suggest_float(
                        param_name, param_spec["low"], param_spec["high"]
                    )
                elif param_type == "int":
                    trial_params[param_name] = trial.suggest_int(
                        param_name, param_spec["low"], param_spec["high"]
                    )
                elif param_type == "int_or_none":
                    # None を含む整数の選択
                    if trial.suggest_categorical(f"{param_name}_is_none", [True, False]):
                        trial_params[param_name] = None
                    else:
                        trial_params[param_name] = trial.suggest_int(
                            param_name, param_spec["low"], param_spec["high"]
                        )
                elif param_type == "categorical":
                    trial_params[param_name] = trial.suggest_categorical(
                        param_name, param_spec["choices"]
                    )

        # 推定器の生成
        est = estimator_cls(**trial_params)
        model = _compose(preprocess, est)
        fit_params = _fit_params(model, sample_weight)

        params = dict(fit_params)
        if groups is not None:
            params["groups"] = groups

        out = cross_validate(
            estimator=model,
            X=X,
            y=y,
            scoring=scorers,
            cv=cv,
            n_jobs=n_jobs_cv,
            params=params,
            return_train_score=False,
        )

        # 主要評価指標の平均スコアを返す
        return float(out[f"test_{primary_metric_key}"].mean())

    # Optuna スタディの作成と最適化の実行
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optuna_trials, timeout=optuna_timeout)

    # 最良パラメータで推定器を生成
    best_params = dict(init_params)
    for param_name, param_value in study.best_trial.params.items():
        # int_or_none の場合の特殊処理
        if not param_name.endswith("_is_none"):
            best_params[param_name] = param_value

    best_estimator = estimator_cls(**best_params)
    best_model = _compose(preprocess, best_estimator)
    best_fit_params = _fit_params(best_model, sample_weight)

    # 最良モデルを学習
    best_model.fit(X, y, **best_fit_params)

    params = dict(best_fit_params)
    if groups is not None:
        params["groups"] = groups

    cv_out = cross_validate(
        estimator=best_model,
        X=X,
        y=y,
        scoring=scorers,
        cv=cv,
        n_jobs=n_jobs_cv,
        params=params,
        return_train_score=False,
    )

    # 結果の返却
    return best_model, {
        "search_method": "optuna",
        "best_value": float(study.best_value),
        "best_params": dict(study.best_trial.params),
        "cv_scores_mean": {k: float(cv_out[f"test_{k}"].mean()) for k in scorers},
    }


def _run_default(
    *,
    estimator_cls: type[BaseEstimator],
    init_params: dict[str, Any],
    scorers: dict[str, Any],
    primary_metric_key: str,
    preprocess: Optional[Pipeline | ColumnTransformer],
    cv: BaseCrossValidator,
    X: Any,
    y: Any,
    sample_weight: Optional[Any],
    groups: Optional[Any],
    n_jobs_cv: int = -1,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """
    デフォルトパラメータでクロスバリデーションを実行する.

    パラメータ探索は行わず、init_params で指定されたパラメータのみを使用して
    推定器をインスタンス化し、クロスバリデーションを実行する。

    Args:
        estimator_cls: 推定器クラス.
        init_params: 推定器の初期化パラメータ.
        scorers: 評価指標 scorers.
        primary_metric_key: 主評価指標キー.
        preprocess: 前処理.
        cv: クロスバリデーション.
        X: 特徴量.
        y: 目的変数.
        sample_weight: サンプル重み.
        groups: CV 用グループ.
        n_jobs_cv: CV 並列数.

    Returns:
        学習済み推定器と評価情報.
    """
    # 推定器の生成（init_params のみを使用）
    estimator = estimator_cls(**init_params)
    model = _compose(preprocess, estimator)
    fit_params = _fit_params(model, sample_weight)

    params = dict(fit_params)
    if groups is not None:
        params["groups"] = groups

    # クロスバリデーションの実行
    cv_out = cross_validate(
        estimator=model,
        X=X,
        y=y,
        scoring=scorers,
        cv=cv,
        n_jobs=n_jobs_cv,
        params=params,
        return_train_score=False,
    )

    # モデルの学習
    model.fit(X, y, **fit_params)

    # 結果の返却
    return model, {
        "search_method": "default",
        "cv_scores_mean": {k: float(cv_out[f"test_{k}"].mean()) for k in scorers},
        "primary_score_mean": float(cv_out[f"test_{primary_metric_key}"].mean()),
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
