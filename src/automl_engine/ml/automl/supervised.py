"""
回帰タスクに対する AutoML 分析エンジン実装モジュール.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, cast

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.pipeline import Pipeline

from automl_engine.common.schema.regression import (
    RegressionAnalyzerConfig,
    RegressionAnalyzerPrepareConfig,
    RegressionTrainConfig,
)
from automl_engine.ml.training.supervised import run_supervised


class RegressionAnalyzer:
    """
    回帰タスクに対する分析・探索・学習・評価を一貫して実行する分析エンジン.

    本クラスは AutoML を前提とし、以下を目的とする：
    - 前処理・モデル探索・評価の統合実行
    - 実験コンテキストの保持
    - 分析結果の比較・要約・説明（LLM 利用を含む）
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        *,
        feature_columns: Optional[list[str]] = None,
        ignore_columns: Optional[list[str]] = None,
        column_types: Optional[dict[str, str]] = None,
        drop_unused_columns: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        """
        分析エンジンを初期化する.

        Args:
            data: 分析対象のデータセット.
            target_column: 目的変数カラム名.
            feature_columns: 使用する特徴量カラムのリスト（省略時は全カラム）.
            ignore_columns: 分析から除外するカラムのリスト.
            column_types: 各カラムの dtype 変換指定（例: {"age": "int64"}）.
            drop_unused_columns: 指定外のカラムを削除するかどうか.
            random_state: 乱数シード.
        """
        # 設定の正規化と検証を実行
        validated = RegressionAnalyzerConfig(
            data=data,
            feature_columns=feature_columns,
            target_column=target_column,
            ignore_columns=ignore_columns,
            column_types=column_types,
            drop_unused_columns=drop_unused_columns,
            random_state=random_state,
        )

        # 型変換を適用
        if validated.column_types:
            validated.data = validated.data.astype(validated.column_types)

        # 不要なカラムを削除
        if drop_unused_columns:
            feature_cols = validated.feature_columns or []
            used_columns = set(feature_cols) | {validated.target_column}
            validated.data = validated.data[list(used_columns)]

        # 検証済み設定をインスタンス変数に保存
        self.data = validated.data
        self.feature_columns = validated.feature_columns
        self.target_column = validated.target_column
        self.ignore_columns = validated.ignore_columns
        self.column_types = validated.column_types
        self.random_state = validated.random_state

        # インスタンス変数に設定を反映
        self._is_prepared = False
        self._is_trained = False

        self.estimators_: dict[str, BaseEstimator] = {}
        self.cv_results_: dict[str, BaseCrossValidator] = {}
        self.preprocess: Optional[Pipeline | ColumnTransformer] = None
        self.cv: int | BaseCrossValidator = 5
        self.search_method: Optional[Literal["grid", "optuna"]] = None
        self.optuna_trials = 50
        self.optuna_timeout: Optional[int] = None

    def prepare(
        self,
        preprocess: Optional[Pipeline | ColumnTransformer] = None,
        cv: int | BaseCrossValidator = 5,
    ) -> dict[str, Any]:
        """
        前処理を実行し、準備結果の概要と CV 分割結果を返す.

        Args:
            preprocess: 前処理パイプラインまたはカラム変換器.
            cv: クロスバリデーションの分割数または分割器.

        Returns:
            dict[str, Any]: 前処理結果および CV 分割結果を含む辞書.

        Raises:
            RuntimeError: setup() が呼ばれていない状態で実行された場合.
        """
        # 設定の正規化と検証を実行
        validated = RegressionAnalyzerPrepareConfig(
            preprocess=preprocess,
            cv=cv,
        )

        # インスタンス変数に設定を反映
        self.preprocess = validated.preprocess

        # 前処理の適用
        X_raw = self.data[self.feature_columns]
        X_prepared = self.preprocess.fit_transform(X_raw) if self.preprocess else X_raw
        y = self.data[self.target_column]

        # cvが整数の場合、KFoldインスタンスに変換
        if isinstance(validated.cv, int):
            cv_obj = KFold(
                n_splits=validated.cv,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            cv_obj = validated.cv

        self.cv = cv_obj

        # クロスバリデーションの分割結果を生成
        cv_splits = [
            {
                "fold": i,
                "train_idx": train_idx,
                "valid_idx": valid_idx,
            }
            for i, (train_idx, valid_idx) in enumerate(cv_obj.split(X_prepared))
        ]

        # prepare完了フラグを立てる
        self._is_prepared = True

        # 結果を返す
        return {
            "data": {
                "X": X_prepared,
                "y": y,
                "is_preprocessed": self.preprocess is not None,
            },
            "cv": {
                "strategy": type(cv_obj).__name__,
                "n_splits": cv_obj.get_n_splits(),
                "splits": cv_splits,
            },
            "overview": {
                "input_shape": X_raw.shape,
                "output_shape": X_prepared.shape,
                "n_features_out": int(X_prepared.shape[1]),
            },
        }

    def train(
        self,
        algorithms: Optional[
            str | list[str] | dict[str, dict[str, Callable[..., Any]]]
        ] = None,
        metrics: Optional[
            str | list[str] | dict[str, dict[str, Callable[..., Any]]]
        ] = None,
        primary_metric_key: Optional[str] = None,
        search_method: Optional[Literal["grid", "optuna"]] = None,
        optuna_trials: int = 50,
        optuna_timeout: Optional[int] = None,
    ) -> RegressionAnalyzer:
        """
        回帰モデルの探索・学習・評価を実行する.

        Args:
            algorithms: 使用する回帰アルゴリズムの指定.
            metrics: 使用する評価指標の指定.
            primary_metric_key: 最適化対象の主要評価指標のキー.


        Returns:
            RegressionAnalyzer: 学習済み状態となった自身のインスタンス.
        """
        # 過去の学習結果をクリア
        self.estimators_.clear()
        self.cv_results_.clear()

        # 設定の正規化と検証を実行
        validated = RegressionTrainConfig(
            algorithms=algorithms,
            metrics=metrics,
            primary_metric_key=primary_metric_key,
            search_method=search_method,
            optuna_trials=optuna_trials,
            optuna_timeout=optuna_timeout,
        )
        if isinstance(validated.algorithms, dict):
            self.algorithms = validated.algorithms
        if isinstance(validated.metrics, dict):
            self.metrics = validated.metrics
        if validated.primary_metric_key:
            self.primary_metric_key = validated.primary_metric_key
        self.search_method = validated.search_method
        self.optuna_trials = validated.optuna_trials
        self.optuna_timeout = validated.optuna_timeout

        if isinstance(self.algorithms, dict):
            validated.algorithms = self.algorithms

        # 学習・評価・探索を実行
        self.fitted_model, self.model_info = run_supervised(
            X=self.data[self.feature_columns],
            y=self.data[self.target_column],
            algorithms=self.algorithms,
            metrics=self.metrics,
            primary_metric_key=self.primary_metric_key,
            preprocess=self.preprocess,
            cv=self.cv,
            search_method=self.search_method,
            optuna_trials=self.optuna_trials,
            optuna_timeout=self.optuna_timeout,
        )

        # 学習完了フラグを立てる
        self._is_trained = True

        return self

    def predict(self, X: Any) -> Any:
        """
        最良モデルを用いて予測を行う.

        Args:
            X: 予測対象の特徴量データ.

        Returns:
            Any: 予測結果.
        """
        return None

    def evaluate(self, X: Any, y: Any) -> dict[str, float]:
        """
        学習済みモデルを用いて評価を行う.

        primary metric を含むすべての評価指標を算出する.

        Args:
            X: 評価用特徴量データ.
            y: 正解ターゲット.

        Returns:
            dict[str, float]: 評価指標名をキー、スコアを値とする辞書.
        """
        return {"dummy_metric": 0.0}

    def finalize(self) -> Pipeline:
        """
        最良モデルを確定し、デプロイ可能な形で出力する.

        Returns:
            Pipeline: デプロイ可能な最良モデルパイプライン.
        """
        return BaseEstimator()

    def compare(self) -> Any:
        """
        探索対象となったモデル群の比較結果を返す.

        各モデルの平均スコア、分散、学習時間などを含む.

        Returns:
            Any: モデル比較結果（ダミー）.
        """
        return []

    def inspect(self) -> dict[str, Any]:
        """
        クロスバリデーションや探索過程の詳細結果を取得する.

        Returns:
            dict[str, Any]: fold / trial 単位の詳細評価結果.
        """
        return {}

    def explain(self) -> dict[str, Any]:
        """
        モデルの判断根拠や重要特徴量を説明する.

        Returns:
            dict[str, Any]:モデル解釈情報.
        """
        return {}

    def summarize(self) -> dict[str, Any]:
        """
        実験全体の要約を生成する.

        人間および LLM による解釈を想定した
        構造化サマリを返す.

        Returns:
            dict[str, Any]:実験コンテキスト要約.
        """
        return {
            "status": "trained" if self._is_trained else "not_trained",
            "summary": "dummy summary",
        }

    def export(self) -> dict[str, Any]:
        """
        学習結果およびメタ情報を外部利用可能な形式で出力する.

        Returns:
            dict[str, Any]: 学習成果物およびメタ情報.
        """
        return {}

    def load(self, artifacts: dict[str, Any]) -> RegressionAnalyzer:
        """
        export() により生成された成果物から状態を復元する.

        Args:
            artifacts: 学習済み成果物およびメタ情報.

        Returns:
            RegressionAnalyzer:復元後の自身のインスタンス.
        """
        self._is_setup = True
        self._is_trained = True
        return self
