"""
回帰タスク向け AutoML / 分析エンジンの設定スキーマ.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

from automl_engine.common.constants.algorithms import REGRESSION_MODEL_REGISTRY
from automl_engine.common.constants.metrics import REGRESSION_METRIC_REGISTRY


class RegressionAnalyzerConfig(BaseModel):
    """
    回帰タスク向け AutoML / 分析エンジンの設定スキーマ.

    Attributes:
        data: 分析対象のデータセット.
        feature_columns: 使用する特徴量カラム（省略時は data から自動推定）.
        target_column: 目的変数カラム名.
        ignore_columns: 分析から除外するカラム.
        column_types: 各カラムの dtype 変換指定（例: {"age": "int64"}）.
        random_state: 乱数シード.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame
    target_column: str

    feature_columns: Optional[list[str]] = None
    ignore_columns: Optional[list[str]] = None
    column_types: Optional[dict[str, str]] = None
    drop_unused_columns: bool = False
    random_state: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_before(cls, values: dict) -> dict:
        """
        入力値を正規化する.

        Args:
            values: 入力値の辞書.

        Returns:
                dict: 正規化後の値の辞書.
        """
        data: pd.DataFrame = values["data"]
        target: str = values["target_column"]

        ignore = values.get("ignore_columns") or []
        col_types = values.get("column_types") or {}

        values["ignore_columns"] = ignore
        values["column_types"] = col_types

        feature_cols = values.get("feature_columns")
        if feature_cols is None:
            values["feature_columns"] = [
                c for c in data.columns.tolist() if c != target and c not in ignore
            ]

        return values

    @model_validator(mode="after")
    def validate_after(self) -> RegressionAnalyzerConfig:
        """
        フィールド間の整合性を検証する.

        Returns:
            RegressionAnalyzerConfig: 検証済みインスタンス.

        Raises:
            ValueError: 検証エラー時.
        """
        # target が data に存在
        if self.target_column not in self.data.columns:
            raise ValueError("target_column is not found in data.")

        # feature_columns は空禁止
        if not self.feature_columns:
            raise ValueError("feature_columns must contain at least one column.")

        # feature_columns は data に存在（target を含めない想定）
        missing = [c for c in self.feature_columns if c not in self.data.columns]
        if missing:
            raise ValueError("feature_columns contains columns not found in data.")

        # ignore と feature の重複禁止
        if set(self.ignore_columns or []) & set(self.feature_columns):
            raise ValueError("ignore_columns must not overlap with feature_columns.")

        # column_types の自動補完
        for col in self.data.columns:
            if col not in (self.column_types or {}):
                (self.column_types or {})[col] = str(self.data[col].dtype)

        return self


class RegressionAnalyzerPrepareConfig(BaseModel):
    """
    prepare() 入力パラメータのバリデーション用モデル.

    Attributes:
        preprocess: 前処理パイプラインまたはカラム変換器.
        cv: クロスバリデーションの分割数または分割器.
    """

    preprocess: Optional[Pipeline | ColumnTransformer] = None
    cv: int | BaseCrossValidator = Field(5, description="CV folds or CV splitter")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("cv")
    @classmethod
    def validate_cv(cls, v: int | BaseCrossValidator) -> int | BaseCrossValidator:
        """
        cv の値を検証する.

        Args:
            v: 入力値.

        Returns:
            int | BaseCrossValidator: 検証済み値.

        Raises:
            ValueError: cv が不正な場合.
        """
        if isinstance(v, int) and v < 2:
            raise ValueError("cv must be >= 2 when int")

        return v


class RegressionTrainConfig(BaseModel):
    """
    train() 入力を正規化するスキーマ.

    Attributes:
        algorithms: 推定器生成 Callable 群.
        metrics: 評価指標 Callable 群.
        primary_metric_key: 最適化対象の評価指標キー.
        search_method: ハイパーパラメータ探索手法（"grid" または "optuna"）.
        optuna_trials: Optuna を使用する場合の試行回数.
        optuna_timeout: Optuna を使用する場合のタイムアウト時間（秒）.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    algorithms: Optional[str | list[str] | dict[str, dict[str, Callable[..., Any]]]] = (
        None
    )
    metrics: Optional[str | list[str] | dict[str, dict[str, Callable[..., float]]]] = (
        None
    )
    primary_metric_key: Optional[str] = None
    search_method: Optional[Literal["grid", "optuna"]] = None
    optuna_trials: int = Field(50, ge=1)
    optuna_timeout: Optional[int] = Field(None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, values: Any) -> Any:
        """
        入力をレジストリ型へ正規化する（before）.

        Args:
            values: 入力値の辞書.
        """
        if values is None:
            return values

        # 辞書以外はそのまま返す
        if isinstance(values, dict):
            data: dict[str, Any] = dict(values)
        else:
            return values

        # algorithms 正規化
        alg_in = data.get("algorithms")
        if isinstance(alg_in, str):
            alg_in = [alg_in]

        if alg_in is None:
            data["algorithms"] = REGRESSION_MODEL_REGISTRY
        elif isinstance(alg_in, list):
            data["algorithms"] = {k: REGRESSION_MODEL_REGISTRY[k] for k in alg_in}
        else:
            data["algorithms"] = alg_in

        # metrics 正規化
        met_in = data.get("metrics")

        if isinstance(met_in, str):
            met_in = [met_in]

        if met_in is None:
            data["metrics"] = REGRESSION_METRIC_REGISTRY
        elif isinstance(met_in, list):
            data["metrics"] = {k: REGRESSION_METRIC_REGISTRY[k] for k in met_in}
        else:
            data["metrics"] = met_in

        # primary_metric_key の自動設定
        pmk = data.get("primary_metric_key")
        metrics_dict = data.get("metrics")

        if pmk is None and isinstance(metrics_dict, dict):
            data["primary_metric_key"] = next(iter(metrics_dict.keys()), None)

        return data
        return data
