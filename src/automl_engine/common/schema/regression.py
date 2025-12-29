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
    train() 入力を正規化するスキーマ（回帰）.

    Attributes:
        algorithms: 使用するアルゴリズムの指定.
        metrics: 使用する評価指標の指定.
        primary_metric_key: 主評価指標キー.
        search_method: ハイパーパラメータ探索手法.
        optuna_trials: Optuna 試行回数.
        optuna_timeout: Optuna タイムアウト秒数.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    algorithms: Optional[str | list[str] | dict[str, dict[str, Any]]] = None
    metrics: Optional[str | list[str] | dict[str, dict[str, Any]]] = None
    primary_metric_key: str | None = None

    search_method: Optional[Literal["grid", "optuna"]] = None
    optuna_trials: int = Field(default=50, ge=1)
    optuna_timeout: int | None = Field(default=None, ge=1)

    @field_validator("algorithms", mode="before")
    @classmethod
    def _v_algorithms(cls, v: Any) -> dict[str, dict[str, Any]]:
        if v is None:
            return {k: dict(spec) for k, spec in REGRESSION_MODEL_REGISTRY.items()}

        if isinstance(v, str):
            if v not in REGRESSION_MODEL_REGISTRY:
                raise ValueError(f"Unknown algorithm key: {v!r}")
            return {v: dict(REGRESSION_MODEL_REGISTRY[v])}

        if isinstance(v, list) and not isinstance(v, (str, bytes)):
            out: dict[str, dict[str, Any]] = {}
            for key in v:
                if key not in REGRESSION_MODEL_REGISTRY:
                    raise ValueError(f"Unknown algorithm key: {key!r}")
                out[str(key)] = dict(REGRESSION_MODEL_REGISTRY[str(key)])
            return out

        raise TypeError("algorithms must be None, str, list[str], or dict[str, dict].")

    @field_validator("metrics", mode="before")
    @classmethod
    def _v_metrics(cls, v: Any) -> dict[str, dict[str, Any]]:
        def _ensure_spec(key: str, spec: dict[str, Any]) -> dict[str, Any]:
            spec_dict = dict(spec)
            if "scorer" not in spec_dict:
                raise ValueError(f"Metric spec must contain 'scorer': {key!r}")

            scorer = spec_dict["scorer"]
            if isinstance(scorer, dict):
                score_func = scorer.get("score_func")
                if not callable(score_func):
                    raise ValueError(
                        f"Metric scorer.score_func must be callable: {key!r}"
                    )
                kwargs = scorer.get("kwargs", {})
                if not isinstance(kwargs, dict):
                    raise ValueError(f"Metric scorer.kwargs must be dict: {key!r}")
            return spec_dict

        if v is None:
            return {
                k: _ensure_spec(k, spec)
                for k, spec in REGRESSION_METRIC_REGISTRY.items()
            }

        if isinstance(v, str):
            if v not in REGRESSION_METRIC_REGISTRY:
                raise ValueError(f"Unknown metric key: {v!r}")
            return {v: _ensure_spec(v, REGRESSION_METRIC_REGISTRY[v])}

        if isinstance(v, list) and not isinstance(v, (str, bytes)):
            out: dict[str, dict[str, Any]] = {}
            for key in v:
                key_s = str(key)
                if key_s not in REGRESSION_METRIC_REGISTRY:
                    raise ValueError(f"Unknown metric key: {key_s!r}")
                out[key_s] = _ensure_spec(key_s, REGRESSION_METRIC_REGISTRY[key_s])
            return out

        if isinstance(v, dict):
            out2: dict[str, dict[str, Any]] = {}
            for k, spec in v.items():
                if not isinstance(spec, dict):
                    raise TypeError(f"Metric spec must be dict: {k!r}")
                # Validate that the key exists in the registry
                if str(k) not in REGRESSION_METRIC_REGISTRY:
                    raise ValueError(
                        f"Unknown metric key: {k!r}. "
                        f"Valid keys are: {list(REGRESSION_METRIC_REGISTRY.keys())}"
                    )
                out2[str(k)] = _ensure_spec(str(k), spec)
            return out2

        raise TypeError("metrics must be None, str, list[str], or dict[str, dict].")

    @model_validator(mode="after")
    def _v_primary_metric_key(self) -> RegressionTrainConfig:
        if not self.metrics:
            raise ValueError("metrics must not be empty")

        if self.primary_metric_key is None:
            if isinstance(self.metrics, list):
                self.primary_metric_key = self.metrics[0]

            if isinstance(self.metrics, dict):
                self.primary_metric_key = next(iter(self.metrics.keys()))

            return self

        if self.primary_metric_key not in self.metrics:
            raise ValueError("primary_metric_key must exist in metrics")
        return self
