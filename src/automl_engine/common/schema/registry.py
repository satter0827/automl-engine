"""
Registry dict バリデーション用Pydanticスキーマ.

このモジュールは、algorithms.py および metrics.py で定義されている
レジストリ辞書の構造を厳密に型定義し、バリデーションを可能にします。
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.base import BaseEstimator

# =============================================================================
# Optuna パラメータ定義スキーマ（type別に厳密化）
# =============================================================================


class OptunaIntParam(BaseModel):
    """
    Optuna int型パラメータ定義.

    Attributes:
        type: パラメータの型（"int"）.
        low: 範囲の下限値.
        high: 範囲の上限値.
    """

    type: Literal["int"]
    low: int
    high: int


class OptunaFloatParam(BaseModel):
    """
    Optuna float型パラメータ定義.

    Attributes:
        type: パラメータの型（"float"）.
        low: 範囲の下限値.
        high: 範囲の上限値.
    """

    type: Literal["float"]
    low: float
    high: float


class OptunaLoguniformParam(BaseModel):
    """
    Optuna loguniform型パラメータ定義.

    Attributes:
        type: パラメータの型（"loguniform"）.
        low: 範囲の下限値（対数空間）.
        high: 範囲の上限値（対数空間）.
    """

    type: Literal["loguniform"]
    low: float
    high: float


class OptunaUniformParam(BaseModel):
    """
    Optuna uniform型パラメータ定義.

    Attributes:
        type: パラメータの型（"uniform"）.
        low: 範囲の下限値.
        high: 範囲の上限値.
    """

    type: Literal["uniform"]
    low: float
    high: float


class OptunaCategoricalParam(BaseModel):
    """
    Optuna categorical型パラメータ定義.

    Attributes:
        type: パラメータの型（"categorical"）.
        choices: 選択肢のリスト.
    """

    type: Literal["categorical"]
    choices: list[Any]


class OptunaIntOrNoneParam(BaseModel):
    """
    Optuna int_or_none型パラメータ定義.

    Attributes:
        type: パラメータの型（"int_or_none"）.
        low: 範囲の下限値.
        high: 範囲の上限値.
    """

    type: Literal["int_or_none"]
    low: int
    high: int


# Optunaパラメータの統合型
OptunaParam = Union[
    OptunaIntParam,
    OptunaFloatParam,
    OptunaLoguniformParam,
    OptunaUniformParam,
    OptunaCategoricalParam,
    OptunaIntOrNoneParam,
]


# =============================================================================
# Search space スキーマ定義
# =============================================================================


class GridSearchSpace(BaseModel):
    """
    Grid searchのパラメータ空間定義.

    GridSearchでは各パラメータに対して値のリストを定義します。
    値の型は厳密に定義せず、柔軟に対応します。
    """

    model_config = ConfigDict(extra="allow")


class OptunaSearchSpace(BaseModel):
    """
    Optunaのパラメータ空間定義.

    Optunaでは各パラメータに対してOptunaParam型の定義を持ちます。
    """

    model_config = ConfigDict(extra="allow")

    @field_validator("*", mode="before")
    @classmethod
    def validate_param(cls, v: Any) -> Any:
        """
        各パラメータがOptunaParam型であることを検証.

        Args:
            v: パラメータ定義.

        Returns:
            Any: 検証済みパラメータ定義.
        """
        # dict形式の場合はOptunaParamとして検証
        if isinstance(v, dict):
            param_type = v.get("type")
            if param_type == "int":
                return OptunaIntParam(**v)
            elif param_type == "float":
                return OptunaFloatParam(**v)
            elif param_type == "loguniform":
                return OptunaLoguniformParam(**v)
            elif param_type == "uniform":
                return OptunaUniformParam(**v)
            elif param_type == "categorical":
                return OptunaCategoricalParam(**v)
            elif param_type == "int_or_none":
                return OptunaIntOrNoneParam(**v)
            else:
                raise ValueError(f"Unknown optuna param type: {param_type}")
        return v


class SearchSpaceSchema(BaseModel):
    """
    ハイパーパラメータ探索空間の定義.

    Attributes:
        grid: Grid searchのパラメータ空間.
        optuna: Optunaのパラメータ空間.
    """

    grid: GridSearchSpace = Field(default_factory=lambda: GridSearchSpace())
    optuna: OptunaSearchSpace = Field(default_factory=lambda: OptunaSearchSpace())


# =============================================================================
# Algorithm registry スキーマ定義
# =============================================================================


class AlgorithmRegistryEntry(BaseModel):
    """
    アルゴリズムレジストリのエントリ定義.

    Attributes:
        label: アルゴリズムの表示名.
        estimator_cls: scikit-learnのEstimatorクラス.
        init_params: Estimator初期化時のパラメータ.
        fit_params: fit()メソッドに渡すパラメータ.
        search_space: ハイパーパラメータ探索空間.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    estimator_cls: type[BaseEstimator]
    init_params: dict[str, Any] = Field(default_factory=dict)
    fit_params: dict[str, Any] = Field(default_factory=dict)
    search_space: SearchSpaceSchema = Field(default_factory=SearchSpaceSchema)


class AlgorithmRegistry(BaseModel):
    """
    アルゴリズムレジストリ全体の定義.

    各キーに対してAlgorithmRegistryEntryを持つ辞書です。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @field_validator("*", mode="before")
    @classmethod
    def validate_entry(cls, v: Any) -> Any:
        """
        各エントリがAlgorithmRegistryEntry型であることを検証.

        Args:
            v: レジストリエントリ.

        Returns:
            Any: 検証済みエントリ.
        """
        if isinstance(v, dict):
            return AlgorithmRegistryEntry(**v)
        return v


# =============================================================================
# Metrics registry スキーマ定義
# =============================================================================


class ScorerDictSchema(BaseModel):
    """
    make_scorer用のスコアラー定義.

    Attributes:
        score_func: スコア計算関数（callable）.
        greater_is_better: スコアが大きいほど良いかどうか.
        response_method: 予測メソッド名（例: "predict", "predict_proba"）.
        kwargs: score_funcに渡す追加引数.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    score_func: Any  # callableだが型定義上はAnyとする
    greater_is_better: bool = True
    response_method: str = "predict"
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score_func")
    @classmethod
    def validate_score_func(cls, v: Any) -> Any:
        """
        score_funcがcallableであることを検証.

        Args:
            v: score_func.

        Returns:
            Any: 検証済みscore_func.

        Raises:
            ValueError: callableでない場合.
        """
        if not callable(v):
            raise ValueError("score_func must be callable")
        return v


class MetricRegistryEntry(BaseModel):
    """
    メトリクスレジストリのエントリ定義.

    Attributes:
        label: メトリクスの表示名.
        scorer: スコアラー（文字列またはScorerDictSchema）.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    scorer: Union[str, ScorerDictSchema]


class MetricRegistry(BaseModel):
    """
    メトリクスレジストリ全体の定義.

    各キーに対してMetricRegistryEntryを持つ辞書です。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @field_validator("*", mode="before")
    @classmethod
    def validate_entry(cls, v: Any) -> Any:
        """
        各エントリがMetricRegistryEntry型であることを検証.

        Args:
            v: レジストリエントリ.

        Returns:
            Any: 検証済みエントリ.
        """
        if isinstance(v, dict):
            return MetricRegistryEntry(**v)
        return v
