"""
Registry dict バリデーション用Pydanticスキーマのテスト.

このテストは、既存のレジストリ辞書が定義したスキーマでバリデーションを
通過することを確認します。
"""

from __future__ import annotations

from automl_engine.common.constants.algorithms import (
    CLASSIFICATION_MODEL_REGISTRY,
    REGRESSION_MODEL_REGISTRY,
)
from automl_engine.common.constants.metrics import (
    CLASSIFICATION_METRIC_REGISTRY,
    REGRESSION_METRIC_REGISTRY,
)
from automl_engine.common.schema.registry import AlgorithmRegistry, MetricRegistry


def test_classification_model_registry_validation() -> None:
    """
    CLASSIFICATION_MODEL_REGISTRYがスキーマバリデーションを通過することを確認.
    """
    # バリデーションが成功することを確認（例外が発生しないことを確認）
    registry = AlgorithmRegistry(**CLASSIFICATION_MODEL_REGISTRY)
    
    # インスタンスが正しく作成されたことを確認
    assert registry is not None


def test_regression_model_registry_validation() -> None:
    """
    REGRESSION_MODEL_REGISTRYがスキーマバリデーションを通過することを確認.
    """
    # バリデーションが成功することを確認（例外が発生しないことを確認）
    registry = AlgorithmRegistry(**REGRESSION_MODEL_REGISTRY)
    
    # インスタンスが正しく作成されたことを確認
    assert registry is not None


def test_classification_metric_registry_validation() -> None:
    """
    CLASSIFICATION_METRIC_REGISTRYがスキーマバリデーションを通過することを確認.
    """
    # バリデーションが成功することを確認（例外が発生しないことを確認）
    registry = MetricRegistry(**CLASSIFICATION_METRIC_REGISTRY)
    
    # インスタンスが正しく作成されたことを確認
    assert registry is not None


def test_regression_metric_registry_validation() -> None:
    """
    REGRESSION_METRIC_REGISTRYがスキーマバリデーションを通過することを確認.
    """
    # バリデーションが成功することを確認（例外が発生しないことを確認）
    registry = MetricRegistry(**REGRESSION_METRIC_REGISTRY)
    
    # インスタンスが正しく作成されたことを確認
    assert registry is not None


def test_individual_classification_algorithm_entries() -> None:
    """
    各分類アルゴリズムエントリが個別にバリデーションを通過することを確認.
    """
    from automl_engine.common.schema.registry import AlgorithmRegistryEntry
    
    for key, entry in CLASSIFICATION_MODEL_REGISTRY.items():
        # 各エントリが個別にバリデーションを通過することを確認
        validated_entry = AlgorithmRegistryEntry(**entry)
        
        # 必須フィールドが存在することを確認
        assert validated_entry.label
        assert validated_entry.estimator_cls
        assert isinstance(validated_entry.init_params, dict)
        assert isinstance(validated_entry.fit_params, dict)
        assert validated_entry.search_space


def test_individual_regression_algorithm_entries() -> None:
    """
    各回帰アルゴリズムエントリが個別にバリデーションを通過することを確認.
    """
    from automl_engine.common.schema.registry import AlgorithmRegistryEntry
    
    for key, entry in REGRESSION_MODEL_REGISTRY.items():
        # 各エントリが個別にバリデーションを通過することを確認
        validated_entry = AlgorithmRegistryEntry(**entry)
        
        # 必須フィールドが存在することを確認
        assert validated_entry.label
        assert validated_entry.estimator_cls
        assert isinstance(validated_entry.init_params, dict)
        assert isinstance(validated_entry.fit_params, dict)
        assert validated_entry.search_space


def test_individual_classification_metric_entries() -> None:
    """
    各分類メトリクスエントリが個別にバリデーションを通過することを確認.
    """
    from automl_engine.common.schema.registry import MetricRegistryEntry
    
    for key, entry in CLASSIFICATION_METRIC_REGISTRY.items():
        # 各エントリが個別にバリデーションを通過することを確認
        validated_entry = MetricRegistryEntry(**entry)
        
        # 必須フィールドが存在することを確認
        assert validated_entry.label
        assert validated_entry.scorer
        # scorerは文字列またはScorerDictSchemaであることを確認
        assert isinstance(validated_entry.scorer, (str, dict)) or hasattr(
            validated_entry.scorer, "score_func"
        )


def test_individual_regression_metric_entries() -> None:
    """
    各回帰メトリクスエントリが個別にバリデーションを通過することを確認.
    """
    from automl_engine.common.schema.registry import MetricRegistryEntry
    
    for key, entry in REGRESSION_METRIC_REGISTRY.items():
        # 各エントリが個別にバリデーションを通過することを確認
        validated_entry = MetricRegistryEntry(**entry)
        
        # 必須フィールドが存在することを確認
        assert validated_entry.label
        assert validated_entry.scorer
        # scorerは文字列またはScorerDictSchemaであることを確認
        assert isinstance(validated_entry.scorer, (str, dict)) or hasattr(
            validated_entry.scorer, "score_func"
        )
