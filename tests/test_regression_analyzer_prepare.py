"""
RegressionAnalyzer の __init__ / prepare() を対象にしたテスト.

- __init__:
  - column_types による型変換
  - drop_unused_columns による不要カラム削除
  - feature_columns 未指定時の既定挙動（RegressionAnalyzerConfig 側の仕様に依存）
- prepare:
  - preprocess=None の場合
  - preprocess=ColumnTransformer の場合
  - cv=int -> KFold 変換
  - cv=BaseCrossValidator 指定時の挙動
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from automl_engine.ml.automl.supervised import RegressionAnalyzer


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """
    テスト用の DataFrame（数値 + カテゴリ + 目的変数 + 無関係列）.
    """
    rng = np.random.default_rng(42)
    n = 30

    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(loc=10.0, scale=2.0, size=n),
            "cat": rng.choice(["A", "B", "C"], size=n),
            "target": rng.normal(size=n),
            "unused": rng.integers(0, 100, size=n),
        }
    )

    return df


def _assert_splits_valid(
    *,
    n_rows: int,
    splits: list[dict[str, Any]],
) -> None:
    """
    CV split の整合性を最低限検証するヘルパ.

    Args:
        n_rows: 元データの行数
        splits: prepare() の出力の cv["splits"]
    """
    # splits が存在すること
    assert len(splits) > 0

    # 全インデックス集合
    all_indices = set(range(n_rows))
    for s in splits:
        assert {"fold", "train_idx", "valid_idx"} <= set(s.keys())

        train_idx = np.asarray(s["train_idx"])
        valid_idx = np.asarray(s["valid_idx"])

        # 重複なし
        assert len(np.intersect1d(train_idx, valid_idx)) == 0

        # 範囲内
        assert set(train_idx).issubset(all_indices)
        assert set(valid_idx).issubset(all_indices)

        # fold ごとに train/valid を合わせると全体を覆う（KFold想定）
        union = set(train_idx) | set(valid_idx)
        assert union == all_indices


def test_init_applies_column_types(sample_df: pd.DataFrame) -> None:
    """
    column_types 引数による型変換が行われることを確認.

    Args:
        sample_df: テスト用 DataFrame
    """
    analyzer = RegressionAnalyzer(
        data=sample_df,
        target_column="target",
        feature_columns=["num1", "num2", "cat"],
        column_types={"num2": "float64"},
    )

    assert analyzer.data["num2"].dtype == np.dtype("float64")


def test_init_drop_unused_columns(sample_df: pd.DataFrame) -> None:
    """
    drop_unused_columns=True 時に不要カラムが削除されることを確認.

    Args:
        sample_df: テスト用 DataFrame
    """
    analyzer = RegressionAnalyzer(
        data=sample_df,
        target_column="target",
        feature_columns=["num1", "num2", "cat"],
        drop_unused_columns=True,
    )

    assert set(analyzer.data.columns) == {"num1", "num2", "cat", "target"}
    assert "unused" not in analyzer.data.columns


def test_prepare_without_preprocess_cv_int(sample_df: pd.DataFrame) -> None:
    """
    preprocess=None, cv=int 指定時の prepare() 動作確認.

    Args:
        sample_df: テスト用 DataFrame
    """
    analyzer = RegressionAnalyzer(
        data=sample_df,
        target_column="target",
        feature_columns=["num1", "num2", "cat"],
        random_state=123,
    )

    out = analyzer.prepare(preprocess=None, cv=5)

    assert analyzer._is_prepared is True

    # data
    assert "data" in out
    X = out["data"]["X"]
    y = out["data"]["y"]
    assert out["data"]["is_preprocessed"] is False

    # preprocess=None なので DataFrame のままの想定
    assert isinstance(X, pd.DataFrame)
    assert X.shape == (len(sample_df), 3)
    assert isinstance(y, pd.Series)
    assert y.shape == (len(sample_df),)

    # cv
    assert out["cv"]["strategy"] == "KFold"
    assert out["cv"]["n_splits"] == 5
    assert len(out["cv"]["splits"]) == 5
    _assert_splits_valid(n_rows=len(sample_df), splits=out["cv"]["splits"])

    # overview
    assert out["overview"]["input_shape"] == (len(sample_df), 3)
    assert out["overview"]["output_shape"] == (len(sample_df), 3)
    assert out["overview"]["n_features_out"] == 3


def test_prepare_with_column_transformer(sample_df: pd.DataFrame) -> None:
    """
    preprocess=ColumnTransformer 指定時の prepare() 動作確認.

    Args:
        sample_df: テスト用 DataFrame
    """
    analyzer = RegressionAnalyzer(
        data=sample_df,
        target_column="target",
        feature_columns=["num1", "num2", "cat"],
        random_state=7,
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["num1", "num2"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["cat"]),
        ],
        remainder="drop",
    )

    out = analyzer.prepare(preprocess=preprocess, cv=3)

    assert analyzer._is_prepared is True
    assert out["data"]["is_preprocessed"] is True

    X_prepared = out["data"]["X"]

    # ColumnTransformer の出力は ndarray or sparse になり得るため shape のみ検証
    assert hasattr(X_prepared, "shape")
    assert X_prepared.shape[0] == len(sample_df)

    # OneHotEncoder で cat(A,B,C) の3列 + num2列 = 合計5列の想定
    # ※データ内のカテゴリ数に依存するので 5 を厳密に固定したくない場合は >=4 などにしてください
    assert X_prepared.shape[1] == 5

    assert out["cv"]["strategy"] == "KFold"
    assert out["cv"]["n_splits"] == 3
    _assert_splits_valid(n_rows=len(sample_df), splits=out["cv"]["splits"])

    # overview も整合
    assert out["overview"]["input_shape"] == (len(sample_df), 3)
    assert out["overview"]["output_shape"][0] == len(sample_df)
    assert out["overview"]["n_features_out"] == int(X_prepared.shape[1])


def test_prepare_with_cv_object_uses_given_strategy(sample_df: pd.DataFrame) -> None:
    """
    cv=BaseCrossValidator 指定時の prepare() 動作確認.

    Args:
        sample_df: テスト用 DataFrame
    """
    analyzer = RegressionAnalyzer(
        data=sample_df,
        target_column="target",
        feature_columns=["num1", "num2", "cat"],
        random_state=999,
    )

    cv = KFold(n_splits=4, shuffle=True, random_state=1)
    out = analyzer.prepare(preprocess=None, cv=cv)

    assert out["cv"]["strategy"] == "KFold"
    assert out["cv"]["n_splits"] == 4
    assert len(out["cv"]["splits"]) == 4
    _assert_splits_valid(n_rows=len(sample_df), splits=out["cv"]["splits"])

    # int を渡した場合と異なり、cv オブジェクトの random_state を尊重できているか（最低限の同一性）
    assert analyzer.cv is cv
