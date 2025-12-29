# 並列実行デモ

このディレクトリには、`automl_engine.ml.training.supervised.run_supervised`の並列実行機能のデモが含まれています。

## parallel_demo.py

基本的な並列実行の使用例を示すスクリプトです。

### 実行方法

```bash
python examples/parallel_demo.py
```

### 機能

- 3つのアルゴリズム（LogisticRegression, RandomForest, DecisionTree）を並列実行
- 2つの評価指標（accuracy, f1）で評価
- joblib バックエンドを使用
- 進捗表示とロギングの例

### 出力例

```
================================================================================
run_supervised 並列実行デモ
================================================================================

1. データセットを生成中...
   データサイズ: X=(200, 20), y=(200,)

2. アルゴリズムを定義中...
   アルゴリズム数: 3
   アルゴリズム: ['LogisticRegression', 'RandomForest', 'DecisionTree']

3. 評価指標を定義中...
   評価指標: ['accuracy', 'f1']

4. 並列実行中（joblib バックエンド）...
   n_jobs=2（アルゴリズム並列数）
   n_jobs_cv=1（CV並列数）
--------------------------------------------------------------------------------
アルゴリズム実行: 100%|██████████| 3/3 [00:01<00:00,  2.16algo/s]

5. 結果:
--------------------------------------------------------------------------------

   アルゴリズム: LogisticRegression
   探索手法: grid
   CV スコア (平均):
      accuracy: 0.8100
      f1: 0.8045

   アルゴリズム: RandomForest
   探索手法: grid
   CV スコア (平均):
      accuracy: 0.8250
      f1: 0.8237

   アルゴリズム: DecisionTree
   探索手法: grid
   CV スコア (平均):
      accuracy: 0.7200
      f1: 0.7169

================================================================================
デモ完了
================================================================================
```

## 並列実行パラメータの説明

### n_jobs
アルゴリズムを並列実行するワーカー数を指定します。
- `-1`: すべてのCPUコアを使用（デフォルト）
- `1`: シーケンシャル実行（並列化なし）
- `2以上`: 指定した数のワーカーを使用

### n_jobs_cv
クロスバリデーションの並列数を指定します。
- `None`: 自動計算（`cpu_count / n_jobs`）（デフォルト）
- `1以上`: 指定した並列数を使用

### parallel_backend
並列実行のバックエンドを選択します。
- `"joblib"`: joblib（loky）を使用（デフォルト）
- `"concurrent"`: concurrent.futures（ThreadPoolExecutor）を使用
- `"ray"`: Ray を使用（要インストール）

### algorithm_timeout
アルゴリズムごとのタイムアウト（秒）を指定します。
- `None`: タイムアウトなし（デフォルト）
- `正の整数`: 指定した秒数でタイムアウト

## 使用例

### 基本的な使用例

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from automl_engine.ml.training.supervised import run_supervised

# データの準備
X, y = make_classification(n_samples=100, n_features=10, random_state=42)

# アルゴリズムの定義
algorithms = {
    "LogisticRegression": {
        "estimator_cls": lambda: LogisticRegression(max_iter=1000, random_state=42)
    },
    "RandomForest": {
        "estimator_cls": lambda: RandomForestClassifier(n_estimators=50, random_state=42)
    },
}

# 評価指標の定義
metrics = {
    "accuracy": {
        "scorer": {
            "score_func": accuracy_score,
            "greater_is_better": True,
        }
    }
}

# CV の定義
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 並列実行
estimators, results = run_supervised(
    X=X,
    y=y,
    algorithms=algorithms,
    metrics=metrics,
    primary_metric_key="accuracy",
    cv=cv,
    search_method="grid",
    n_jobs=2,  # 2つのワーカーを使用
    n_jobs_cv=1,  # CVは並列化しない
    parallel_backend="joblib",
)
```

### concurrent.futures バックエンドの使用例

```python
estimators, results = run_supervised(
    X=X,
    y=y,
    algorithms=algorithms,
    metrics=metrics,
    primary_metric_key="accuracy",
    cv=cv,
    search_method="grid",
    n_jobs=3,
    parallel_backend="concurrent",  # ThreadPoolExecutor を使用
)
```

### タイムアウトの使用例

```python
estimators, results = run_supervised(
    X=X,
    y=y,
    algorithms=algorithms,
    metrics=metrics,
    primary_metric_key="accuracy",
    cv=cv,
    search_method="grid",
    n_jobs=2,
    algorithm_timeout=60,  # 各アルゴリズムは60秒でタイムアウト
    parallel_backend="joblib",
)
```

## 注意事項

1. **lambda関数の制限**: ProcessPoolExecutorを使用する場合、lambda関数はピクル化できないため、ThreadPoolExecutorまたはjoblibを使用してください。

2. **リソース管理**: `n_jobs`と`n_jobs_cv`の設定により、使用するCPUリソースが変わります。適切な値を設定してください。

3. **Ray バックエンド**: Rayを使用する場合は、事前に`pip install ray`でインストールが必要です。

4. **タイムアウト**: joblibのSequentialBackendではタイムアウトがサポートされていません（`n_jobs=1`の場合）。
