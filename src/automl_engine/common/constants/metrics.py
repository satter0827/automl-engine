"""
評価指標定義レジストリ
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

# 分類指標定義レジストリ
CLASSIFICATION_METRIC_REGISTRY: dict[str, dict[str, Any]] = {
    "accuracy": {
        "display_name": "Accuracy",
        "direction": "maximize",
        "scorer": accuracy_score,
        "greater_is_better": True,
    },
    "f1_macro": {
        "display_name": "F1 (macro)",
        "direction": "maximize",
        "scorer": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
        "greater_is_better": True,
    },
    "f1_weighted": {
        "display_name": "F1 (weighted)",
        "direction": "maximize",
        "scorer": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
        "greater_is_better": True,
    },
    "roc_auc_ovr": {
        "display_name": "ROC-AUC (OvR)",
        "direction": "maximize",
        "scorer": lambda y_true, y_score: roc_auc_score(
            y_true, y_score, multi_class="ovr"
        ),
        "greater_is_better": True,
    },
}


# 回帰指標定義レジストリ
REGRESSION_METRIC_REGISTRY: dict[str, dict[str, Any]] = {
    "rmse": {
        "display_name": "RMSE",
        "direction": "minimize",
        "scorer": lambda y_true, y_pred: root_mean_squared_error(
            y_true, y_pred, squared=False
        ),
        "greater_is_better": False,
    },
    "mse": {
        "display_name": "MSE",
        "direction": "minimize",
        "scorer": root_mean_squared_error,
        "greater_is_better": False,
    },
    "mae": {
        "display_name": "MAE",
        "direction": "minimize",
        "scorer": mean_absolute_error,
        "greater_is_better": False,
    },
    "r2": {
        "display_name": "R2",
        "direction": "maximize",
        "scorer": r2_score,
        "greater_is_better": True,
    },
}
