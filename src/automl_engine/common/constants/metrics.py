"""
評価指標レジストリ.
各エントリは {"label": str, "scorer": ...} の形を取る.

scorer は以下を許容:
- str: sklearn の scoring 名
- callable: sklearn が受け付ける scorer callable（estimator, X, y -> float）
- dict: make_scorer 用定義 {"score_func": callable, "greater_is_better": bool, "response_method": str, "kwargs": dict}
"""

from __future__ import annotations

from typing import Final

REGRESSION_METRIC_REGISTRY: Final[dict[str, dict[str, object]]] = {
    "r2": {"label": "R2", "scorer": "r2"},
    "mae": {"label": "MAE", "scorer": "neg_mean_absolute_error"},
    "rmse": {"label": "RMSE", "scorer": "neg_root_mean_squared_error"},
    # --- カスタム例 ---
    # "my_metric": {
    #     "label": "My Metric",
    #     "scorer": {
    #         "score_func": my_metric_func,
    #         "greater_is_better": True,
    #         "response_method": "predict",
    #         "kwargs": {},
    #     },
    # },
}

CLASSIFICATION_METRIC_REGISTRY: Final[dict[str, dict[str, object]]] = {
    "accuracy": {"label": "Accuracy", "scorer": "accuracy"},
    "f1_macro": {"label": "F1 (macro)", "scorer": "f1_macro"},
    "precision_macro": {"label": "Precision (macro)", "scorer": "precision_macro"},
    "recall_macro": {"label": "Recall (macro)", "scorer": "recall_macro"},
    "roc_auc_ovr": {"label": "ROC-AUC (ovr)", "scorer": "roc_auc_ovr"},
    # --- カスタム例 ---
    # "my_auc": {
    #     "label": "My AUC",
    #     "scorer": {
    #         "score_func": my_auc_func,
    #         "greater_is_better": True,
    #         "response_method": "predict_proba",
    #         "kwargs": {},
    #     },
    # },
}
