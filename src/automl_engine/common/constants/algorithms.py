"""
AutoML 実行時に使用可能な分類モデルを辞書形式で定義する。
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR

# 分類モデル定義レジストリ。
CLASSIFICATION_MODEL_REGISTRY = {
    "logreg": {
        "display_name": "Logistic Regression",
        "estimator_cls": LogisticRegression,
        "init_params": {
            "max_iter": 5000,
            "n_jobs": None,
        },
        "fit_params": {},
        "search_space": {
            "grid": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
            "optuna": {
                "C": {"type": "loguniform", "low": 1e-3, "high": 1e2},
            },
        },
    },
    "svm": {
        "display_name": "Support Vector Machine (RBF)",
        "estimator_cls": SVC,
        "init_params": {
            "kernel": "rbf",
            "probability": True,
        },
        "fit_params": {},
        "search_space": {
            "grid": {
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", "auto"],
            },
            "optuna": {
                "C": {"type": "loguniform", "low": 1e-2, "high": 1e2},
                "gamma": {
                    "type": "categorical",
                    "choices": ["scale", "auto"],
                },
            },
        },
    },
    "rf": {
        "display_name": "Random Forest Classifier",
        "estimator_cls": RandomForestClassifier,
        "init_params": {
            "random_state": None,
            "n_jobs": -1,
        },
        "fit_params": {},
        "search_space": {
            "grid": {
                "n_estimators": [100, 300],
                "max_depth": [None, 5, 15],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [1, 4],
            },
            "optuna": {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_depth": {"type": "int_or_none", "low": 2, "high": 30},
                "min_samples_split": {"type": "int", "low": 2, "high": 30},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
                "max_features": {
                    "type": "categorical",
                    "choices": ["sqrt", "log2", None],
                },
            },
        },
    },
    "nb": {
        "display_name": "Gaussian Naive Bayes",
        "estimator_cls": GaussianNB,
        "init_params": {},
        "fit_params": {},
        "search_space": {
            "grid": {
                "var_smoothing": [1e-9, 1e-8, 1e-7],
            },
            "optuna": {
                "var_smoothing": {
                    "type": "loguniform",
                    "low": 1e-10,
                    "high": 1e-6,
                },
            },
        },
    },
}


# 回帰モデル定義レジストリ。
REGRESSION_MODEL_REGISTRY = {
    "lr": {
        "display_name": "Linear Regression",
        "estimator_cls": LinearRegression,
        "init_params": {
            "fit_intercept": True,
            "n_jobs": None,
        },
        "fit_params": {},
        "search_space": {
            "grid": {
                "fit_intercept": [True, False],
            },
            "optuna": {
                "fit_intercept": {
                    "type": "categorical",
                    "choices": [True, False],
                },
            },
        },
    },
    "ridge": {
        "display_name": "Ridge Regression",
        "estimator_cls": Ridge,
        "init_params": {
            "random_state": None,
        },
        "fit_params": {},
        "search_space": {
            "grid": {
                "alpha": [0.1, 1.0, 10.0],
                "fit_intercept": [True, False],
                "solver": ["auto", "svd", "lsqr"],
            },
            "optuna": {
                "alpha": {"type": "loguniform", "low": 1e-3, "high": 1e2},
                "fit_intercept": {
                    "type": "categorical",
                    "choices": [True, False],
                },
                "solver": {
                    "type": "categorical",
                    "choices": ["auto", "svd", "lsqr", "sag"],
                },
            },
        },
    },
    "lasso": {
        "display_name": "Lasso Regression",
        "estimator_cls": Lasso,
        "init_params": {
            "random_state": None,
            "max_iter": 5000,
        },
        "fit_params": {},
        "search_space": {
            "grid": {
                "alpha": [1e-3, 1e-2, 1e-1, 1.0],
                "fit_intercept": [True, False],
                "selection": ["cyclic", "random"],
            },
            "optuna": {
                "alpha": {"type": "loguniform", "low": 1e-4, "high": 1e1},
                "fit_intercept": {
                    "type": "categorical",
                    "choices": [True, False],
                },
                "selection": {
                    "type": "categorical",
                    "choices": ["cyclic", "random"],
                },
            },
        },
    },
    "elastic_net": {
        "display_name": "Elastic Net",
        "estimator_cls": ElasticNet,
        "init_params": {
            "random_state": None,
            "max_iter": 5000,
        },
        "fit_params": {},
        "search_space": {
            "grid": {
                "alpha": [1e-3, 1e-2, 1e-1, 1.0],
                "l1_ratio": [0.1, 0.5, 0.9],
                "fit_intercept": [True, False],
            },
            "optuna": {
                "alpha": {"type": "loguniform", "low": 1e-4, "high": 1e1},
                "l1_ratio": {"type": "uniform", "low": 0.0, "high": 1.0},
                "fit_intercept": {
                    "type": "categorical",
                    "choices": [True, False],
                },
            },
        },
    },
    "svr": {
        "display_name": "Support Vector Regression (RBF)",
        "estimator_cls": SVR,
        "init_params": {
            "kernel": "rbf",
        },
        "fit_params": {},
        "search_space": {
            "grid": {
                "C": [0.1, 1.0, 10.0],
                "epsilon": [0.01, 0.1, 0.2],
                "gamma": ["scale", "auto"],
            },
            "optuna": {
                "C": {"type": "loguniform", "low": 1e-2, "high": 1e2},
                "epsilon": {"type": "loguniform", "low": 1e-3, "high": 1.0},
                "gamma": {
                    "type": "categorical",
                    "choices": ["scale", "auto"],
                },
            },
        },
    },
    "rf": {
        "display_name": "Random Forest Regressor",
        "estimator_cls": RandomForestRegressor,
        "init_params": {
            "random_state": None,
            "n_jobs": -1,
        },
        "fit_params": {},
        "search_space": {
            "grid": {
                "n_estimators": [100, 300],
                "max_depth": [None, 5, 15],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [1, 4],
            },
            "optuna": {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_depth": {"type": "int_or_none", "low": 2, "high": 30},
                "min_samples_split": {"type": "int", "low": 2, "high": 30},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
                "max_features": {
                    "type": "categorical",
                    "choices": ["sqrt", "log2", None],
                },
            },
        },
    },
}
