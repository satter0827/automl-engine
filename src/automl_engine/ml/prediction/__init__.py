"""
推論パッケージ。
学習済み成果物を用いた推論処理を提供する。
"""

from automl_engine.ml.prediction.predictor import predict_supervised

__all__: list[str] = ["predict_supervised"]
