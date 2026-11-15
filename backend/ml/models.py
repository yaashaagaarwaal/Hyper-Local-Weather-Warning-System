"""Model factories. One place that knows how each algorithm is constructed,
so train.py stays a clean loop over (target, horizon, algorithm).

Algorithm roster and why each one is here (full reasoning in
docs/model_report.md "Architecture decision"):

- persistence: the mandatory naive baseline ("what did it look like right
  now") -- every other model must beat this to be worth deploying.
- linear / logistic: a second, weaker baseline representing the simplest
  possible learned (non-naive) model.
- random_forest: a non-linear tabular baseline, robust to the feature scale
  and the heavy class imbalance in the hazard targets.
- xgboost: the advanced model this pipeline is built around -- gradient-
  boosted trees are the standard choice for structured/tabular weather
  nowcasting at this data scale, and remain inspectable (feature
  importances) consistent with this project's transparency requirement.
- mlp: a small feed-forward neural network, included specifically to give
  the "evaluate suitable architectures" requirement a genuine empirical data
  point for a neural approach, without building an unjustified ConvLSTM /
  ConvLSTM-scale architecture this dataset's structure (5 point stations in
  one city, ~175k rows, no raster grid) does not support. See
  docs/model_report.md for the full ConvLSTM/CNN-LSTM/Transformer rejection
  reasoning.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

from ml.config import RANDOM_SEED

ALL_ALGORITHMS = ["persistence", "linear", "random_forest", "xgboost", "mlp"]
# Algorithms run at every horizon (not just the flagship one).
REDUCED_ALGORITHMS = ["persistence", "xgboost"]


def get_regressor(algorithm: str):
    if algorithm == "linear":
        return LinearRegression()
    if algorithm == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    if algorithm == "xgboost":
        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    if algorithm == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            early_stopping=True,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"no regressor for algorithm={algorithm!r} (use get_persistence for 'persistence')")


def get_classifier(algorithm: str, scale_pos_weight: float = 1.0):
    """`scale_pos_weight` (negatives/positives in the TRAIN split only) is
    XGBoost's equivalent of sklearn's `class_weight="balanced"` -- it has no
    such string option, so the caller computes the ratio from the training
    labels and passes it in. Both hazard targets here are heavily imbalanced
    (see docs/model_report.md); without this, a classifier could reach high
    accuracy by always predicting "no hazard".
    """
    if algorithm == "linear":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    if algorithm == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
    if algorithm == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            early_stopping=True,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"no classifier for algorithm={algorithm!r} (use get_persistence for 'persistence')")
