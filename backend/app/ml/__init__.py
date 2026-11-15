"""Inference-only ML code: loads artifacts produced by `backend/ml/`
(training) and turns live weather history into predictions. Never trains,
never touches the raw CSV dataset, never imports pandas/sklearn/xgboost's
*training* APIs directly beyond what's needed to deserialize and call
`.predict()` on an already-fitted model. See CLAUDE.md "Model management"
for why training and inference are kept in separate packages.
"""
