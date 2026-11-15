"""Offline ML training pipeline for AtmoGuard (Phase 4).

Everything in this package is a training-time tool: it reads the historical
CSV dataset, engineers features, trains and evaluates models, and writes
artifacts to `backend/ml/artifacts/`. It is never imported by the running
FastAPI app — the app only loads the artifacts this package produces, via
`app/ml/model_registry.py`. This separation is deliberate (see CLAUDE.md /
docs/architecture.md, "Phase 4" / "Model management").

Run with: `python -m ml.train` from `backend/`, inside the venv.
"""
