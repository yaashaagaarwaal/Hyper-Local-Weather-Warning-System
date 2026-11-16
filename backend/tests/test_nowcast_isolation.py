"""Runs the PyTorch engine tests (`tests_torch/`) in their own process.

They cannot be collected into this session: importing torch here would put
PyTorch and XGBoost in one process, which segfaults on macOS (see
app/services/imagery/nowcast_worker.py). `tests_torch/` has no conftest, so
the child run never touches the test database this session is using.
"""
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")
def test_torch_engine_suite_passes_in_its_own_process():
    env = {**os.environ, "PYTHONPATH": str(BACKEND)}
    env.pop("NOWCAST_MODEL_DIR", None)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", "tests_torch", "-q", "-p", "no:cacheprovider"],
        cwd=BACKEND, env=env, capture_output=True, text=True, timeout=900,
    )
    tail = result.stdout[-4000:] + result.stderr[-1500:]
    assert result.returncode == 0, tail
    passed = re.search(r"(\d+) passed", result.stdout)
    assert passed and int(passed.group(1)) >= 30, tail
    assert "failed" not in result.stdout.splitlines()[-1], tail
    print(f"tests_torch: {result.stdout.strip().splitlines()[-1]}")


def test_this_session_has_not_imported_torch():
    assert "torch" not in sys.modules
