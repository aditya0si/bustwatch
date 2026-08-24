"""Unit and integration tests for evals benchmark suite."""

import pytest
import json
from pathlib import Path
from evals.evaluate_models import run_evaluations


def test_run_evaluations_end_to_end(tmp_path):
    """Verify that run_evaluations executes and exports valid metrics."""
    metrics = run_evaluations(
        n_samples=100,
        output_dir=tmp_path / "evals",
        random_seed=42,
    )

    assert "metrics_summary" in metrics
    assert "dataset" in metrics
    assert "lead_time_breakdown" in metrics
    assert "reliability_curve" in metrics

    # Check files created
    assert (tmp_path / "evals" / "metrics.json").exists()
    assert (tmp_path / "evals" / "reliability_diagram.png").exists()

    with open(tmp_path / "evals" / "metrics.json") as f:
        loaded = json.load(f)
    assert loaded["dataset"]["n_samples"] == 100
