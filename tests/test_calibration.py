"""Unit tests for calibration algorithms and verification metrics."""

import numpy as np

from bustwatch.calibration.calibrator import (
    ProbabilityCalibrator,
    compute_brier_score,
    compute_brier_skill_score,
    compute_expected_calibration_error,
    compute_reliability_curve,
)


def test_brier_score_bounds():
    """Verify Brier Score boundaries and mathematical behavior."""
    # Perfect forecast
    assert compute_brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0
    # Worst possible forecast
    assert compute_brier_score([0.0, 1.0, 0.0], [1, 0, 1]) == 1.0
    # Intermediate values
    bs = compute_brier_score([0.8, 0.2, 0.6], [1, 0, 0])
    assert 0.0 < bs < 1.0


def test_brier_skill_score():
    """Verify Brier Skill Score calculation against climatology."""
    y_true = [1, 1, 0, 0]
    # Perfect predictions give BSS = 1.0
    bss_perfect = compute_brier_skill_score([1.0, 1.0, 0.0, 0.0], y_true)
    assert np.isclose(bss_perfect, 1.0)

    # Climatology predictions give BSS = 0.0
    bss_climatology = compute_brier_skill_score([0.5, 0.5, 0.5, 0.5], y_true)
    assert np.isclose(bss_climatology, 0.0, atol=1e-3)


def test_expected_calibration_error():
    """Verify ECE calculation for calibrated vs uncalibrated predictions."""
    # Perfectly calibrated
    ece_perf = compute_expected_calibration_error([0.5, 0.5], [1, 0], n_bins=2)
    assert np.isclose(ece_perf, 0.0, atol=1e-3)

    # Miscalibrated
    ece_bad = compute_expected_calibration_error([0.9, 0.9], [0, 0], n_bins=2)
    assert ece_bad > 0.5


def test_probability_calibrator_isotonic():
    """Verify Isotonic probability calibration transformer."""
    cal = ProbabilityCalibrator(method="isotonic")
    y_prob = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_true = np.array([0, 0, 1, 1, 1])

    cal.fit(y_prob, y_true)
    calibrated = cal.transform([0.2, 0.8])
    assert len(calibrated) == 2
    assert (calibrated >= 0.0).all() and (calibrated <= 1.0).all()


def test_probability_calibrator_sigmoid():
    """Verify Platt scaling (sigmoid) calibrator."""
    cal = ProbabilityCalibrator(method="sigmoid")
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    y_true = np.array([0, 0, 0, 1, 1, 1])

    cal.fit(y_prob, y_true)
    calibrated = cal.transform([0.15, 0.85])
    assert len(calibrated) == 2
    assert (calibrated >= 0.0).all() and (calibrated <= 1.0).all()


def test_reliability_curve():
    """Verify reliability curve binning structure."""
    y_prob = [0.1, 0.2, 0.45, 0.55, 0.8, 0.9]
    y_true = [0, 0, 0, 1, 1, 1]
    rel = compute_reliability_curve(y_prob, y_true, n_bins=5)

    assert "bin_centers" in rel
    assert "fraction_positives" in rel
    assert "mean_predicted_probs" in rel
    assert "bin_counts" in rel
    assert len(rel["bin_centers"]) == 5
    assert sum(rel["bin_counts"]) == len(y_prob)
