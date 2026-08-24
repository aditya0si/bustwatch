"""Probability calibration (Isotonic / Platt) and reliability verification metrics."""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple, Any, Literal, Union
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Fits and transforms uncalibrated GBDT raw confidence scores into well-calibrated probabilities.
    Supports Isotonic Regression (non-parametric) and Platt Scaling (parametric sigmoid).
    """

    def __init__(self, method: Literal["isotonic", "sigmoid"] = "isotonic"):
        self.method = method
        self.is_fitted = False
        if self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        elif self.method == "sigmoid":
            self.calibrator_ = LogisticRegression(C=1.0, solver="lbfgs")
        else:
            raise ValueError(f"Unknown calibration method: {method}. Must be 'isotonic' or 'sigmoid'.")

    def fit(self, y_prob: Union[np.ndarray, List[float]], y_true: Union[np.ndarray, List[int]]) -> ProbabilityCalibrator:
        """Fit calibrator on validation probabilities and true binary outcomes."""
        probs = np.asarray(y_prob, dtype=np.float64).ravel()
        labels = np.asarray(y_true, dtype=int).ravel()

        if len(probs) == 0:
            raise ValueError("Empty probability array passed to calibrator fit.")

        if self.method == "isotonic":
            self.calibrator_.fit(probs, labels)
        else:
            self.calibrator_.fit(probs.reshape(-1, 1), labels)

        self.is_fitted = True
        return self

    def transform(self, y_prob: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Transform uncalibrated probabilities to calibrated probabilities in [0, 1]."""
        probs = np.asarray(y_prob, dtype=np.float64).ravel()
        if not self.is_fitted:
            return np.clip(probs, 0.0, 1.0)

        if self.method == "isotonic":
            calibrated = self.calibrator_.transform(probs)
        else:
            calibrated = self.calibrator_.predict_proba(probs.reshape(-1, 1))[:, 1]

        return np.clip(calibrated, 0.0, 1.0)


def compute_brier_score(y_prob: Union[np.ndarray, List[float]], y_true: Union[np.ndarray, List[int]]) -> float:
    """
    Calculate Mean Squared Probability Error (Brier Score).
    BS = (1/N) * sum((p_i - y_i)^2)
    Ranges from 0.0 (perfect) to 1.0 (worst).
    """
    probs = np.asarray(y_prob, dtype=np.float64).ravel()
    labels = np.asarray(y_true, dtype=np.float64).ravel()
    if len(probs) == 0:
        return 0.0
    return float(np.mean((probs - labels) ** 2))


def compute_brier_skill_score(
    y_prob: Union[np.ndarray, List[float]],
    y_true: Union[np.ndarray, List[int]],
    y_ref_prob: Union[np.ndarray, List[float], None] = None,
) -> float:
    """
    Compute Brier Skill Score (BSS) relative to a reference climatology baseline.
    BSS = 1 - (BS_forecast / BS_reference)
    BSS > 0 means improvement over climatology. Perfect forecast has BSS = 1.0.
    """
    bs = compute_brier_score(y_prob, y_true)
    labels = np.asarray(y_true, dtype=np.float64).ravel()
    if len(labels) == 0:
        return 0.0

    if y_ref_prob is None:
        # Climatological base rate: p_ref = mean(y_true)
        p_base = np.mean(labels)
        bs_ref = float(np.mean((p_base - labels) ** 2))
    else:
        bs_ref = compute_brier_score(y_ref_prob, y_true)

    if bs_ref < 1e-9:
        return 0.0  # Avoid division by zero when no events or all events occur

    return float(1.0 - (bs / bs_ref))


def compute_expected_calibration_error(
    y_prob: Union[np.ndarray, List[float]],
    y_true: Union[np.ndarray, List[int]],
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    ECE = sum_{m=1}^M (|B_m|/N) * |acc(B_m) - conf(B_m)|
    """
    probs = np.asarray(y_prob, dtype=np.float64).ravel()
    labels = np.asarray(y_true, dtype=int).ravel()
    n_total = len(probs)
    if n_total == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            mask = (probs >= bin_lower) & (probs < bin_upper)

        count = np.sum(mask)
        if count > 0:
            bin_conf = np.mean(probs[mask])
            bin_acc = np.mean(labels[mask])
            ece += (count / n_total) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_reliability_curve(
    y_prob: Union[np.ndarray, List[float]],
    y_true: Union[np.ndarray, List[int]],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute bin statistics for reliability diagrams (calibration curve).
    Returns bin centers, mean predicted probability, observed fraction, and counts.
    """
    probs = np.asarray(y_prob, dtype=np.float64).ravel()
    labels = np.asarray(y_true, dtype=int).ravel()
    n_total = len(probs)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers: List[float] = []
    mean_predicted_probs: List[float] = []
    fraction_positives: List[float] = []
    bin_counts: List[int] = []

    for i in range(n_bins):
        bin_lower, bin_upper = bin_edges[i], bin_edges[i + 1]
        bin_center = (bin_lower + bin_upper) / 2.0
        if i == n_bins - 1:
            mask = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            mask = (probs >= bin_lower) & (probs < bin_upper)

        count = int(np.sum(mask))
        bin_centers.append(float(bin_center))
        bin_counts.append(count)

        if count > 0:
            mean_predicted_probs.append(float(np.mean(probs[mask])))
            fraction_positives.append(float(np.mean(labels[mask])))
        else:
            mean_predicted_probs.append(float(bin_center))
            fraction_positives.append(float(bin_center))  # Default for empty bin

    return {
        "n_bins": n_bins,
        "bin_edges": [float(x) for x in bin_edges],
        "bin_centers": bin_centers,
        "mean_predicted_probs": mean_predicted_probs,
        "fraction_positives": fraction_positives,
        "bin_counts": bin_counts,
        "brier_score": compute_brier_score(probs, labels),
        "brier_skill_score": compute_brier_skill_score(probs, labels),
        "ece": compute_expected_calibration_error(probs, labels, n_bins=n_bins),
    }
