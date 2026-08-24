"""Comprehensive Evaluation & Reliability Verification Benchmark Suite for BustWatch."""

from __future__ import annotations
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from bustwatch.config import settings
from bustwatch.data.synthetic import generate_synthetic_dataset
from bustwatch.data.features import FEATURE_COLUMNS, compute_bust_labels
from bustwatch.models.bust_classifier import ForecastBustClassifier
from bustwatch.models.error_regressor import QuantileErrorRegressor
from bustwatch.calibration.calibrator import (
    compute_brier_score,
    compute_brier_skill_score,
    compute_expected_calibration_error,
    compute_reliability_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_reliability_plot(
    reliability_uncal: Dict[str, Any],
    reliability_cal: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate professional reliability diagram with bin count subplots."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Diagonal reference line (perfect reliability)
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Reliability (y = x)")

    # Uncalibrated curve
    ax1.plot(
        reliability_uncal["mean_predicted_probs"],
        reliability_uncal["fraction_positives"],
        "s-",
        color="#ef4444",
        label=f"Uncalibrated GBDT (ECE = {reliability_uncal['ece']:.3f}, BS = {reliability_uncal['brier_score']:.3f})",
        linewidth=2,
        markersize=6,
    )

    # Calibrated curve
    ax1.plot(
        reliability_cal["mean_predicted_probs"],
        reliability_cal["fraction_positives"],
        "o-",
        color="#10b981",
        label=f"Calibrated (Isotonic) (ECE = {reliability_cal['ece']:.3f}, BS = {reliability_cal['brier_score']:.3f})",
        linewidth=2.5,
        markersize=7,
    )

    ax1.set_ylabel("Observed Relative Frequency", fontsize=11)
    ax1.set_title("BustWatch Forecast Bust Probability Reliability Diagram", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Subplot 2: Forecast frequency histogram
    bin_centers = reliability_cal["bin_centers"]
    width = 0.04
    ax2.bar(
        [x - 0.02 for x in bin_centers],
        reliability_uncal["bin_counts"],
        width=width,
        color="#ef4444",
        alpha=0.6,
        label="Uncalibrated",
    )
    ax2.bar(
        [x + 0.02 for x in bin_centers],
        reliability_cal["bin_counts"],
        width=width,
        color="#10b981",
        alpha=0.6,
        label="Calibrated",
    )
    ax2.set_xlabel("Forecast Bust Probability", fontsize=11)
    ax2.set_ylabel("Sample Count", fontsize=10)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Reliability diagram saved to {output_path}")


def run_evaluations(
    n_samples: int = 1500,
    model_dir: Path = settings.model_dir,
    output_dir: Path = settings.evals_dir,
    random_seed: int = 999,
) -> Dict[str, Any]:
    """Execute end-to-end evaluation benchmark across test dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load or train model
    clf_path = Path(model_dir) / "bust_classifier.joblib"
    if clf_path.exists():
        logger.info(f"Loading classifier from {clf_path}...")
        classifier = ForecastBustClassifier.load(clf_path)
    else:
        logger.info("Classifier artifact not found; training inline...")
        from scripts.train_models import train_and_save_models
        train_and_save_models(n_samples=2000, output_dir=model_dir, random_seed=42)
        classifier = ForecastBustClassifier.load(clf_path)

    # 2. Generate independent evaluation dataset
    logger.info(f"Generating independent verification dataset ({n_samples} samples, seed={random_seed})...")
    eval_df = generate_synthetic_dataset(n_samples=n_samples, random_seed=random_seed)
    eval_df = compute_bust_labels(eval_df)

    X_eval = eval_df[FEATURE_COLUMNS]
    y_true = eval_df["is_bust"].values

    # 3. Model Predictions
    raw_probs = classifier.predict_proba_raw(X_eval)
    cal_probs = classifier.predict_proba(X_eval)

    # 4. Verification Metrics
    bs_uncal = compute_brier_score(raw_probs, y_true)
    bs_cal = compute_brier_score(cal_probs, y_true)
    bss_uncal = compute_brier_skill_score(raw_probs, y_true)
    bss_cal = compute_brier_skill_score(cal_probs, y_true)
    ece_uncal = compute_expected_calibration_error(raw_probs, y_true, n_bins=10)
    ece_cal = compute_expected_calibration_error(cal_probs, y_true, n_bins=10)
    roc_auc = float(roc_auc_score(y_true, cal_probs))
    pr_auc = float(average_precision_score(y_true, cal_probs))

    # Lead-time stratification
    eval_df["raw_prob"] = raw_probs
    eval_df["cal_prob"] = cal_probs

    lead_time_breakdown: Dict[str, Dict[str, float]] = {}
    for regime, (d_start, d_end) in [
        ("Day 1–3 (Early)", (1, 3)),
        ("Day 4–7 (Medium)", (4, 7)),
        ("Day 8–10 (Extended)", (8, 10)),
    ]:
        subset = eval_df[(eval_df["lead_time_days"] >= d_start) & (eval_df["lead_time_days"] <= d_end)]
        if len(subset) > 0:
            sub_y = subset["is_bust"].values
            sub_p = subset["cal_prob"].values
            lead_time_breakdown[regime] = {
                "sample_count": int(len(subset)),
                "bust_rate": float(np.mean(sub_y)),
                "brier_score": float(compute_brier_score(sub_p, sub_y)),
                "brier_skill_score": float(compute_brier_skill_score(sub_p, sub_y)),
                "ece": float(compute_expected_calibration_error(sub_p, sub_y, n_bins=5)),
                "roc_auc": float(roc_auc_score(sub_y, sub_p)) if len(np.unique(sub_y)) > 1 else 1.0,
            }

    # Reliability curves
    rel_uncal = compute_reliability_curve(raw_probs, y_true, n_bins=10)
    rel_cal = compute_reliability_curve(cal_probs, y_true, n_bins=10)

    # Save Reliability Diagram Plot
    plot_path = output_dir / "reliability_diagram.png"
    generate_reliability_plot(rel_uncal, rel_cal, plot_path)

    # Feature Importance Summary
    feat_imp = classifier.get_feature_importances()

    metrics_payload = {
        "dataset": {
            "n_samples": n_samples,
            "overall_bust_rate": float(np.mean(y_true)),
            "lead_time_range_days": [1, 10],
        },
        "metrics_summary": {
            "uncalibrated": {
                "brier_score": round(bs_uncal, 4),
                "brier_skill_score": round(bss_uncal, 4),
                "expected_calibration_error": round(ece_uncal, 4),
            },
            "calibrated": {
                "brier_score": round(bs_cal, 4),
                "brier_skill_score": round(bss_cal, 4),
                "expected_calibration_error": round(ece_cal, 4),
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
            },
            "calibration_improvement": {
                "brier_score_reduction_pct": round(((bs_uncal - bs_cal) / max(1e-6, bs_uncal)) * 100, 2),
                "ece_reduction_pct": round(((ece_uncal - ece_cal) / max(1e-6, ece_uncal)) * 100, 2),
            },
        },
        "lead_time_breakdown": lead_time_breakdown,
        "feature_importances": {k: round(v, 4) for k, v in sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)},
        "reliability_curve": {
            "bin_centers": [round(x, 3) for x in rel_cal["bin_centers"]],
            "calibrated_observed_freq": [round(x, 3) for x in rel_cal["fraction_positives"]],
            "calibrated_mean_predicted": [round(x, 3) for x in rel_cal["mean_predicted_probs"]],
            "bin_counts": rel_cal["bin_counts"],
        },
    }

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info(f"Metrics saved to {metrics_file}")
    _print_metrics_summary(metrics_payload)
    return metrics_payload


def _print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """Print ASCII verification table to stdout."""
    ms = metrics["metrics_summary"]
    print("\n" + "=" * 68)
    print("      BUSTWATCH NWP FORECAST BUST DETECTION VERIFICATION SUMMARY")
    print("=" * 68)
    print(f" Dataset Size: {metrics['dataset']['n_samples']} samples | Base Bust Rate: {metrics['dataset']['overall_bust_rate']:.2%}")
    print("-" * 68)
    print(f" {'Metric':<28} | {'Uncalibrated':<14} | {'Calibrated':<14}")
    print("-" * 68)
    print(f" {'Brier Score (BS)':<28} | {ms['uncalibrated']['brier_score']:<14.4f} | {ms['calibrated']['brier_score']:<14.4f}")
    print(f" {'Brier Skill Score (BSS)':<28} | {ms['uncalibrated']['brier_skill_score']:<14.4f} | {ms['calibrated']['brier_skill_score']:<14.4f}")
    print(f" {'Expected Calib. Error (ECE)':<28} | {ms['uncalibrated']['expected_calibration_error']:<14.4f} | {ms['calibrated']['expected_calibration_error']:<14.4f}")
    print(f" {'ROC-AUC Score':<28} | {'-':<14} | {ms['calibrated']['roc_auc']:<14.4f}")
    print(f" {'PR-AUC Score':<28} | {'-':<14} | {ms['calibrated']['pr_auc']:<14.4f}")
    print("-" * 68)
    print(f" ECE Improvement: {ms['calibration_improvement']['ece_reduction_pct']:.1f}% reduction in calibration error")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BustWatch forecast bust detection models.")
    parser.add_argument("--n-samples", type=int, default=1500, help="Number of test samples")
    parser.add_argument("--model-dir", type=str, default=str(settings.model_dir), help="Directory of saved models")
    parser.add_argument("--output-dir", type=str, default=str(settings.evals_dir), help="Output directory for metrics")
    args = parser.parse_args()

    run_evaluations(
        n_samples=args.n_samples,
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
    )
