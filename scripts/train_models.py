"""Model training pipeline for BustWatch classifiers, regressors, and probability calibrators."""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from bustwatch.config import settings
from bustwatch.data.synthetic import generate_synthetic_dataset
from bustwatch.data.features import FEATURE_COLUMNS, compute_bust_labels
from bustwatch.models.bust_classifier import ForecastBustClassifier
from bustwatch.models.error_regressor import QuantileErrorRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_save_models(
    n_samples: int = 3000,
    output_dir: Path = settings.model_dir,
    calibration_method: str = settings.calibration_method,
    random_seed: int = settings.random_seed,
) -> None:
    """Generate training data, train GBDT models, fit calibrators, and serialize artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating training dataset with {n_samples} meteorological forecast-observation pairs...")
    df = generate_synthetic_dataset(n_samples=n_samples, random_seed=random_seed)
    df = compute_bust_labels(df)

    X = df[FEATURE_COLUMNS]
    y_bust = df["is_bust"]
    y_z500_err = df["error_z500"]
    y_t2m_err = df["error_t2m"]

    X_train, X_test, y_bust_train, y_bust_test, y_z_train, y_z_test, y_t_train, y_t_test = train_test_split(
        X, y_bust, y_z500_err, y_t2m_err, test_size=0.2, random_state=random_seed, stratify=y_bust
    )

    logger.info(f"Training ForecastBustClassifier on {len(X_train)} samples (Bust rate: {y_bust_train.mean():.2%})...")
    classifier = ForecastBustClassifier(
        calibration_method=calibration_method,
        random_state=random_seed,
        n_estimators=120,
    )
    classifier.fit(X_train, y_bust_train, val_split=0.25)

    clf_path = output_dir / "bust_classifier.joblib"
    classifier.save(clf_path)

    logger.info("Training QuantileErrorRegressor for Z500 and T2M error uncertainty intervals...")
    regressor = QuantileErrorRegressor(
        quantiles=(0.10, 0.50, 0.90),
        random_state=random_seed,
        n_estimators=80,
    )
    regressor.fit(X_train, y_z_train, y_t_train)

    reg_path = output_dir / "error_regressor.joblib"
    regressor.save(reg_path)

    logger.info("Model training and calibration completed successfully!")
    logger.info(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BustWatch forecast bust detection models.")
    parser.add_argument("--n-samples", type=int, default=3000, help="Number of training samples")
    parser.add_argument("--calibration", type=str, default="isotonic", choices=["isotonic", "sigmoid"], help="Calibration method")
    parser.add_argument("--output-dir", type=str, default=str(settings.model_dir), help="Output directory for model artifacts")
    args = parser.parse_args()

    train_and_save_models(
        n_samples=args.n_samples,
        output_dir=Path(args.output_dir),
        calibration_method=args.calibration,
    )
