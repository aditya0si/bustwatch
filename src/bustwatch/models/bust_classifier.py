"""Gradient Boosted Decision Tree Classifier for Medium-Range Forecast Busts."""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
from sklearn.ensemble import HistGradientBoostingClassifier

from bustwatch.config import settings
from bustwatch.data.features import FEATURE_COLUMNS, extract_atmospheric_features
from bustwatch.calibration.calibrator import ProbabilityCalibrator

logger = logging.getLogger(__name__)


class ForecastBustClassifier:
    """
    Gradient-boosted model for binary prediction of NWP forecast busts with probability calibration.
    """

    def __init__(
        self,
        calibration_method: str = settings.calibration_method,
        random_state: int = settings.random_seed,
        use_lightgbm: bool = HAS_LIGHTGBM,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
    ):
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.use_lightgbm = use_lightgbm
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.model: Any = None
        self.calibrator = ProbabilityCalibrator(method=calibration_method)
        self.feature_names = list(FEATURE_COLUMNS)
        self.is_fitted = False

        self._init_base_model()

    def _init_base_model(self) -> None:
        """Initialize LightGBM or HistGradientBoosting estimator."""
        if self.use_lightgbm:
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                class_weight="balanced",
                importance_type="gain",
                verbose=-1,
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                class_weight="balanced",
            )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        val_split: float = 0.2,
    ) -> ForecastBustClassifier:
        """
        Fit the base GBDT classifier and calibrate probabilities on validation split.
        """
        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_names].values.astype(np.float32)
        else:
            X_mat = np.asarray(X, dtype=np.float32)

        y_vec = np.asarray(y, dtype=int)

        n_samples = len(X_mat)
        n_val = int(n_samples * val_split)
        if n_val > 10:
            indices = np.arange(n_samples)
            np.random.seed(self.random_state)
            np.random.shuffle(indices)
            train_idx, val_idx = indices[n_val:], indices[:n_val]

            X_train, y_train = X_mat[train_idx], y_vec[train_idx]
            X_val, y_val = X_mat[val_idx], y_vec[val_idx]
        else:
            X_train, y_train = X_mat, y_vec
            X_val, y_val = X_mat, y_vec

        # Train base model
        self.model.fit(X_train, y_train)

        # Predict raw probabilities on validation set
        val_raw_probs = self.model.predict_proba(X_val)[:, 1]

        # Fit calibrator
        self.calibrator.fit(val_raw_probs, y_val)
        self.is_fitted = True
        logger.info(f"Fitted {self.__class__.__name__} on {len(X_train)} samples with {self.calibration_method} calibration.")
        return self

    def predict_proba_raw(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get uncalibrated probabilities from base model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba_raw.")
        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_names].values.astype(np.float32)
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            X_mat = X.reshape(1, -1).astype(np.float32)
        else:
            X_mat = np.asarray(X, dtype=np.float32)

        return self.model.predict_proba(X_mat)[:, 1]

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get calibrated bust probabilities."""
        raw_probs = self.predict_proba_raw(X)
        return self.calibrator.transform(raw_probs)

    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5) -> np.ndarray:
        """Binary prediction using calibrated probability threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return normalized feature importance dictionary."""
        if not self.is_fitted:
            return {f: 1.0 / len(self.feature_names) for f in self.feature_names}

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            importances = np.ones(len(self.feature_names))

        total = sum(importances)
        if total > 0:
            importances = importances / total

        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importances)
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize model and calibrator state to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "calibrator": self.calibrator,
                "feature_names": self.feature_names,
                "calibration_method": self.calibration_method,
                "use_lightgbm": self.use_lightgbm,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> ForecastBustClassifier:
        """Load serialized model artifact."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        data = joblib.load(path)
        instance = cls(
            calibration_method=data.get("calibration_method", "isotonic"),
            use_lightgbm=data.get("use_lightgbm", HAS_LIGHTGBM),
        )
        instance.model = data["model"]
        instance.calibrator = data["calibrator"]
        instance.feature_names = data["feature_names"]
        instance.is_fitted = data.get("is_fitted", True)
        return instance
