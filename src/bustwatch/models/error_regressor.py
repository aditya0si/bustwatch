"""Quantile Regression Model for NWP Forecast Error Bounds."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
from sklearn.ensemble import HistGradientBoostingRegressor

from bustwatch.config import settings
from bustwatch.data.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class QuantileErrorRegressor:
    """
    Multi-quantile gradient-boosted regression for estimating forecast error intervals (10th, 50th, 90th percentiles).
    """

    def __init__(
        self,
        quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90),
        random_state: int = settings.random_seed,
        use_lightgbm: bool = HAS_LIGHTGBM,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
    ):
        self.quantiles = quantiles
        self.random_state = random_state
        self.use_lightgbm = use_lightgbm
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models_z500: dict[float, Any] = {}
        self.models_t2m: dict[float, Any] = {}
        self.feature_names = list(FEATURE_COLUMNS)
        self.is_fitted = False

        self._init_models()

    def _init_models(self) -> None:
        """Initialize quantile regression models for each quantile level."""
        for q in self.quantiles:
            if self.use_lightgbm:
                self.models_z500[q] = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    verbose=-1,
                )
                self.models_t2m[q] = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    verbose=-1,
                )
            else:
                self.models_z500[q] = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q,
                    max_iter=self.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                )
                self.models_t2m[q] = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q,
                    max_iter=self.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                )

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y_z500_error: np.ndarray | pd.Series,
        y_t2m_error: np.ndarray | pd.Series,
    ) -> QuantileErrorRegressor:
        """Fit quantile regressors for Z500 error and T2M error."""
        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_names].values.astype(np.float32)
        else:
            X_mat = np.asarray(X, dtype=np.float32)

        y_z = np.asarray(y_z500_error, dtype=np.float32)
        y_t = np.asarray(y_t2m_error, dtype=np.float32)

        for q in self.quantiles:
            self.models_z500[q].fit(X_mat, y_z)
            self.models_t2m[q].fit(X_mat, y_t)

        self.is_fitted = True
        logger.info(f"Fitted QuantileErrorRegressor for quantiles {self.quantiles}")
        return self

    def predict_error_quantiles(
        self, X: np.ndarray | pd.DataFrame
    ) -> dict[str, dict[float, np.ndarray]]:
        """Predict error quantiles for Z500 and T2M."""
        if not self.is_fitted:
            raise RuntimeError("Regressor must be fitted before predict_error_quantiles.")

        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_names].values.astype(np.float32)
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            X_mat = X.reshape(1, -1).astype(np.float32)
        else:
            X_mat = np.asarray(X, dtype=np.float32)

        res_z500 = {q: np.maximum(0.0, self.models_z500[q].predict(X_mat)) for q in self.quantiles}
        res_t2m = {q: np.maximum(0.0, self.models_t2m[q].predict(X_mat)) for q in self.quantiles}

        return {"z500_error_m": res_z500, "t2m_error_c": res_t2m}

    def save(self, filepath: str | Path) -> None:
        """Serialize regressor state to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models_z500": self.models_z500,
                "models_t2m": self.models_t2m,
                "quantiles": self.quantiles,
                "feature_names": self.feature_names,
                "use_lightgbm": self.use_lightgbm,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.info(f"QuantileErrorRegressor saved to {path}")

    @classmethod
    def load(cls, filepath: str | Path) -> QuantileErrorRegressor:
        """Load serialized regressor artifact."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Regressor file not found at {path}")

        data = joblib.load(path)
        instance = cls(
            quantiles=data.get("quantiles", (0.10, 0.50, 0.90)),
            use_lightgbm=data.get("use_lightgbm", HAS_LIGHTGBM),
        )
        instance.models_z500 = data["models_z500"]
        instance.models_t2m = data["models_t2m"]
        instance.feature_names = data["feature_names"]
        instance.is_fitted = data.get("is_fitted", True)
        return instance
