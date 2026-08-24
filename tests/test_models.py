"""Unit tests for ML models (ForecastBustClassifier and QuantileErrorRegressor)."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from bustwatch.data.synthetic import generate_synthetic_dataset
from bustwatch.data.features import FEATURE_COLUMNS, compute_bust_labels
from bustwatch.models.bust_classifier import ForecastBustClassifier
from bustwatch.models.error_regressor import QuantileErrorRegressor


@pytest.fixture
def sample_dataset():
    df = generate_synthetic_dataset(n_samples=150, random_seed=123)
    return compute_bust_labels(df)


def test_forecast_bust_classifier_lifecycle(tmp_path, sample_dataset):
    """Test full classifier training, probability calibration, prediction, and serialization."""
    X = sample_dataset[FEATURE_COLUMNS]
    y = sample_dataset["is_bust"]

    clf = ForecastBustClassifier(calibration_method="isotonic", n_estimators=30, random_state=42)
    assert not clf.is_fitted

    clf.fit(X, y, val_split=0.2)
    assert clf.is_fitted

    # Raw probabilities
    raw_probs = clf.predict_proba_raw(X)
    assert len(raw_probs) == len(X)
    assert (raw_probs >= 0.0).all() and (raw_probs <= 1.0).all()

    # Calibrated probabilities
    cal_probs = clf.predict_proba(X)
    assert len(cal_probs) == len(X)
    assert (cal_probs >= 0.0).all() and (cal_probs <= 1.0).all()

    # Binary predictions
    preds = clf.predict(X, threshold=0.5)
    assert set(np.unique(preds)).issubset({0, 1})

    # Feature importances
    importances = clf.get_feature_importances()
    assert len(importances) == len(FEATURE_COLUMNS)
    assert np.isclose(sum(importances.values()), 1.0, atol=1e-3)

    # Save and reload
    save_path = tmp_path / "test_clf.joblib"
    clf.save(save_path)
    assert save_path.exists()

    loaded_clf = ForecastBustClassifier.load(save_path)
    assert loaded_clf.is_fitted
    loaded_probs = loaded_clf.predict_proba(X)
    np.testing.assert_allclose(cal_probs, loaded_probs, atol=1e-4)


def test_quantile_error_regressor_lifecycle(tmp_path, sample_dataset):
    """Test quantile error regressor fitting, multi-quantile prediction, and serialization."""
    X = sample_dataset[FEATURE_COLUMNS]
    y_z = sample_dataset["error_z500"]
    y_t = sample_dataset["error_t2m"]

    reg = QuantileErrorRegressor(quantiles=(0.10, 0.50, 0.90), n_estimators=25, random_state=42)
    assert not reg.is_fitted

    reg.fit(X, y_z, y_t)
    assert reg.is_fitted

    # Predict quantiles
    res = reg.predict_error_quantiles(X)
    assert "z500_error_m" in res
    assert "t2m_error_c" in res

    q10_z = res["z500_error_m"][0.10]
    q50_z = res["z500_error_m"][0.50]
    q90_z = res["z500_error_m"][0.90]

    assert len(q50_z) == len(X)
    # Non-negative error bounds
    assert (q10_z >= 0.0).all()
    assert (q50_z >= 0.0).all()
    assert (q90_z >= 0.0).all()

    # Save and reload
    save_path = tmp_path / "test_reg.joblib"
    reg.save(save_path)
    assert save_path.exists()

    loaded_reg = QuantileErrorRegressor.load(save_path)
    assert loaded_reg.is_fitted
    loaded_res = loaded_reg.predict_error_quantiles(X)
    np.testing.assert_allclose(q50_z, loaded_res["z500_error_m"][0.50], atol=1e-4)
