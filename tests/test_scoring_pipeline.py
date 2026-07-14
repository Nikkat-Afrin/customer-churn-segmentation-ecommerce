"""Tests for the purchase-intent scoring pipeline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_pipeline import build_pipeline, load_raw  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load_raw()


@pytest.fixture(scope="module")
def fitted(data):
    X, y = data
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline = build_pipeline(X)
    pipeline.set_params(model__n_estimators=60)   # fast for CI
    pipeline.fit(X_tr, y_tr)
    return pipeline, X_te, y_te


def test_labels_are_binary(data):
    _, y = data
    assert set(y.unique()) == {0, 1}
    assert 0.10 < y.mean() < 0.25   # ~15.5% conversion rate


def test_no_leakage_columns(data):
    X, _ = data
    assert "Revenue" not in X.columns


def test_auc_beats_notebook_baseline(fitted):
    pipeline, X_te, y_te = fitted
    auc = roc_auc_score(y_te, pipeline.predict_proba(X_te)[:, 1])
    assert auc > 0.88, f"AUC {auc:.3f} regressed below expected floor"


def test_pipeline_handles_unseen_category(fitted):
    """OneHotEncoder(handle_unknown='ignore') must not crash on new levels."""
    pipeline, X_te, _ = fitted
    sample = X_te.head(5).copy()
    sample["Month"] = "Undecember"
    proba = pipeline.predict_proba(sample)[:, 1]
    assert len(proba) == 5
    assert np.isfinite(proba).all()


def test_pipeline_is_single_artifact(fitted, tmp_path):
    """The whole pipeline (preprocessing + model) must survive a save/load."""
    import joblib
    pipeline, X_te, y_te = fitted
    path = tmp_path / "pipe.joblib"
    joblib.dump(pipeline, path)
    reloaded = joblib.load(path)
    a = pipeline.predict_proba(X_te.head(20))[:, 1]
    b = reloaded.predict_proba(X_te.head(20))[:, 1]
    assert np.allclose(a, b)
