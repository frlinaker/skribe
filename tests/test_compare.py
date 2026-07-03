"""Tests for promptlearn.compare_models (offline: sklearn baselines + a dummy
promptlearn-style estimator, no LLM calls)."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from promptlearn.base import BasePromptEstimator
from promptlearn.compare import compare_models, _infer_task


class _DummyPrompt(BasePromptEstimator):
    """A promptlearn estimator that records the columns it was fitted on, so we
    can verify it receives the RAW frame (no one-hot wrapping)."""

    def __init__(self):
        super().__init__(model="dummy", verbose=False, max_train_rows=10)
        self.seen_columns_ = None

    def fit(self, X, y):
        self.seen_columns_ = list(pd.DataFrame(X).columns)
        return self

    def predict(self, X):
        return np.zeros(len(pd.DataFrame(X)), dtype=int)


def _clf_data():
    X = pd.DataFrame({"x1": [0, 1, 2, 3, 4, 5], "x2": [5, 4, 3, 2, 1, 0]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    return X.iloc[:4], y.iloc[:4], X.iloc[4:], y.iloc[4:]


def test_infer_task():
    assert _infer_task(pd.Series([0.1, 0.2, 0.3])) == "regression"
    assert _infer_task(pd.Series([0, 1, 0, 1])) == "classification"
    assert _infer_task(pd.Series(["a", "b", "a"])) == "classification"
    assert _infer_task(pd.Series(range(100))) == "regression"  # many distinct ints


def test_compare_classification_shapes_and_metrics():
    Xtr, ytr, Xte, yte = _clf_data()
    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000),
    }
    metrics, preds = compare_models(models, Xtr, ytr, Xte, yte)

    assert list(metrics.index) == ["dummy", "logreg"]
    assert list(metrics.columns) == [
        "accuracy",
        "f1_macro",
        "fit_time_sec",
        "predict_time_sec",
    ]
    # predictions: y_true + one column per model, one row per test instance
    assert list(preds.columns) == ["y_true", "dummy", "logreg"]
    assert len(preds) == len(yte)
    assert metrics["accuracy"].notna().all()


def test_compare_regression_uses_regression_metrics():
    X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
    y = pd.Series([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2x
    models = {"linreg": LinearRegression(), "dummy": DummyRegressor()}
    metrics, preds = compare_models(
        models, X.iloc[:4], y.iloc[:4], X.iloc[4:], y.iloc[4:]
    )

    assert list(metrics.columns) == ["rmse", "r2", "fit_time_sec", "predict_time_sec"]
    # linear regression should recover y = 2x almost exactly
    assert metrics.loc["linreg", "rmse"] < 1e-6


def test_promptlearn_estimator_gets_raw_columns():
    """A promptlearn estimator must NOT be one-hot wrapped — it should see the
    original categorical column, unchanged."""
    Xtr = pd.DataFrame({"color": ["red", "blue", "red", "green"], "n": [1, 2, 3, 4]})
    ytr = pd.Series([0, 1, 0, 1])
    Xte = pd.DataFrame({"color": ["blue"], "n": [5]})
    yte = pd.Series([1])

    dummy = _DummyPrompt()
    metrics, preds = compare_models({"prompt": dummy}, Xtr, ytr, Xte, yte)

    assert dummy.seen_columns_ == ["color", "n"]  # raw columns, no expansion
    assert "prompt" in preds.columns


def test_failing_model_yields_nan_not_crash():
    Xtr, ytr, Xte, yte = _clf_data()

    class Exploding(DummyClassifier):
        def fit(self, X, y, **kw):
            raise RuntimeError("boom")

    models = {"good": DummyClassifier(strategy="most_frequent"), "bad": Exploding()}
    metrics, preds = compare_models(models, Xtr, ytr, Xte, yte)

    assert metrics.loc["good", "accuracy"] == metrics.loc["good", "accuracy"]  # not NaN
    assert np.isnan(metrics.loc["bad", "accuracy"])
    assert preds["bad"].isna().all()
