"""Tests for skribe.compare_models and explain_comparison (offline)."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from skribe.base import BaseSkribeEstimator
from skribe.compare import _infer_task, compare_models, explain_comparison
from skribe.explain import Explanation


class _DummySkribe(BaseSkribeEstimator):
    """A skribe estimator that records the columns it was fitted on, so we
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
    metrics, preds = compare_models(models, X.iloc[:4], y.iloc[:4], X.iloc[4:], y.iloc[4:])

    assert list(metrics.columns) == ["rmse", "r2", "fit_time_sec", "predict_time_sec"]
    # linear regression should recover y = 2x almost exactly
    assert metrics.loc["linreg", "rmse"] < 1e-6


def test_skribe_estimator_gets_raw_columns():
    """A skribe estimator must NOT be one-hot wrapped — it should see the
    original categorical column, unchanged."""
    Xtr = pd.DataFrame({"color": ["red", "blue", "red", "green"], "n": [1, 2, 3, 4]})
    ytr = pd.Series([0, 1, 0, 1])
    Xte = pd.DataFrame({"color": ["blue"], "n": [5]})
    yte = pd.Series([1])

    dummy = _DummySkribe()
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


# ---------------------------------------------------------------------------
# explain_comparison tests
# ---------------------------------------------------------------------------


class _FittedSkribe(BaseSkribeEstimator):
    """Pre-fitted skribe stub: always predicts label based on x1 threshold."""

    def __init__(self):
        super().__init__(model="gpt-5.4-mini", verbose=False, max_train_rows=10)
        self.python_code_ = "def predict(**f): return 1 if float(f.get('x1', 0)) > 2 else 0"
        self.feature_names_ = ["x1", "x2"]
        self.target_name_ = "y"
        self.predict_fn = lambda **f: 1 if float(f.get("x1", 0)) > 2 else 0

    def fit(self, X, y):
        return self

    def predict(self, X):
        df = pd.DataFrame(X)
        return np.array([1 if row["x1"] > 2 else 0 for _, row in df.iterrows()])


def _explain_data():
    X = pd.DataFrame({"x1": [0, 1, 2, 3, 4, 5], "x2": [5, 4, 3, 2, 1, 0]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    return X, y


def test_explain_comparison_returns_explanation(monkeypatch):
    """explain_comparison returns an Explanation with the expected keys."""
    X, y = _explain_data()
    prompt_model = _FittedSkribe()
    logreg = LogisticRegression(max_iter=1000).fit(X, y)

    monkeypatch.setattr(
        prompt_model,
        "_call_llm",
        lambda p, **kw: "Model A uses an explicit threshold; Model B uses a linear boundary.",
    )

    result = explain_comparison(
        {"prompt": prompt_model, "logreg": logreg},
        X,
        y,
        task="classification",
        shap_sample=6,
    )

    assert isinstance(result, Explanation)
    assert result.summary
    assert "prompt" in result.data["metrics"].get("accuracy", result.data["metrics"])
    assert isinstance(result.data["disagreement_rate"], float)


def test_explain_comparison_includes_generated_code_in_prompt(monkeypatch):
    """The generated code from skribe estimators must appear in the LLM prompt."""
    X, y = _explain_data()
    prompt_model = _FittedSkribe()
    dummy = DummyClassifier(strategy="most_frequent").fit(X, y)

    captured = {}

    def fake_llm(p, **kw):
        captured["prompt"] = p
        return "narrative"

    monkeypatch.setattr(prompt_model, "_call_llm", fake_llm)

    explain_comparison(
        {"prompt": prompt_model, "dummy": dummy},
        X,
        y,
        task="classification",
        shap_sample=6,
    )

    assert "def predict" in captured["prompt"]
    assert "x1" in captured["prompt"]


def test_explain_comparison_disagreement_rate(monkeypatch):
    """Disagreement rate is 0 when all models agree, >0 when they differ."""
    X, y = _explain_data()
    # Two identical dummy models: no disagreements
    m1 = DummyClassifier(strategy="most_frequent").fit(X, y)
    m2 = DummyClassifier(strategy="most_frequent").fit(X, y)

    # Use a skribe stub so we can monkeypatch _call_llm
    stub = _FittedSkribe()
    monkeypatch.setattr(stub, "_call_llm", lambda p, **kw: "same")

    # Use a skribe stub that also always predicts 0 (same as DummyClassifier most_frequent)
    always_zero = _FittedSkribe()
    always_zero.python_code_ = "def predict(**f): return 0"
    always_zero.predict = lambda X: np.zeros(len(pd.DataFrame(X)), dtype=int)
    monkeypatch.setattr(always_zero, "_call_llm", lambda p, **kw: "same")
    result = explain_comparison(
        {"a": m1, "b": m2, "stub": always_zero}, X, y, task="classification", shap_sample=6
    )
    assert result.data["disagreement_rate"] == 0.0

    # Prompt model disagrees with dummy on some rows
    prompt_model = _FittedSkribe()
    monkeypatch.setattr(prompt_model, "_call_llm", lambda p, **kw: "diff")
    result2 = explain_comparison(
        {"prompt": prompt_model, "dummy": m1}, X, y, task="classification", shap_sample=6
    )
    assert result2.data["disagreement_rate"] > 0.0


def test_explain_comparison_feature_importance_keys(monkeypatch):
    """feature_importance dict has one entry per model that scored successfully."""
    X, y = _explain_data()
    m1 = LogisticRegression(max_iter=1000).fit(X, y)
    m2 = DummyClassifier(strategy="most_frequent").fit(X, y)

    # Patch the BaseSkribeEstimator constructor used internally for the LLM call
    stub = _FittedSkribe()
    monkeypatch.setattr(stub, "_call_llm", lambda p, **kw: "ok")
    # Pass stub as one of the models so its _call_llm gets picked up
    result = explain_comparison(
        {"logreg": m1, "dummy": m2, "stub": stub}, X, y, task="classification", shap_sample=6
    )
    fi = result.data["feature_importance"]
    assert "logreg" in fi
    assert "dummy" in fi
    assert "x1" in fi["logreg"]


def test_explain_comparison_no_shap_fallback(monkeypatch):
    """When shap is unavailable, permutation importance is used and result is still valid."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("no shap")
        return real_import(name, *args, **kwargs)

    X, y = _explain_data()
    m1 = LogisticRegression(max_iter=1000).fit(X, y)
    stub = _FittedSkribe()
    monkeypatch.setattr(stub, "_call_llm", lambda p, **kw: "fallback")

    monkeypatch.setattr(builtins, "__import__", mock_import)

    result = explain_comparison(
        {"logreg": m1, "stub": stub}, X, y, task="classification", shap_sample=6
    )
    assert isinstance(result, Explanation)
    assert result.data["feature_importance"]
