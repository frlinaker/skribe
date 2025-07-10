import pytest
import pandas as pd
import numpy as np
import joblib
import os
import tempfile

from promptlearn.classifier import PromptClassifier


def is_int(val):
    try:
        import numpy as np

        return isinstance(val, (int, np.integer))
    except ImportError:
        return isinstance(val, int)


@pytest.fixture
def sample_Xy():
    X = pd.DataFrame({"x1": [0, 1, 2], "x2": [2, 3, 4]})
    y = pd.Series([0, 1, 0], name="y")
    return X, y


def test_stuff():
    assert is_int(3)
    assert is_int(np.int64(2))


def test_fit_predict_dataframe(sample_Xy):
    X, y = sample_Xy
    clf = PromptClassifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)
    assert all(is_int(p) for p in preds)


def test_fit_predict_ndarray(sample_Xy):
    X, y = sample_Xy
    clf = PromptClassifier()
    clf.fit(X.values, y.values)
    preds = clf.predict(X.values)
    assert len(preds) == len(y)


def test_score_accuracy(sample_Xy):
    X, y = sample_Xy
    clf = PromptClassifier()
    clf.fit(X, y)
    acc = clf.score(X, y)
    assert 0.0 <= acc <= 1.0


def test_zero_row_fit_predict():
    X = pd.DataFrame(columns=["x"])
    y = pd.Series(name="y", dtype=int)
    clf = PromptClassifier()
    clf.fit(X, y)
    preds = clf.predict(pd.DataFrame([{"x": 1}]))
    assert is_int(preds[0])


def test_predict_missing_column(sample_Xy):
    X, y = sample_Xy
    clf = PromptClassifier()
    clf.fit(X, y)
    # Remove one column
    X2 = X.copy().drop("x2", axis=1)
    preds = clf.predict(X2)
    assert len(preds) == len(X2)


def test_predict_extra_column(sample_Xy):
    X, y = sample_Xy
    clf = PromptClassifier()
    clf.fit(X, y)
    X2 = X.copy()
    X2["extra"] = 99
    preds = clf.predict(X2)
    assert len(preds) == len(X2)


def test_predict_without_fit_raises():
    clf = PromptClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(pd.DataFrame({"x": [1, 2]}))


def test_predict_invalid_type_after_fit():
    clf = PromptClassifier()
    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)
    with pytest.raises(ValueError):
        clf.predict("not a dataframe or array")


def test_joblib_save_load(sample_Xy):
    X, y = sample_Xy
    clf = PromptClassifier()
    clf.fit(X, y)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "clf.joblib")
        joblib.dump(clf, path)
        clf2 = joblib.load(path)
        preds = clf2.predict(X)
        assert len(preds) == len(y)


def test_missing_api_key(monkeypatch):
    # Remove env var and force reload
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import importlib
    import promptlearn.base

    importlib.reload(promptlearn.base)
    with pytest.raises(RuntimeError):
        PromptClassifier()
