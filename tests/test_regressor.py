import pytest
import pandas as pd
import numpy as np
import joblib
import os
import tempfile

from promptlearn.regressor import PromptRegressor


@pytest.fixture
def sample_Xy_reg():
    X = pd.DataFrame({"x1": [0, 1, 2], "x2": [2, 3, 4]})
    y = pd.Series([1.5, 2.5, 3.5], name="target")
    return X, y


def test_fit_predict_dataframe(sample_Xy_reg):
    X, y = sample_Xy_reg
    reg = PromptRegressor()
    reg.fit(X, y)
    preds = reg.predict(X)
    assert len(preds) == len(y)
    assert all(isinstance(p, float) for p in preds)


def test_fit_predict_ndarray(sample_Xy_reg):
    X, y = sample_Xy_reg
    reg = PromptRegressor()
    reg.fit(X.values, y.values)
    preds = reg.predict(X.values)
    assert len(preds) == len(y)


def test_score_mse(sample_Xy_reg):
    X, y = sample_Xy_reg
    reg = PromptRegressor()
    reg.fit(X, y)
    mse = reg.score(X, y)
    assert isinstance(mse, float)


def test_regressor_predict_without_fit():
    reg = PromptRegressor()
    with pytest.raises(RuntimeError):
        reg.predict(pd.DataFrame({"x": [1]}))


def test_regressor_predict_invalid_type():
    reg = PromptRegressor()
    # Fit with dummy data first
    X = pd.DataFrame({"x": [1.0]})
    y = pd.Series([1.0], name="y")
    reg.fit(X, y)
    # Now test bad input type
    with pytest.raises(ValueError):
        reg.predict("bad input")


def test_joblib_save_load(sample_Xy_reg):
    X, y = sample_Xy_reg
    reg = PromptRegressor()
    reg.fit(X, y)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "reg.joblib")
        joblib.dump(reg, path)
        reg2 = joblib.load(path)
        preds = reg2.predict(X)
        assert len(preds) == len(y)
