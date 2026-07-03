"""Run several models on one dataset and compare them side by side.

``compare_models`` fits any mix of promptlearn estimators and plain
scikit-learn / XGBoost estimators on the same train/test split and returns two
tables: per-model metrics, and per-row predictions. promptlearn estimators are
handed the raw DataFrame (they reason over column names and values directly),
while plain estimators are auto-wrapped in a one-hot + scaling pipeline so they
accept the same input.
"""

import logging
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import BasePromptEstimator

logger = logging.getLogger("promptlearn")

_METRIC_COLUMNS = {
    "classification": ["accuracy", "f1_macro"],
    "regression": ["rmse", "r2"],
}


def _infer_task(y) -> str:
    """Guess 'classification' vs 'regression' from the target dtype/cardinality."""
    s = pd.Series(np.asarray(y))
    if s.dtype.kind == "f":
        return "regression"
    if s.dtype.kind in "iu":
        return "classification" if s.nunique() <= 20 else "regression"
    return "classification"  # object / bool / category


def _wrap_for_sklearn(estimator, X: pd.DataFrame) -> Pipeline:
    """Pipe a plain estimator through one-hot + scaling so it accepts the same
    raw DataFrame the promptlearn estimators get."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    steps = []
    if numeric:
        num_pipe = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        steps.append(("num", num_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        steps.append(("cat", cat_pipe, categorical))
    pre = ColumnTransformer(steps) if steps else "passthrough"
    return Pipeline([("pre", pre), ("model", estimator)])


def compare_models(models, X_train, y_train, X_test, y_test, task=None):
    """Fit each model on the same data and compare them.

    Parameters
    ----------
    models : dict[str, estimator]
        Named estimators. promptlearn estimators receive the raw DataFrame;
        plain sklearn/xgboost estimators are auto-wrapped in a one-hot +
        scaling pipeline so they accept the same input.
    X_train, y_train, X_test, y_test : array-like / DataFrame / Series
    task : {"classification", "regression"}, optional
        Inferred from ``y_train`` when omitted.

    Returns
    -------
    metrics : pandas.DataFrame
        One row per model (indexed by name): the task metrics plus
        ``fit_time_sec`` / ``predict_time_sec``. A model that errors out gets
        NaN metrics rather than aborting the whole comparison.
    predictions : pandas.DataFrame
        One row per test instance: ``y_true`` plus one prediction column per model.
    """
    task = task or _infer_task(y_train)
    if task not in _METRIC_COLUMNS:
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")

    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_test = pd.DataFrame(X_test).reset_index(drop=True)
    y_train = pd.Series(np.asarray(y_train))
    y_test = pd.Series(np.asarray(y_test))

    metric_rows = []
    predictions = pd.DataFrame({"y_true": y_test})

    for name, estimator in models.items():
        is_prompt = isinstance(estimator, BasePromptEstimator)
        # promptlearn estimators and pre-built Pipelines (e.g. a
        # PromptFeatureEngineer + classifier) accept the raw DataFrame as-is;
        # only bare sklearn/xgboost estimators get the one-hot wrapper.
        if is_prompt or isinstance(estimator, Pipeline):
            model = estimator
        else:
            model = _wrap_for_sklearn(estimator, X_train)

        row = {"model": name}
        try:
            start = time.time()
            model.fit(X_train, y_train)
            row["fit_time_sec"] = time.time() - start

            start = time.time()
            y_pred = np.asarray(model.predict(X_test))
            row["predict_time_sec"] = time.time() - start

            predictions[name] = y_pred
            if task == "classification":
                row["accuracy"] = accuracy_score(y_test, y_pred)
                row["f1_macro"] = f1_score(y_test, y_pred, average="macro")
            else:
                row["rmse"] = mean_squared_error(y_test, y_pred) ** 0.5
                row["r2"] = r2_score(y_test, y_pred)
        except Exception as e:  # one bad model shouldn't sink the comparison
            logger.warning("Model %r failed during comparison: %s", name, e)
            predictions[name] = np.nan

        metric_rows.append(row)

    columns = _METRIC_COLUMNS[task] + ["fit_time_sec", "predict_time_sec"]
    metrics = pd.DataFrame(metric_rows).set_index("model").reindex(columns=columns)
    return metrics, predictions
