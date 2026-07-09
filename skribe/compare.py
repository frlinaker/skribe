"""Run several models on one dataset and compare them side by side.

``compare_models`` fits any mix of skribe estimators and plain
scikit-learn / XGBoost estimators on the same train/test split and returns two
tables: per-model metrics, and per-row predictions. skribe estimators are
handed the raw DataFrame (they reason over column names and values directly),
while plain estimators are auto-wrapped in a one-hot + scaling pipeline so they
accept the same input.

``explain_comparison`` takes the same set of *already-fitted* models plus test
data and produces a contrastive narrative: SHAP-based feature importance per
model, disagreement analysis, and an LLM-generated plain-English summary of
why the models differ.  SHAP is an optional dependency; if it is not installed
the function falls back to permutation importance.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import BaseSkribeEstimator
from .explain import Explanation

logger = logging.getLogger("skribe")

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
    raw DataFrame the skribe estimators get."""
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
        Named estimators. skribe estimators receive the raw DataFrame;
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
        is_prompt = isinstance(estimator, BaseSkribeEstimator)
        # skribe estimators and pre-built Pipelines (e.g. a
        # SkribeFeatureEngineer + classifier) accept the raw DataFrame as-is;
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


# ---------------------------------------------------------------------------
# explain_comparison
# ---------------------------------------------------------------------------


def _shap_importance(model, X_sample: pd.DataFrame) -> Optional[pd.Series]:
    """Return mean |SHAP| per feature, or None if shap is not installed."""
    try:
        import shap
    except ImportError:
        return None

    # Encode X_sample to a numeric array for KernelExplainer
    numeric = X_sample.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X_sample.columns if c not in numeric]

    if cat:
        from sklearn.preprocessing import OrdinalEncoder

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_arr = enc.fit_transform(X_sample[cat])
        num_arr = (
            X_sample[numeric].to_numpy(dtype=float) if numeric else np.empty((len(X_sample), 0))
        )
        X_arr = np.hstack([num_arr, cat_arr])
        col_names = numeric + cat
    else:
        X_arr = X_sample.to_numpy(dtype=float)
        col_names = list(X_sample.columns)

    background = shap.kmeans(X_arr, min(10, len(X_arr)))

    def predict_fn(arr):
        df = pd.DataFrame(arr, columns=col_names)
        if cat:
            # Re-map categorical columns back to original strings via inverse_transform.
            # Must pass the full cat sub-frame at once so column count matches the encoder.
            cat_sub = df[cat].clip(-1).astype(int)
            for i, c in enumerate(cat):
                cat_sub[c] = cat_sub[c].clip(-1, len(enc.categories_[i]) - 1)
            decoded = enc.inverse_transform(cat_sub)
            for j, c in enumerate(cat):
                df[c] = decoded[:, j]
        return np.asarray(model.predict(df), dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_vals = explainer.shap_values(X_arr, nsamples=100, silent=True)

    if isinstance(shap_vals, list):
        # Multi-class: average across classes
        arr = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
    else:
        arr = np.abs(shap_vals)

    return pd.Series(np.mean(arr, axis=0), index=col_names)


def _permutation_importance(model, X_sample: pd.DataFrame, y_sample: pd.Series) -> pd.Series:
    """Fallback when shap is unavailable: mean accuracy drop per feature."""
    baseline = accuracy_score(y_sample, model.predict(X_sample))
    scores = {}
    for col in X_sample.columns:
        shuffled = X_sample.copy()
        shuffled[col] = X_sample[col].sample(frac=1, random_state=0).values
        try:
            scores[col] = baseline - accuracy_score(y_sample, model.predict(shuffled))
        except Exception:
            scores[col] = 0.0
    return pd.Series(scores)


def _build_comparison_prompt(
    model_names: list[str],
    metrics: pd.DataFrame,
    importance_df: Optional[pd.DataFrame],
    disagreement_summary: str,
    codes: dict[str, str],
    task: str,
    dataset_description: str,
) -> str:
    lines = [
        "You are an expert ML analyst. Several models were evaluated on the same dataset. "
        "Your task is to write a concise, insightful plain-English comparison explaining "
        "WHY the models differ in performance — not just THAT they differ.",
        "",
    ]

    if dataset_description:
        lines += [f"Dataset: {dataset_description}", ""]

    lines += ["## Performance metrics", metrics.to_string(), ""]

    if importance_df is not None:
        lines += [
            "## Feature importance per model (mean |SHAP| or permutation drop, higher = more influential)",
            importance_df.round(4).to_string(),
            "",
        ]

    if disagreement_summary:
        lines += ["## Where models disagree", disagreement_summary, ""]

    for name, code in codes.items():
        lines += [f"## Generated prediction code for '{name}'", "```python", code, "```", ""]

    lines += [
        "Write a comparison covering:",
        "1. Which model performs best and by how much.",
        "2. Which features each model relies on most — and whether they agree.",
        "3. For any skribe model whose code is shown: what explicit rules or "
        "domain knowledge the code encodes that statistical models cannot capture.",
        "4. A hypothesis for why the best model outperforms the others on this dataset.",
        "Be specific and faithful to the numbers and code above. "
        "Do not speculate beyond what the data shows. Plain English, no markdown headers.",
    ]

    return "\n".join(lines)


def explain_comparison(
    models: dict,
    X_test,
    y_test,
    metrics: Optional[pd.DataFrame] = None,
    task: Optional[str] = None,
    shap_sample: int = 50,
    dataset_description: str = "",
    llm_model: Optional[str] = None,
) -> Explanation:
    """Produce a contrastive explanation of why fitted models differ.

    Parameters
    ----------
    models : dict[str, estimator]
        Already-fitted estimators (same interface as ``compare_models``).
    X_test, y_test : array-like / DataFrame / Series
        Held-out evaluation data.
    metrics : pd.DataFrame, optional
        Pre-computed metrics table from ``compare_models``.  If omitted,
        accuracy/RMSE is computed here.
    task : {"classification", "regression"}, optional
        Inferred from ``y_test`` when omitted.
    shap_sample : int
        Rows to use for SHAP KernelExplainer (keep ≤100 for speed).
    dataset_description : str
        Free-text context passed to the LLM narrator.
    llm_model : str, optional
        LiteLLM model ID for the narrative call.  Defaults to the model of
        the first skribe estimator found, then ``DEFAULT_MODEL``.

    Returns
    -------
    Explanation
        ``data`` keys: ``summary`` (str), ``metrics`` (dict), ``feature_importance``
        (dict of {model: {feature: score}}), ``disagreement_rate`` (float).
    """
    from .base import DEFAULT_MODEL

    X_test = pd.DataFrame(X_test).reset_index(drop=True)
    y_test = pd.Series(np.asarray(y_test)).reset_index(drop=True)
    task = task or _infer_task(y_test)

    # --- resolve LLM caller ---
    llm_caller: Optional[BaseSkribeEstimator] = None
    for est in models.values():
        if isinstance(est, BaseSkribeEstimator):
            llm_caller = est
            break
    if llm_caller is None:
        # Synthesize a minimal caller just for _call_llm
        llm_caller = BaseSkribeEstimator(
            model=llm_model or DEFAULT_MODEL, verbose=False, max_train_rows=None
        )

    if llm_model:
        llm_caller.model = llm_model

    # --- metrics (recompute if not supplied) ---
    if metrics is None:
        rows = []
        for name, est in models.items():
            try:
                y_pred = np.asarray(est.predict(X_test))
                if task == "classification":
                    rows.append(
                        {
                            "model": name,
                            "accuracy": accuracy_score(y_test, y_pred),
                            "f1_macro": f1_score(y_test, y_pred, average="macro"),
                        }
                    )
                else:
                    rows.append(
                        {
                            "model": name,
                            "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
                            "r2": r2_score(y_test, y_pred),
                        }
                    )
            except Exception as e:
                logger.warning("Could not score model %r: %s", name, e)
        metrics = pd.DataFrame(rows).set_index("model")

    # --- SHAP / permutation importance ---
    sample_idx = np.random.default_rng(42).choice(
        len(X_test), min(shap_sample, len(X_test)), replace=False
    )
    X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
    y_sample = y_test.iloc[sample_idx].reset_index(drop=True)

    importance_rows = {}
    use_shap = True
    try:
        import shap as _shap_check  # noqa: F401
    except ImportError:
        use_shap = False
        logger.info("shap not installed — falling back to permutation importance")

    for name, est in models.items():
        try:
            if use_shap:
                imp = _shap_importance(est, X_sample)
            else:
                imp = _permutation_importance(est, X_sample, y_sample)
            if imp is not None:
                importance_rows[name] = imp
        except Exception as e:
            logger.warning("Importance failed for %r: %s", name, e)

    importance_df = pd.DataFrame(importance_rows).T if importance_rows else None

    # --- disagreement analysis ---
    preds = {}
    for name, est in models.items():
        try:
            preds[name] = np.asarray(est.predict(X_test))
        except Exception:
            pass

    disagreement_summary = ""
    disagreement_rate = None
    if len(preds) >= 2:
        pred_df = pd.DataFrame(preds)
        disagree_mask = pred_df.nunique(axis=1) > 1
        disagreement_rate = float(disagree_mask.mean())
        disagree_rows = X_test[disagree_mask].copy()
        disagree_rows["y_true"] = y_test[disagree_mask].values
        for name, arr in preds.items():
            disagree_rows[f"pred_{name}"] = arr[disagree_mask]

        lines = [
            f"{disagreement_rate:.1%} of test rows ({disagree_mask.sum()}/{len(X_test)}) have at least one model disagreement."
        ]
        if len(disagree_rows) > 0:
            # Show which model is right on disagreements
            for name in preds:
                correct = (disagree_rows[f"pred_{name}"] == disagree_rows["y_true"]).mean()
                lines.append(
                    f"  On disagreement rows: '{name}' is correct {correct:.1%} of the time."
                )
            # Show a few example rows
            sample = disagree_rows.head(5)
            lines.append(f"\nExample disagreement rows (up to 5):\n{sample.to_string()}")
        disagreement_summary = "\n".join(lines)

    # --- collect generated code from skribe estimators ---
    codes = {}
    for name, est in models.items():
        code = getattr(est, "python_code_", None)
        if code:
            codes[name] = code

    # --- build prompt and call LLM ---
    prompt = _build_comparison_prompt(
        model_names=list(models.keys()),
        metrics=metrics,
        importance_df=importance_df,
        disagreement_summary=disagreement_summary,
        codes=codes,
        task=task,
        dataset_description=dataset_description,
    )

    logger.info("explain_comparison: calling LLM for narrative…")
    summary = llm_caller._call_llm(prompt)

    return Explanation(
        meta={
            "name": "explain_comparison",
            "type": ["contrastive"],
            "explanations": ["global"],
            "models": list(models.keys()),
            "shap_used": use_shap and bool(importance_rows),
        },
        data={
            "summary": summary.strip(),
            "metrics": metrics.to_dict(),
            "feature_importance": {k: v.to_dict() for k, v in importance_rows.items()},
            "disagreement_rate": disagreement_rate,
            "disagreement_summary": disagreement_summary,
        },
    )
