import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .base import BaseSkribeEstimator, resolve_model
from .utils import generate_feature_dicts, normalize_feature_name

logger = logging.getLogger("skribe")

DEFAULT_FEATURE_ENGINEERING_PROMPT_TEMPLATE = """
You are doing feature engineering for a tabular machine-learning model.

Write a single valid Python function called 'transform' that, given the feature variables of ONE row (passed as keyword arguments), returns a dict of NEW engineered features derived from the inputs. These new features should make a downstream model more accurate by encoding domain knowledge.

Input columns: {columns}
{target_line}{dataset_stats_section}
Guidelines:
- Derive features using real-world/domain knowledge from semantically meaningful columns (e.g. map a country to its continent or GDP-per-capita tier, parse a date into is_weekend/month, bucket ages, combine related numeric columns into ratios).
- Prefer NUMERIC output features (ints/floats); booleans are fine encoded as 0/1. Avoid free-text outputs.
- Do NOT simply copy the input columns through; only return NEW features.
- Return the SAME set of dict keys for every possible input row. Use a sensible default (e.g. 0) when a value is unknown, missing, or out-of-vocabulary, so the function never raises.
- Coerce inputs with float(x)/int(x) as needed at the top of the function before using them.
- For categorical lookups, include an exhaustive mapping (aim for 100+ keys where relevant: countries, US states, common animals, colors, etc.) with a default fallback.
- IMPORTANT: if the existing features are already highly predictive on their own (e.g. logreg accuracy already near ceiling), or the dataset is too small to support new generalizable features, return an empty dict {{}}.

Every string literal MUST be valid, properly terminated Python. If a key or value contains an apostrophe (e.g. grevy's zebra), wrap that string in double quotes ("grevy's zebra"); if it contains a double quote, wrap it in single quotes. Never leave an unterminated string literal.

The function signature must be: def transform(**features): ...

Only output valid Python code, no markdown or explanations.

Data (sample rows; the last column is the target if one is present):
{data}
"""


class SkribeFeatureEngineer(TransformerMixin, BaseSkribeEstimator):
    """LLM-powered feature engineering as a scikit-learn transformer.

    At ``fit`` the LLM writes a standalone ``transform(**features)`` function
    that derives new, more predictive columns from semantically meaningful ones
    (the same code-generation + validation-retry approach the estimators use, so
    there are no per-row LLM calls). ``transform`` runs that function over each
    row and appends the engineered columns to the input, making it a drop-in
    preprocessing step before any classical model in a ``Pipeline``.
    """

    def __init__(
        self,
        model=None,
        verbose: bool = True,
        max_train_rows: int | None = None,
        max_retries: int = 2,
        web_search: bool = False,
        context_prepass: bool = True,
        vertex_location: str | None = None,
        llm_timeout: float = 120,
    ):
        super().__init__(
            model=resolve_model(model),
            verbose=verbose,
            max_train_rows=max_train_rows,
            max_retries=max_retries,
            web_search=web_search,
            context_prepass=context_prepass,
            vertex_location=vertex_location,
            llm_timeout=llm_timeout,
        )
        self.new_feature_names_ = None

    def fit(self, X, y=None, dataset_stats: dict | None = None) -> "SkribeFeatureEngineer":
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "SkribeFeatureEngineer requires a pandas DataFrame with named columns."
            )
        self.explanation_ = None  # invalidate any cached explanation from a prior fit

        data = X.copy()
        data.columns = [normalize_feature_name(c) for c in data.columns]
        self.feature_names_ = list(data.columns)

        # Include the target column (when supervised) so the LLM can engineer
        # features that are actually relevant to what we are predicting.
        if y is not None:
            self.target_name_ = normalize_feature_name(getattr(y, "name", None) or "target")
            sample_source = data.copy()
            sample_source[self.target_name_] = y.values if hasattr(y, "values") else y
            target_line = (
                f"The downstream model predicts the column '{self.target_name_}'. "
                "Engineer features that help predict it.\n"
            )
        else:
            self.target_name_ = None
            sample_source = data
            target_line = "No target is provided; engineer broadly useful features.\n"

        if self.max_train_rows is not None and len(sample_source) > self.max_train_rows:
            logger.info(
                "Reducing training data from %d to %d rows (max_train_rows).",
                len(sample_source),
                self.max_train_rows,
            )
            sample_df = sample_source.sample(self.max_train_rows, random_state=42)
        else:
            sample_df = sample_source

        if dataset_stats:
            stats_lines = "\n".join(f"  {k}: {v}" for k, v in dataset_stats.items())
            dataset_stats_section = (
                f"\nDataset statistics (use to judge whether FE will help):\n{stats_lines}\n"
            )
        else:
            dataset_stats_section = ""

        base_prompt = DEFAULT_FEATURE_ENGINEERING_PROMPT_TEMPLATE.format(
            data=sample_df.to_csv(index=False),
            columns=", ".join(self.feature_names_),
            target_line=target_line,
            dataset_stats_section=dataset_stats_section,
        )
        logger.info(f"[LLM Prompt]\n{base_prompt}")

        validation_rows = list(
            generate_feature_dicts(data[self.feature_names_], self.feature_names_)
        )

        raw_code, extended_code, fn = self._generate_code(base_prompt, validation_rows)
        self.raw_python_code_ = raw_code
        self.python_code_ = extended_code
        self.predict_fn = fn
        self.new_feature_names_ = self._infer_new_feature_names(validation_rows)
        return self

    def transform(self, X) -> pd.DataFrame:
        if self.predict_fn is None:
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "SkribeFeatureEngineer requires a pandas DataFrame with named columns."
            )

        X_norm = X.copy()
        X_norm.columns = [normalize_feature_name(c) for c in X_norm.columns]

        rows = [
            self._safe_transform_row(feats)
            for feats in generate_feature_dicts(X_norm[self.feature_names_], self.feature_names_)
        ]
        new_df = pd.DataFrame(rows, index=X.index)
        if self.new_feature_names_:
            new_df = new_df.reindex(columns=self.new_feature_names_)
        # Never overwrite an existing input column.
        new_df = new_df[[c for c in new_df.columns if c not in X.columns]]
        return pd.concat([X, new_df], axis=1)

    def get_feature_names_out(self, input_features=None):
        base = (
            list(input_features) if input_features is not None else list(self.feature_names_ or [])
        )
        extra = [c for c in (self.new_feature_names_ or []) if c not in base]
        return np.asarray(base + extra, dtype=object)

    def _validate_predict_fn(self, predict_fn, rows: list, labels: list = []) -> None:
        """Confirm the generated function returns a consistent dict of features."""
        if not rows:
            raise ValueError(
                "SkribeFeatureEngineer needs at least one training row to validate "
                "the generated features."
            )
        keysets = set()
        first = None
        for row in rows[:25]:
            out = predict_fn(**row)
            if not isinstance(out, dict):
                raise ValueError(
                    "transform() must return a dict of new features, but returned "
                    f"{type(out).__name__}."
                )
            keysets.add(tuple(sorted(out.keys())))
            first = out
        if len(keysets) > 1:
            raise ValueError("transform() must return the same set of feature keys for every row.")
        # empty dict is valid — LLM decided no new features are useful (pass-through)

    def _infer_new_feature_names(self, rows: list) -> list:
        for row in rows:
            try:
                out = self.predict_fn(**row)
            except Exception:
                continue
            if isinstance(out, dict) and out:
                return [k for k in out.keys() if k not in self.feature_names_]
        return []

    def _safe_transform_row(self, features: dict) -> dict:
        try:
            out = self.predict_fn(**features)
            if isinstance(out, dict):
                return out
        except Exception as e:
            logger.error(f"[FeatureEngineer ERROR] {e} on features={features}")
        return {k: np.nan for k in (self.new_feature_names_ or [])}


def _make_logreg_pipeline(X: pd.DataFrame) -> Pipeline:
    """Logreg pipeline with imputation and ordinal encoding for mixed-type DataFrames."""
    cat_cols = [
        c
        for c in X.columns
        if X[c].dtype == object
        or str(X[c].dtype) in ("category", "string", "str")
        or pd.api.types.is_string_dtype(X[c])
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="mean"), num_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "enc",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                cat_cols,
            )
        )
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    if not transformers:
        return Pipeline([("clf", clf)])
    return Pipeline([("pre", ColumnTransformer(transformers, remainder="drop")), ("clf", clf)])


def _probe_fe_delta(
    X: pd.DataFrame,
    y,
    fe: "SkribeFeatureEngineer",
    cv: int,
    probe_size: float,
) -> tuple[float, float, int]:
    """Estimate FE lift on a small stratified probe split.

    Fits ``fe`` on a probe subset, measures logreg CV accuracy with and without
    the engineered features, and returns (score_base, score_fe, probe_n_rows).
    """
    n = len(X)
    probe_n = max(int(n * probe_size), 40)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    try:
        train_idx, _ = next(sss.split(X, y))
    except ValueError:
        rng = np.random.default_rng(42)
        train_idx = rng.permutation(n)[: int(n * 0.7)]

    if len(train_idx) > probe_n:
        rng = np.random.default_rng(42)
        train_idx = rng.choice(train_idx, size=probe_n, replace=False)

    X_probe = X.iloc[train_idx].reset_index(drop=True)
    y_probe = y.iloc[train_idx].reset_index(drop=True) if hasattr(y, "iloc") else y[train_idx]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        score_base = float(
            np.mean(
                cross_val_score(
                    _make_logreg_pipeline(X_probe),
                    X_probe,
                    y_probe,
                    cv=cv,
                    scoring="accuracy",
                )
            )
        )

        try:
            X_probe_fe = fe.fit_transform(X_probe, y_probe)
            score_fe = float(
                np.mean(
                    cross_val_score(
                        _make_logreg_pipeline(X_probe_fe),
                        X_probe_fe,
                        y_probe,
                        cv=cv,
                        scoring="accuracy",
                    )
                )
            )
        except Exception as e:
            logger.warning("[AdaptiveFE] probe FE failed: %s", e)
            score_fe = score_base

    return score_base, score_fe, len(train_idx)


class AdaptiveSkribeEngineer(BaseEstimator, TransformerMixin):
    """Feature engineer with automatic pass-through for datasets unlikely to benefit.

    Uses a two-stage decision:

    1. **Size guard** (no LLM): skip if n_rows < ``min_rows``.

    2. **Probe CV** (one LLM call on a small subset): fit SkribeFeatureEngineer
       on a stratified probe split, measure logreg CV accuracy with and without
       the engineered features. If the delta is <= ``min_delta`` (default 0.0),
       discard the probe and return X unchanged at transform time. If positive,
       re-fit FE on the full X_train and use it.

    Attributes after fit:
        probe_score_base_ (float): logreg CV accuracy without FE on probe.
        probe_score_fe_ (float): logreg CV accuracy with FE on probe.
        probe_delta_ (float): probe_score_fe_ - probe_score_base_.
        skip_reason_ (str | None): why FE was skipped, or None if it ran.
        fe_ (SkribeFeatureEngineer | None): fitted FE on full X, or None if skipped.
    """

    def __init__(
        self,
        model=None,
        min_rows: int = 200,
        min_delta: float = 0.0,
        probe_size: float = 0.3,
        cv: int = 3,
        verbose: bool = True,
        max_train_rows: int | None = None,
        max_retries: int = 2,
    ):
        self.model = model
        self.min_rows = min_rows
        self.min_delta = min_delta
        self.probe_size = probe_size
        self.cv = cv
        self.verbose = verbose
        self.max_train_rows = max_train_rows
        self.max_retries = max_retries

    def fit(self, X: pd.DataFrame, y=None) -> "AdaptiveSkribeEngineer":
        if not isinstance(X, pd.DataFrame):
            raise ValueError("AdaptiveSkribeEngineer requires a pandas DataFrame.")

        self.skip_reason_: str | None = None
        self.probe_score_base_: float = float("nan")
        self.probe_score_fe_: float = float("nan")
        self.probe_delta_: float = float("nan")

        # Stage 1: size guard — no LLM call
        if X.shape[0] < self.min_rows:
            self.skip_reason_ = (
                f"n_rows={X.shape[0]} < min_rows={self.min_rows} — "
                "too few samples for engineered features to generalise"
            )
            if self.verbose:
                logger.info("[AdaptiveFE] SKIP — %s", self.skip_reason_)
            self.fe_ = None
            return self

        # Stage 2: probe CV — one LLM call on a small subset
        probe_fe = SkribeFeatureEngineer(
            model=self.model,
            verbose=False,
            max_train_rows=self.max_train_rows,
            max_retries=self.max_retries,
        )
        self.probe_score_base_, self.probe_score_fe_, probe_n = _probe_fe_delta(
            X, y, probe_fe, cv=self.cv, probe_size=self.probe_size
        )
        self.probe_delta_ = self.probe_score_fe_ - self.probe_score_base_

        if self.verbose:
            logger.info(
                "[AdaptiveFE] probe n=%d  base=%.3f  +FE=%.3f  delta=%+.3f",
                probe_n,
                self.probe_score_base_,
                self.probe_score_fe_,
                self.probe_delta_,
            )

        if self.probe_delta_ <= self.min_delta:
            self.skip_reason_ = (
                f"probe_delta={self.probe_delta_:+.3f} <= min_delta={self.min_delta} — "
                "FE did not improve logreg accuracy on probe split"
            )
            if self.verbose:
                logger.info("[AdaptiveFE] SKIP — %s", self.skip_reason_)
            self.fe_ = None
            return self

        # Probe showed lift — fit FE on full X
        if self.verbose:
            logger.info(
                "[AdaptiveFE] RUN — probe delta=%+.3f, fitting FE on full data (n=%d)",
                self.probe_delta_,
                X.shape[0],
            )
        self.fe_ = SkribeFeatureEngineer(
            model=self.model,
            verbose=self.verbose,
            max_train_rows=self.max_train_rows,
            max_retries=self.max_retries,
        )
        self.fe_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "skip_reason_"):
            raise RuntimeError("Call fit() before transform().")
        if self.fe_ is not None:
            return self.fe_.transform(X)
        return X

    def get_feature_names_out(self, input_features=None):
        if self.fe_ is not None:
            return self.fe_.get_feature_names_out(input_features)
        base = list(input_features) if input_features is not None else []
        return np.asarray(base, dtype=object)
