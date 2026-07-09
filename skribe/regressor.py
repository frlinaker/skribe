import logging
import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score

from .base import BaseSkribeEstimator, resolve_model
from .prompt_markers import DATA_MARKER
from .utils import generate_feature_dicts, safe_regress

logger = logging.getLogger("skribe")

DEFAULT_REGRESSION_PROMPT_TEMPLATE = (
"""
Output a single valid Python function called 'predict' that, given the feature variables (see below), predicts a continuous value (float or int).

Do NOT use any variable not defined below or present in the provided data. If you need external lookups, include them as Python lists or dicts at the top of your output.

All numeric feature values may be provided as strings or numbers. At the top of your function, coerce ALL numeric variables (e.g., weight_kg, area, age, etc.) to float (or int for integer features) using float(x) or int(x) before calculations or comparisons.

Your function must always return a valid float or int prediction for any input, even if some features are unknown, missing, or out-of-vocabulary. Use a fallback/default prediction if no match is found — see the context block below for a representative typical value to use as that default. Do not default to 0.0 unless the context block says 0.0 is a representative value for this target.

For categorical inputs, aim for complete coverage of all plausible real-world values in any mapping you make — not just the values seen in the data sample. Always include a fallback/default for any unlisted keys.

If there is no data given, analyze the names of the input and output columns (assume the last column is the output/target column) and reason what will be expected as an outcome, and generate code based on that.

Your function must have signature: def predict(**features): ... (or with explicit arguments).

Every string literal MUST be valid, properly terminated Python. If a dictionary key or value contains an apostrophe (e.g. grevy's zebra), wrap that string in double quotes ("grevy's zebra"); if it contains a double quote, wrap it in single quotes. Never leave an unterminated string literal.

Only output valid Python code, no markdown or explanations.

"""
+ DATA_MARKER + "\n{data}\n"
)


class SkribeRegressor(RegressorMixin, BaseSkribeEstimator):
    def __init__(
        self,
        model=None,
        verbose: bool = True,
        max_train_rows: int | None = None,
        max_retries: int = 2,
        web_search: bool = False,
        context_prepass: bool = True,
        vertex_location: str | None = None,
    ):
        super().__init__(
            model=resolve_model(model),
            verbose=verbose,
            max_train_rows=max_train_rows,
            max_retries=max_retries,
            web_search=web_search,
            context_prepass=context_prepass,
            vertex_location=vertex_location,
        )

    def fit(
        self, X, y, synthetic_features=None, dataset_description=None
    ) -> "SkribeRegressor":
        # See SkribeClassifier.fit()'s majority_class_ for the analogous
        # classification fix — the prompt's generic "such as 0.0" fallback
        # wording is often a nonsensical value for real regression targets
        # (e.g. an age or price of 0.0). The median is a representative,
        # outlier-robust typical value to fall back to instead.
        y_series = pd.Series(y)
        median = y_series.median() if len(y_series) else 0.0
        self.median_target_ = float(median) if pd.notna(median) else 0.0

        return super()._fit(
            X,
            y,
            DEFAULT_REGRESSION_PROMPT_TEMPLATE,
            synthetic_features=synthetic_features,
            dataset_description=dataset_description,
            majority_class=self.median_target_,
        )

    def predict(self, X) -> np.ndarray:
        if self.predict_fn is None:
            raise RuntimeError("Call fit() before predict().")
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            # Use pre-computed self.feature_names_
            results = [
                safe_regress(self.predict_fn, features)
                for features in generate_feature_dicts(X, self.feature_names_)
            ]
            return np.array(results, dtype=float)
        raise ValueError("X must be a DataFrame or ndarray.")

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.array(y)
        y_pred = np.array([float(v) if v is not None else 0.0 for v in y_pred])
        return r2_score(y_true, y_pred)
