import logging
import numpy as np

from sklearn.metrics import r2_score

from .base import BasePromptEstimator
from .utils import generate_feature_dicts, safe_regress

logger = logging.getLogger("promptlearn")

DEFAULT_REGRESSION_PROMPT_TEMPLATE = """
Output a single valid Python function called 'predict' that, given the feature variables (see below), predicts a continuous value (float or int).

Do NOT use any variable not defined below or present in the provided data. If you need external lookups, include them as Python lists or dicts at the top of your output.

All numeric feature values may be provided as strings or numbers. At the top of your function, coerce ALL numeric variables (e.g., weight_kg, area, age, etc.) to float (or int for integer features) using float(x) or int(x) before calculations or comparisons.

Your function must always return a valid float or int prediction for any input, even if some features are unknown, missing, or out-of-vocabulary. Use a fallback/default prediction (such as 0.0) if no match is found.

For categorical inputs, include an exhaustive mapping if possible (e.g., known country names, brands, colors), but ALWAYS include a fallback/default for unlisted keys.

If there is no data given, analyze the names of the input and output columns (assume the last column is the output/target column) and reason what will be expected as an outcome, and generate code based on that.

Your function must have signature: def predict(**features): ... (or with explicit arguments).

Only output valid Python code, no markdown or explanations.

Data:
{data}
"""


class PromptRegressor(BasePromptEstimator):
    def __init__(self, model="gpt-4o", verbose: bool = True, max_train_rows: int = 100):
        super().__init__(model=model, verbose=verbose, max_train_rows=max_train_rows)

    def fit(self, X, y) -> "PromptRegressor":
        return super()._fit(X, y, DEFAULT_REGRESSION_PROMPT_TEMPLATE)

    def predict(self, X) -> np.ndarray:
        if self.predict_fn is None:
            raise RuntimeError("Call fit() before predict().")
        # Use pre-computed self.feature_names_
        results = [
            safe_regress(self.predict_fn, features)
            for features in generate_feature_dicts(X, self.feature_names_)
        ]
        return np.array(results, dtype=float)

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.array(y)
        y_pred = np.array([float(v) if v is not None else 0.0 for v in y_pred])
        return r2_score(y_true, y_pred)
