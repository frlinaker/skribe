import logging
import numpy as np
import pandas as pd

from .base import BasePromptEstimator
from .utils import (
    generate_feature_dicts,
    safe_predict,
)

logger = logging.getLogger("promptlearn")

# Updated LLM prompt template with strong type casting and fallback instructions
DEFAULT_CLASSIFICATION_PROMPT_TEMPLATE = """
Output a single valid Python function called 'predict' that, given the feature variables (see below), predicts the class as an integer (e.g., 0, 1).

Do NOT use any variable not defined below or present in the provided data. If you need external lookups, include them as Python lists or dicts at the top of your output.

All numeric feature values may be provided as strings or numbers. At the top of your function, coerce ALL numeric variables (e.g., weight_kg, lifespan_years, etc.) to float (or int for integer features) using float(x) or int(x) before calculations or comparisons.

Your function must always return an integer class for any input, even if some features are unknown, missing, or out-of-vocabulary. Use a fallback/default prediction (such as 0) if no match is found.

For categorical inputs, include an really exhaustive list of keys (try to get to 100+) in any mapping you make, i.e. names of countries, animals, colors, fruits, etc.

If there is no data given, analyze the names of the input and output columns (assume the last column is the output or target column) and reason to what will be expected as an outcome, and generate code based on that.

Your function must have signature: def predict(**features): ... (or with explicit arguments).

If you use double quotes inside a dictionary key, always use single quotes to surround the key, or escape the inner double quotes.

Only output valid Python code, no markdown or explanations.

Data:
{data}
"""


class PromptClassifier(BasePromptEstimator):
    def __init__(self, model="gpt-4o", verbose: bool = True, max_train_rows: int = 100):
        super().__init__(model=model, verbose=verbose, max_train_rows=max_train_rows)

    def fit(self, X, y) -> "PromptClassifier":
        return super()._fit(X, y, DEFAULT_CLASSIFICATION_PROMPT_TEMPLATE)

    def predict(self, X) -> np.ndarray:
        if self.predict_fn is None:
            raise RuntimeError("Call fit() before predict().")
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            # Use pre-computed self.feature_names_
            results = [
                safe_predict(self.predict_fn, features)
                for features in generate_feature_dicts(X, self.feature_names_)
            ]
            return np.array(results, dtype=int)
        raise ValueError("X must be a DataFrame or ndarray.")

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.array(y)
        # Remove None or unknowns from y_pred for scoring (force 0)
        y_pred = np.array([int(v) if v is not None else 0 for v in y_pred])
        return (y_true == y_pred).mean()
