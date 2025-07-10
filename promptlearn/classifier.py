import logging
import pandas as pd
import numpy as np
from typing import Optional, Callable

from .base import BasePromptEstimator, normalize_feature_name

logger = logging.getLogger("promptlearn")

# Updated LLM prompt template with strong type casting and fallback instructions
DEFAULT_PROMPT_TEMPLATE = """
Output a single valid Python function called 'predict' that, given the feature variables (see below), predicts the class as an integer (e.g., 0, 1).

Do NOT use any variable not defined below or present in the provided data. If you need external lookups, include them as Python lists or dicts at the top of your output.

All numeric feature values may be provided as strings or numbers. At the top of your function, coerce ALL numeric variables (e.g., weight_kg, lifespan_years, etc.) to float (or int for integer features) using float(x) or int(x) before calculations or comparisons.

Your function must always return an integer class for any input, even if some features are unknown, missing, or out-of-vocabulary. Use a fallback/default prediction (such as 0) if no match is found.

For categorical inputs, include an really exhaustive list of keys (try to get to 100+) in any mapping you make, i.e. names of countries, animals, colors, fruits, etc.

If there is no data given, analyze the names of the input and output columns (assume the last column is the output or target column) and reason to what will be expected as an outcome, and generate code based on that.

Your function must have signature: def predict(**features): ... (or with explicit arguments).

Only output valid Python code, no markdown or explanations.

Data:
{data}
"""

class PromptClassifier(BasePromptEstimator):
    def __init__(self, model="gpt-4o", verbose: bool = True, max_train_rows: int = 100):
        super().__init__(model=model, verbose=verbose)
        self.max_train_rows = max_train_rows
        self.python_code_: Optional[str] = None
        self.predict_fn: Optional[Callable] = None
        self.target_name_: Optional[str] = None
        self.feature_names_: Optional[list] = None

    def fit(self, X, y):
        # Handle DataFrame or array
        if isinstance(X, pd.DataFrame):
            data = X.copy()
            self.feature_names_ = [normalize_feature_name(col) for col in data.columns]
            self.target_name_ = normalize_feature_name(y.name if hasattr(y, "name") else "target")
            data[self.target_name_] = y
            # Normalize column names for the LLM prompt
            data.columns = [normalize_feature_name(col) for col in data.columns]
        elif isinstance(X, np.ndarray):
            data = pd.DataFrame(X)
            self.feature_names_ = [f"col{i}" for i in range(X.shape[1])]
            self.target_name_ = "target"
            data[self.target_name_] = y
            data.columns = self.feature_names_ + [self.target_name_]
        else:
            raise ValueError("X must be a pandas DataFrame or numpy array.")

        # Use a small sample for LLM to avoid expensive calls
        sample_df = data.head(self.max_train_rows)
        csv_data = sample_df.to_csv(index=False)

        prompt = DEFAULT_PROMPT_TEMPLATE.format(data=csv_data)
        logger.info(f"[LLM Prompt]\n{prompt}")

        # Call LLM and get code
        code = self._call_llm(prompt)
        if not isinstance(code, str):
            code = str(code)
        logger.info(f"[LLM Output]\n{code}")

        # Remove markdown/code block if present (triple backticks)
        code = self._extract_python_code(code)
        if not code.strip():
            logger.error("LLM output is empty after removing markdown/code block.")
            raise ValueError("No code to exec from LLM output.")

        self.python_code_ = code
        print(f"the cleaned up code is: [START]{code}[END]")

        # Compile the code into a function
        self.predict_fn = self._make_predict_fn(code)

        return self

    def predict(self, X) -> np.ndarray:
        if self.predict_fn is None:
            raise RuntimeError("Call fit() before predict().")

        def normalize_dict_keys(d):
            return {normalize_feature_name(k): v for k, v in d.items()}

        results = []
        if isinstance(X, pd.DataFrame):
            for idx, row in X.iterrows():
                features = normalize_dict_keys(row.to_dict())
                res = self._safe_predict(self.predict_fn, features)
                results.append(res)
        elif isinstance(X, np.ndarray):
            cols = self._feature_names_for_array(X)
            cols = [normalize_feature_name(c) for c in cols]
            for arr in X:
                features = dict(zip(cols, arr))
                features = normalize_dict_keys(features)
                res = self._safe_predict(self.predict_fn, features)
                results.append(res)
        else:
            raise ValueError("X must be a DataFrame or ndarray.")
        return np.array(results, dtype=int)

    def _feature_names_for_array(self, X: np.ndarray):
        # Try to recover column names from training (not ideal, but best-effort)
        if hasattr(self, "feature_names_") and self.feature_names_ is not None:
            return self.feature_names_
        # fallback: col0, col1, ..., colN
        return [f"col{i}" for i in range(X.shape[1])]

    @staticmethod
    def _extract_python_code(text: str) -> str:
        # Remove code fences and cut at any obvious example markers
        if "```python" in text:
            text = text.split("```python", 1)[-1]
        if "```" in text:
            text = text.split("```", 1)[0]
        return text

    def _make_predict_fn(self, code: str):
        # Use a shared dictionary for globals/locals
        local_vars = {}
        try:
            exec(code, local_vars, local_vars)
        except Exception as e:
            raise ValueError(f"Could not exec LLM code: {e}\nCode was:\n{code}")
        # Look for 'predict' function
        fn = local_vars.get("predict", None)
        if not callable(fn):
            raise ValueError("No valid function named 'predict' or any callable found in LLM output.")
        return fn

    def _safe_predict(self, fn: Callable, features: dict) -> int:
        # Try to cast all numbers (as string) to float or int
        clean = {}
        for k, v in features.items():
            if v is None:
                clean[k] = v
                continue
            if isinstance(v, (float, int)):
                clean[k] = v
            elif isinstance(v, str):
                try:
                    if "." in v:
                        f = float(v)
                        # Try to coerce to int if appropriate
                        if f.is_integer():
                            clean[k] = int(f)
                        else:
                            clean[k] = f
                    else:
                        clean[k] = int(v)
                except Exception:
                    clean[k] = v
            else:
                clean[k] = v
        try:
            res = fn(**clean)
            # Always coerce output to int (default/fallback 0)
            return int(res) if res is not None else 0
        except Exception as e:
            logger.error(f"[PredictFn ERROR] {e} on features={features}")
            return 0

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.array(y)
        # Remove None or unknowns from y_pred for scoring (force 0)
        y_pred = np.array([int(v) if v is not None else 0 for v in y_pred])
        return (y_true == y_pred).mean()
