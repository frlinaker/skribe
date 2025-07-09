import logging
import pandas as pd
import numpy as np
from typing import Optional, Callable
import re

from .base import BasePromptEstimator

logger = logging.getLogger("promptlearn")

def normalize_feature_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    name = re.sub(r"__+", "_", name)
    return name.strip("_").lower()

DEFAULT_PROMPT_TEMPLATE = """
Output a single valid Python function called 'regress' that, given the feature variables (see below), predicts a continuous value (float or int).

Do NOT use any variable not defined below or present in the provided data. If you need external lookups, include them as Python lists or dicts at the top of your output.

All numeric feature values may be provided as strings or numbers. At the top of your function, coerce ALL numeric variables (e.g., weight_kg, area, age, etc.) to float (or int for integer features) using float(x) or int(x) before calculations or comparisons.

Your function must always return a valid float or int prediction for any input, even if some features are unknown, missing, or out-of-vocabulary. Use a fallback/default prediction (such as 0.0) if no match is found.

For categorical inputs, include an exhaustive mapping if possible (e.g., known country names, brands, colors), but ALWAYS include a fallback/default for unlisted keys.

If there is no data given, analyze the names of the input and output columns (assume the last column is the output/target column) and reason what will be expected as an outcome, and generate code based on that.

Your function must have signature: def regress(**features): ... (or with explicit arguments).

Only output valid Python code, no markdown or explanations.

Data:
{data}
"""

class PromptRegressor(BasePromptEstimator):
    def __init__(
        self,
        model: str = "gpt-4o",
        verbose: bool = True,
        max_train_rows: int = 10,
    ):
        super().__init__(model=model, verbose=verbose)
        self.max_train_rows = max_train_rows
        self.heuristic_: Optional[str] = None
        self.regress_fn: Optional[Callable] = None
        self.target_name_: Optional[str] = None
        self.feature_names_: Optional[list] = None

    def get_params(self, deep=True):
        # Only include arguments that are accepted by __init__
        return {
            "model": self.model,
            "verbose": self.verbose,
            "max_train_rows": self.max_train_rows
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-pickleable objects
        if "regress_fn" in state:
            del state["regress_fn"]
        if "llm_client" in state:
            del state["llm_client"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recompile code after loading
        if getattr(self, "heuristic_", None):
            try:
                self.regress_fn = self._make_regress_fn(self.heuristic_)
            except Exception as e:
                logger.warning(f"Failed to recompile regression function: {e}")
                self.regress_fn = None
        # Re-initialize LLM client if needed
        self._reconnect_llm_client()

    def _reconnect_llm_client(self):
        # Only re-import and reconnect if you actually need the client after loading
        try:
            import openai
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm_client = openai.OpenAI(api_key=api_key)
            else:
                self.llm_client = None
        except Exception as e:
            self.llm_client = None
            logger.warning(f"Could not reconnect LLM client: {e}")

    def fit(self, X, y):
        # Accept DataFrame or ndarray, and always normalize feature names
        if isinstance(X, pd.DataFrame):
            data = X.copy()
            self.feature_names_ = [normalize_feature_name(col) for col in data.columns]
            self.target_name_ = normalize_feature_name(y.name if hasattr(y, "name") else "target")
            data[self.target_name_] = y
            data.columns = self.feature_names_ + [self.target_name_]
        elif isinstance(X, np.ndarray):
            self.feature_names_ = [f"col{i}" for i in range(X.shape[1])]
            self.target_name_ = "target"
            data = pd.DataFrame(X, columns=self.feature_names_)
            data[self.target_name_] = y
        else:
            raise ValueError("X must be a pandas DataFrame or numpy array.")

        # Use a small sample for LLM prompt
        sample_df = data.head(self.max_train_rows)
        csv_data = sample_df.to_csv(index=False)

        prompt = DEFAULT_PROMPT_TEMPLATE.format(data=csv_data)
        logger.info(f"[LLM Prompt]\n{prompt}")

        code = self._call_llm(prompt)
        if not isinstance(code, str):
            code = str(code)
        logger.info(f"[LLM Output]\n{code}")

        code = self._extract_python_code(code)
        if not code.strip():
            logger.error("LLM output is empty after removing markdown/code block.")
            raise ValueError("No code to exec from LLM output.")

        self.heuristic_ = code
        print(f"the cleaned up code is: [START]{code}[END]")

        self.regress_fn = self._make_regress_fn(code)
        return self

    def predict(self, X) -> np.ndarray:
        if self.regress_fn is None:
            raise RuntimeError("Call fit() before predict().")

        def normalize_dict_keys(d):
            return {normalize_feature_name(k): v for k, v in d.items()}

        results = []
        if isinstance(X, pd.DataFrame):
            for idx, row in X.iterrows():
                features = normalize_dict_keys(row.to_dict())
                res = self._safe_regress(self.regress_fn, features)
                results.append(res)
        elif isinstance(X, np.ndarray):
            cols = self._feature_names_for_array(X)
            cols = [normalize_feature_name(c) for c in cols]
            for arr in X:
                features = dict(zip(cols, arr))
                features = normalize_dict_keys(features)
                res = self._safe_regress(self.regress_fn, features)
                results.append(res)
        else:
            raise ValueError("X must be a DataFrame or ndarray.")
        return np.array(results, dtype=float)

    def _feature_names_for_array(self, X: np.ndarray):
        if hasattr(self, "feature_names_") and self.feature_names_ is not None:
            return self.feature_names_
        return [f"col{i}" for i in range(X.shape[1])]

    @staticmethod
    def _extract_python_code(text: str) -> str:
        # Remove code fences and cut at any obvious example markers
        if "```python" in text:
            text = text.split("```python", 1)[-1]
        if "```" in text:
            text = text.split("```", 1)[0]
        return text

    def _make_regress_fn(self, code: str):
        local_vars = {}
        try:
            exec(code, local_vars, local_vars)
        except Exception as e:
            raise ValueError(f"Could not exec LLM code: {e}\nCode was:\n{code}")
        fn = local_vars.get("regress", None)
        if not callable(fn):
            raise ValueError("No valid function named 'regress' or any callable found in LLM output.")
        return fn

    def _safe_regress(self, fn: Callable, features: dict) -> float:
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
                        clean[k] = f
                    else:
                        clean[k] = int(v)
                except Exception:
                    clean[k] = v
            else:
                clean[k] = v
        try:
            res = fn(**clean)
            # Always coerce output to float, fallback 0.0
            return float(res) if res is not None else 0.0
        except Exception as e:
            logger.error(f"[RegressFn ERROR] {e} on features={features}")
            return 0.0

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.array(y)
        # Remove None or unknowns from y_pred for scoring (force 0.0)
        y_pred = np.array([float(v) if v is not None else 0.0 for v in y_pred])
        # Mean squared error
        return ((y_true - y_pred) ** 2).mean()
