import logging
from typing import Optional
import pandas as pd
import numpy as np
import re
from .base import BasePromptEstimator

logger = logging.getLogger("promptlearn")

DEFAULT_PROMPT_TEMPLATE = """\
You are a seasoned data scientist. Output a single valid Python function called 'predict' that, given the feature variables (see below), predicts the class as an integer (e.g., 0 or 1).

Do NOT use any variable not defined below or present in the provided data. If you need external lookups, include them as Python lists or dicts at the top of your output.

**If you use a dictionary, mapping, or lookup for a categorical variable, make it EXHAUSTIVE for all possible real-world or logically possible values of that category (e.g., all countries, all brands, all colors, etc.), not just those in the sample data.**  
If you can't know all possible values, use a general fallback rule for unlisted categories.

If there is no data given, analyze the names of the input and output columns (assume the last column is the output or target column) and reason to what will be expected as an outcome, and generate code based on that.

Your function must have signature: def predict(**features): ... (or with explicit arguments).

Only output valid Python code, no markdown or explanations.

{scratchpad}
Data:
{data}
"""

def sanitize_col(col):
    # Remove units, replace spaces and non-word chars, ensure valid Python
    name = re.sub(r"[\s\-]+", "_", str(col)).strip("_")
    name = re.sub(r"[^0-9a-zA-Z_]", "", name)
    # Can't start with digit
    if re.match(r"^\d", name):
        name = "_" + name
    return name

class PromptClassifier(BasePromptEstimator):
    def __init__(
        self,
        model: str = "o4-mini",
        verbose: bool = False,
        chunk_threshold: int = 300,
        force_chunking: bool = False,
        max_chunks: Optional[int] = None,
        save_dir: Optional[str] = None,
    ):
        super().__init__(model, DEFAULT_PROMPT_TEMPLATE, verbose)
        self.chunk_threshold = chunk_threshold
        self.force_chunking = force_chunking
        self.max_chunks = max_chunks
        self.save_dir = save_dir

    def fit(self, X, y, scratchpad: str = ""):
        # Normalize input to DataFrame/Series
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(y, (np.ndarray, list)):
            y = pd.Series(y)
        orig_features = list(X.columns)
        target_orig_name = y.name if y.name else "target"

        # --- Sanitize feature/target names ---
        self.feature_names_ = orig_features
        self.col_map_ = {col: sanitize_col(col) for col in orig_features}
        self.inv_col_map_ = {v: k for k, v in self.col_map_.items()}
        X_sanitized = X.rename(columns=self.col_map_)
        sanitized_target = sanitize_col(target_orig_name)
        y = pd.Series(y, name=sanitized_target).astype(int)
        X_sanitized = X_sanitized.copy()
        X_sanitized[sanitized_target] = y

        # Only use rows with complete target for prompt
        df_clean = X_sanitized.dropna(subset=[sanitized_target])
        if df_clean.shape[0] < 2:
            logger.warning("Not enough complete rows for prompt construction.")

        formatted_data = df_clean.head(10).to_csv(index=False)
        prompt = self.safe_format(self.prompt_template, data=formatted_data, scratchpad=scratchpad)

        code = self._call_llm(prompt)
        self.heuristic_history_.append(code)
        self.heuristic_ = self._make_predict_fn(code, aux_data=getattr(self, "aux_data_", {}))
        self.target_name_ = target_orig_name
        self.sanitized_target_ = sanitized_target
        return self

    def predict(self, X):
        # Normalize input and sanitize columns
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Apply column sanitization mapping
        X_sanitized = X.rename(columns=self.col_map_)

        results = []
        for idx, row in X_sanitized.iterrows():
            try:
                features = row.to_dict()
                result = self.heuristic_(features)
                results.append(int(result))
            except Exception as e:
                logger.error(f"[Predict Error] Row {idx}: {e}")
                results.append(None)
        return results

    def score(self, X, y):
        y_pred = self.predict(X)
        pairs = [(a, b) for a, b in zip(y_pred, y) if a is not None]
        if not pairs:
            logger.warning("No successful predictions!")
            return 0
        correct = sum(int(a == b) for a, b in pairs)
        return correct / len(pairs)
