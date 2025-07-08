import logging
from typing import Optional
import pandas as pd
import numpy as np
from .base import BasePromptEstimator

logger = logging.getLogger("promptlearn")

DEFAULT_PROMPT_TEMPLATE = """\
You are a seasoned data scientist. Output a single valid Python function called 'regress' that, given the feature variables (see below), predicts a continuous value.

Do NOT use any variable not defined below or present in the provided data. If you need external lookups, include them as Python lists or dicts at the top of your output.

Your function must have signature: def regress(**features): ... (or with explicit arguments).

Only output valid Python code, no markdown or explanations.

{scratchpad}
Data:
{data}
"""

class PromptRegressor(BasePromptEstimator):
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
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if isinstance(y, (np.ndarray, list)):
            y = pd.Series(y)

        target_name = y.name if y.name else "target"
        y = y.astype(float)
        X = X.copy()
        X[target_name] = y

        df_clean = X.dropna(subset=[target_name])
        if df_clean.shape[0] < 2:
            logger.warning("Not enough complete rows for prompt construction.")

        formatted_data = df_clean.head(10).to_csv(index=False)
        prompt = self.safe_format(self.prompt_template, data=formatted_data, scratchpad=scratchpad)
        if self.verbose:
            logger.info(f"[LLM Prompt]\n{prompt}")

        code = self._call_llm(prompt)
        self.heuristic_history_.append(code)
        self.heuristic_ = self._make_predict_fn(code, aux_data=getattr(self, "aux_data_", {}))
        self.target_name_ = target_name
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            if hasattr(self, "target_name_"):
                n_features = X.shape[1]
                feature_cols = [col for col in getattr(self, "feature_names_", [f"x{i}" for i in range(n_features)])]
            else:
                feature_cols = [f"x{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_cols)
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not hasattr(self, "feature_names_"):
            self.feature_names_ = list(X.columns)

        results = []
        for idx, row in X.iterrows():
            try:
                features = row.to_dict()
                result = self.heuristic_(features)
                results.append(float(result))
            except Exception as e:
                logger.error(f"[Predict Error] Row {idx}: {e}")
                results.append(None)
        return results

    def score(self, X, y):
        y_pred = self.predict(X)
        pairs = [(a, b) for a, b in zip(y_pred, y) if a is not None]
        if not pairs:
            logger.warning("No successful predictions!")
            return 0.0
        mse = sum((float(a) - float(b)) ** 2 for a, b in pairs) / len(pairs)
        return mse
