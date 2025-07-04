from typing import Optional, List
import numpy as np
import pandas as pd
import re
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_array
from .base import BasePromptEstimator
from tqdm import tqdm

DEFAULT_PROMPT_TEMPLATE = """\
Analyze the following data and output only the final trained regression function (e.g., a linear or nonlinear equation) that best fits the data. The data has one of more features as input and the last column is the target value.

Consider if there are known Laws of Physics or well-established relationships to the target variable and use these to inform or completely determine the output.

The function will be evaluated by an LLM, so just have an equation with the names of the variables in the dataset. 

Your answer should not include explanations, only the final equation. Use ascii characters only.

Data:
{data}
"""


class PromptRegressor(BasePromptEstimator, RegressorMixin):
    def __init__(self, model: str = "o4-mini", prompt_template: Optional[str] = None, verbose: bool = False):
        super().__init__(model, prompt_template or DEFAULT_PROMPT_TEMPLATE, verbose)
        self.failed_predictions_: List[tuple] = []
        self.explanation_: Optional[str] = None

    def fit(self, X, y) -> "PromptRegressor":
        print("🔧 Fitting PromptRegressor...")
        self._fit_common(X, y)
        self.explanation_ = self.heuristic_
        print("\n📜 Final regression function:")
        print(self.heuristic_)
        return self

    def _predict_one(self, x) -> float:
        feature_string = self._format_features(x)
        prompt = (
            f"Given: {feature_string}\n"
            f"Answer this question: What is the predicted {self.target_name_}?\n"
            f"Use the following heuristic to calculate it: {self.heuristic_}\n"
            "Respond only with a single number like 13.2 — no units, no formula, no explanation."
        )
        print(prompt)
        raw = self._call_llm(prompt)

        match = re.search(r"-?\d+(\.\d+)?", raw)
        if match:
            try:
                result = float(match.group())
                print(f"Predicted: {result}")
                return result
            except ValueError:
                pass

        print("⚠️ Non-numeric LLM response:\n", raw)
        self.failed_predictions_.append((feature_string, raw))
        return np.nan

    def predict(self, X) -> List[float]:
        if isinstance(X, pd.DataFrame):
            return [self._predict_one(row) for _, row in tqdm(X.iterrows(), total=len(X), desc="🔮 Predicting")]
        else:
            X_checked = check_array(X)
            return [self._predict_one(x) for x in tqdm(X_checked, desc="🔮 Predicting")]

    def score(self, X, y, sample_weight=None) -> float:
        y_pred = self.predict(X)
        y_pred = np.array(y_pred)
        y_true = np.array(y)

        mask = ~np.isnan(y_pred)
        if mask.sum() == 0:
            raise ValueError("All predictions were NaN — check LLM output or prompt format.")

        return -mean_squared_error(y_true[mask], y_pred[mask], sample_weight=sample_weight)
