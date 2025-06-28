# promptlearn/regressor.py

from typing import Optional, List
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_X_y, check_array
from .base import BasePromptEstimator

DEFAULT_PROMPT_TEMPLATE = """\
You are a seasoned data scientist. Analyze the following data and output only the final trained regression function (e.g., a linear or nonlinear equation) that best fits the data. The data has one of more features as input and the last column is the target value.

The function must be executable as written — include weights, operations, and any thresholds required to use it as a predictive formula. Your answer should not include explanations, only the final model. Respond in plain text ascii only.

Data:
{data}
"""

class PromptRegressor(BasePromptEstimator, RegressorMixin):
    def __init__(self, model: str = "o4-mini", prompt_template: Optional[str] = None, verbose: bool = False):
        super().__init__(model, prompt_template or DEFAULT_PROMPT_TEMPLATE, verbose)

    def fit(self, X, y) -> "PromptRegressor":
        if not isinstance(X, pd.DataFrame):
            X, y = check_X_y(X, y)

        self.feature_names_in_ = self._get_feature_names(X)
        self.target_name_ = self._get_target_name(y)
        X_values = X.values if isinstance(X, pd.DataFrame) else X

        formatted_data = self._format_training_data(X_values, y, self.feature_names_in_, self.target_name_)
        self.training_prompt_ = self.prompt_template.format(data=formatted_data)

        self.regression_formula_ = self._call_llm(self.training_prompt_)
        return self

    def _predict_one(self, x) -> float:
        feature_string = self._format_features(x)
        prompt = (
            self.regression_formula_ + "\n\n"
            f"Given: {feature_string}\n"
            f"What is the predicted {self.target_name_}?\n"
            "Respond only with a number (e.g., 4.2)"
        )
        return float(self._call_llm(prompt))

    def predict(self, X) -> List[float]:
        if isinstance(X, pd.DataFrame):
            return [self._predict_one(row) for _, row in X.iterrows()]
        else:
            X_checked = check_array(X)
            return [self._predict_one(x) for x in X_checked]

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)
