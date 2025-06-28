# promptlearn/classifier.py

from typing import Optional, List, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array

from .base import BasePromptEstimator


DEFAULT_PROMPT_TEMPLATE = """\
You are a seasoned data scientist tasked with building a classification prompt for an LLM.

Treat the data as a sample of a much larger problem domain, so don't just memorize the data as-is.

Look at the name of the target column and figure out its meaning. It the input features seem to be text or text entities, it is OK to output a prompt that will ask the LLM to reason by itself what the target value could be.

Conduct an analysis based on the following data, and output only the final trained classifier (like a decision tree, human-readable instructions, etc) that will be conveyed in the form of an LLM prompt to another system. The rules will be executed as given so you need to have all the weights, equations, thresholds, etc in your output. The classifier should be able to accurately predict the value (class) of the last column based on the data in the other columns. Respond in plain text ascii only.

Data:
{data}
"""


class PromptClassifier(BaseEstimator, ClassifierMixin, BasePromptEstimator):
    def __init__(
        self,
        prompt_template: Optional[str] = None,
        model: str = "o4-mini",
        verbose: bool = False
    ) -> None:
        BasePromptEstimator.__init__(self, model, prompt_template or DEFAULT_PROMPT_TEMPLATE, verbose)

    def fit(self, X, y) -> "PromptClassifier":
        if not isinstance(X, pd.DataFrame):
            X, y = check_X_y(X, y)

        self.feature_names_in_ = self._get_feature_names(X)
        self.target_name_ = self._get_target_name(y)
        X_values = X.values if isinstance(X, pd.DataFrame) else X

        formatted_data = self._format_training_data(X_values, y, self.feature_names_in_, self.target_name_)
        self.training_prompt_ = self.prompt_template.format(data=formatted_data)

        self.classification_prompt_ = self._call_llm(self.training_prompt_)
        return self

    def _predict_one(self, x) -> int:
        feature_string = self._format_features(x)
        prompt = (
            self.classification_prompt_ + "\n\n"
            f"Given: {feature_string}\n"
            f"What is the predicted {self.target_name_}?\n"
            "Respond only with a number (e.g., 0, 1, 2)."
        )
        return int(self._call_llm(prompt))

    def predict(self, X) -> List[int]:
        if isinstance(X, pd.DataFrame):
            return [self._predict_one(row) for _, row in X.iterrows()]
        else:
            X_checked = check_array(X)
            return [self._predict_one(x) for x in X_checked]

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
