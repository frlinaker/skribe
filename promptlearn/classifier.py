# promptlearn/classifier.py

from typing import Optional, List
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array
from .base import BasePromptEstimator

DEFAULT_PROMPT_TEMPLATE = """\
You are a seasoned data scientist tasked with building a classification prompt for an LLM.

Treat the data as a sample of a much larger problem domain, so don't just memorize the data as-is.

Look at the name of the target column and figure out its meaning. It the input features seem to be text or text entities, it is OK to output a prompt that will ask the LLM to reason by itself what the target value could be.

Conduct an analysis based on the following data, and output only the final trained classifier (like a decision tree, human-readable instructions, etc) that will be conveyed in the form of an LLM prompt to another system. The rules will be executed as given so you need to have all the weights, equations, thresholds, etc in your output. The classifier should be able to accurately predict the value (class) of the last column based on the data in the other columns. Respond in plain text ascii only.

Data:
{data}
"""

class PromptClassifier(BasePromptEstimator, ClassifierMixin):
    def __init__(self, model: str = "o4-mini", prompt_template: Optional[str] = None, verbose: bool = False):
        super().__init__(model, prompt_template or DEFAULT_PROMPT_TEMPLATE, verbose)

    def fit(self, X, y) -> "PromptClassifier":
        self._fit_common(X, y)
        return self

    def _predict_one(self, x) -> int:
        feature_string = self._format_features(x)
        prompt = (
            self.heuristic_ + "\n\n"
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

    def score(self, X, y, sample_weight=None) -> float:
        y_pred = self.predict(X)
        return float(accuracy_score(y, y_pred, sample_weight=sample_weight))
