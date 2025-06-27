import os
import openai
import logging
from typing import Optional, List, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


DEFAULT_PROMPT_TEMPLATE = """\
You are a principal data scientist. Conduct an analysis based on the following data, and output only the final trained classifier (like a decision tree, equation, etc) that will be conveyed in the form of an LLM prompt to another system. The rules will be executed as given so you need to have all the weights, equations, thresholds, etc in your output. The classifier should be able to accurately predict the value (class) of the last column based on the data in the other columns.

Data:
{data}
"""

class PromptClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        prompt_template: Optional[str] = None,
        model: str = "o4-mini",
        verbose: bool = False
    ) -> None:
        self.prompt_template: str = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.model: str = model
        self.verbose: bool = verbose

        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.llm_client = openai.OpenAI()

    def fit(self, X: np.ndarray, y: Union[np.ndarray, List[Union[int, str]]]) -> "PromptClassifier":
        n_features: int = X.shape[1]
        headers: List[str] = [f"x{i+1}" for i in range(n_features)] + ["target"]
        data_rows: List[str] = ["\t".join(headers)]

        for xi, yi in zip(X, y):
            row: List[str] = list(map(str, xi)) + [str(yi)]
            data_rows.append("\t".join(row))

        formatted_data: str = "\n".join(data_rows)
        self.training_prompt_: str = self.prompt_template.format(data=formatted_data)

        if self.verbose:
            logging.info("Generated training prompt:\n" + self.training_prompt_)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.training_prompt_}]
            )
            self.classification_prompt_: str = response.choices[0].message.content.strip()

            if self.verbose:
                logging.info("Classifier rules:\n" + self.classification_prompt_)

        except Exception as e:
            raise RuntimeError(f"LLM failed to generate classifier prompt: {e}")

        return self

    def _predict_one(self, x: np.ndarray) -> int:
        feature_names: List[str] = [f"x{i+1}" for i in range(len(x))]
        feature_string: str = ", ".join(f"{name}={value:.3f}" for name, value in zip(feature_names, x))

        inference_prompt: str = (
            self.classification_prompt_ + "\n\n"
            f"Given: {feature_string}\n"
            "What is the predicted target class?\n"
            "Respond only with a number (e.g., 0, 1, 2)."
        )

        if self.verbose:
            logging.info("Inference prompt:\n" + inference_prompt)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": inference_prompt}]
            )
            result: str = response.choices[0].message.content.strip()
            if self.verbose:
                logging.info("Prediction result: " + result)
            return int(result)
        except Exception as e:
            raise RuntimeError(f"Prediction failed for input {x}: {e}")

    def predict(self, X: np.ndarray) -> List[int]:
        return [self._predict_one(x) for x in X]

    def score(self, X: np.ndarray, y: Union[np.ndarray, List[int]]) -> float:
        y_pred: List[int] = self.predict(X)
        return accuracy_score(y, y_pred)
