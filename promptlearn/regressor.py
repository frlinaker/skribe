import os
import openai
import logging
from typing import Optional, List, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error


DEFAULT_REGRESSION_PROMPT = """\
You are a principal data scientist. Analyze the following data and output only the final trained regression function (e.g., a linear or nonlinear equation) that best fits the data. The data has one of more features as input and the last column is the target value.

The function must be executable as written — include weights, operations, and any thresholds required to use it as a predictive formula. Your answer should not include explanations, only the final model. Respond in plain text ascii only.

Data:
{data}
"""

class PromptRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        prompt_template: Optional[str] = None,
        model: str = "o4-mini",
        verbose: bool = False
    ) -> None:
        self.prompt_template: str = prompt_template or DEFAULT_REGRESSION_PROMPT
        self.model: str = model
        self.verbose: bool = verbose

        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.llm_client = openai.OpenAI()

    def fit(self, X: np.ndarray, y: Union[np.ndarray, List[float]]) -> "PromptRegressor":
        n_features: int = X.shape[1]
        headers: List[str] = [f"x{i+1}" for i in range(n_features)] + ["target"]
        data_rows: List[str] = ["\t".join(headers)]

        for xi, yi in zip(X, y):
            row: List[str] = list(map(str, xi)) + [str(yi)]
            data_rows.append("\t".join(row))

        formatted_data: str = "\n".join(data_rows)
        self.training_prompt_: str = self.prompt_template.format(data=formatted_data)

        if self.verbose:
            logging.info("Generated regression prompt:\n" + self.training_prompt_)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.training_prompt_}]
            )
            self.regression_formula_: str = response.choices[0].message.content.strip()

            if self.verbose:
                logging.info("Learned regression formula:\n" + self.regression_formula_)

        except Exception as e:
            raise RuntimeError(f"LLM failed to generate regression prompt: {e}")

        return self

    def _predict_one(self, x: np.ndarray) -> float:
        feature_names: List[str] = [f"x{i+1}" for i in range(len(x))]
        feature_string: str = ", ".join(f"{name}={value:.3f}" for name, value in zip(feature_names, x))

        inference_prompt: str = (
            self.regression_formula_ + "\n\n"
            f"Given: {feature_string}\n"
            "What is the predicted target value?\n"
            "Respond only with a number (e.g., 4.2)"
        )

        if self.verbose:
            logging.info("Regression inference prompt:\n" + inference_prompt)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": inference_prompt}],
            )
            result: str = response.choices[0].message.content.strip()
            if self.verbose:
                logging.info(f"Regression prediction result: {result}\n")
            return float(result)
        except Exception as e:
            raise RuntimeError(f"Prediction failed for input {x}: {e}")

    def predict(self, X: np.ndarray) -> List[float]:
        return [self._predict_one(x) for x in X]

    def score(self, X: np.ndarray, y: Union[np.ndarray, List[float]]) -> float:
        y_pred: List[float] = self.predict(X)
        return -mean_squared_error(y, y_pred)
