import os
import openai
import logging
from typing import Optional, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_X_y, check_array


DEFAULT_REGRESSION_PROMPT = """\
You are a seasoned data scientist. Analyze the following data and output only the final trained regression function (e.g., a linear or nonlinear equation) that best fits the data. The data has one of more features as input and the last column is the target value.

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

    def _get_feature_names(self, X: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        else:
            return [f"x{i+1}" for i in range(X.shape[1])]

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, List[float], pd.Series]
    ) -> "PromptRegressor":
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values, y = check_X_y(X, y)

        self.feature_names_in_: List[str] = self._get_feature_names(X)
        data_rows: List[str] = ["\t".join(self.feature_names_in_ + ["target"])]

        for xi, yi in zip(X_values, y):
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

    def _predict_one(self, x: Union[np.ndarray, pd.Series]) -> float:
        if isinstance(x, pd.Series):
            feature_string: str = ", ".join(f"{k}={v:.3f}" for k, v in x.items())
        else:
            feature_string: str = ", ".join(
                f"{name}={value:.3f}" for name, value in zip(self.feature_names_in_, x)
            )

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
                messages=[{"role": "user", "content": inference_prompt}]
            )
            result: str = response.choices[0].message.content.strip()
            if self.verbose:
                logging.info(f"Regression prediction result: {result}")
            return float(result)
        except Exception as e:
            raise RuntimeError(f"Prediction failed for input {x}: {e}")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> List[float]:
        if isinstance(X, pd.DataFrame):
            return [self._predict_one(row) for _, row in X.iterrows()]
        else:
            X_checked = check_array(X)
            return [self._predict_one(x) for x in X_checked]

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, List[float], pd.Series]
    ) -> float:
        y_pred: List[float] = self.predict(X)
        return -mean_squared_error(y, y_pred)
