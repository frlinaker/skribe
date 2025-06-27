import os
import openai
import logging
from typing import Optional, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array


DEFAULT_PROMPT_TEMPLATE = """\
You are a seasoned data scientist tasked with building a classification prompt for an LLM.

Treat the data as a sample of a much larger problem domain, so don't just memorize the data as-is.

Look at the name of the target column and figure out its meaning. It the input features seem to be text or text entities, it is OK to output a prompt that will ask the LLM to reason by itself what the target value could be.

Conduct an analysis based on the following data, and output only the final trained classifier (like a decision tree, human-readable instructions, etc) that will be conveyed in the form of an LLM prompt to another system. The rules will be executed as given so you need to have all the weights, equations, thresholds, etc in your output. The classifier should be able to accurately predict the value (class) of the last column based on the data in the other columns. Respond in plain text ascii only.

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

    def _get_feature_names(self, X: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        """Extract or generate feature names from input X."""
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        else:
            return [f"x{i+1}" for i in range(X.shape[1])]

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, List[Union[int, str]], pd.Series]
    ) -> "PromptClassifier":
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values, y = check_X_y(X, y)

        self.feature_names_in_: List[str] = self._get_feature_names(X)
        self.target_name_: str = y.name if isinstance(y, pd.Series) and y.name else "target"

        header: List[str] = self.feature_names_in_ + [self.target_name_]
        data_rows: List[str] = ["\t".join(header)]

        for xi, yi in zip(X_values, y):
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

    def _predict_one(self, x: Union[np.ndarray, pd.Series]) -> int:
        if isinstance(x, pd.Series):
            feature_string = ", ".join(
                f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}='{v}'"
                for k, v in x.items()
            )
        else:
            feature_string = ", ".join(
                f"{name}={value:.3f}" if isinstance(value, (int, float)) else f"{name}='{value}'"
                for name, value in zip(self.feature_names_in_, x)
            )

        inference_prompt = (
            self.classification_prompt_ + "\n\n"
            f"Given: {feature_string}\n"
            f"What is the predicted {self.target_name_}?\n"
            "Respond only with a number (e.g., 0, 1, 2)."
        )

        if self.verbose:
            logging.info("Inference prompt:\n" + inference_prompt)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": inference_prompt}]
            )
            result = response.choices[0].message.content.strip()
            if self.verbose:
                logging.info("Prediction result: " + result)
            return int(result)
        except Exception as e:
            raise RuntimeError(f"Prediction failed for input {x}: {e}")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> List[int]:
        if isinstance(X, pd.DataFrame):
            return [self._predict_one(row) for _, row in X.iterrows()]
        else:
            X_checked = check_array(X)
            return [self._predict_one(x) for x in X_checked]

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, List[int], pd.Series]
    ) -> float:
        y_pred: List[int] = self.predict(X)
        return accuracy_score(y, y_pred)
