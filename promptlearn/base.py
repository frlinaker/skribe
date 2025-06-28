import os
import openai
import logging
from typing import List, Union
import numpy as np
import pandas as pd
import warnings

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y


class BasePromptEstimator(BaseEstimator):
    model: str
    prompt_template: str
    verbose: bool
    feature_names_in_: List[str]
    heuristic_: str
    target_name_: str

    def __init__(self, model: str, prompt_template: str, verbose: bool = False):
        self.model = model
        self.prompt_template = prompt_template
        self.verbose = verbose

        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.llm_client = openai.OpenAI()

    def _get_feature_names(self, X: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        return X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"x{i+1}" for i in range(X.shape[1])]

    def _get_target_name(self, y: Union[np.ndarray, pd.Series]) -> str:
        return str(y.name) if isinstance(y, pd.Series) and y.name else "target"

    def _format_training_data(self, X: np.ndarray, y: Union[np.ndarray, List], feature_names: List[str], target_name: str) -> str:
        rows = ["\t".join(feature_names + [target_name])]
        for xi, yi in zip(X, y):
            row = list(map(str, xi)) + [str(yi)]
            rows.append("\t".join(row))
        return "\n".join(rows)

    def _format_features(self, x: Union[np.ndarray, pd.Series]) -> str:
        if isinstance(x, pd.Series):
            return ", ".join(
                f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}='{v}'"
                for k, v in x.items()
            )
        else:
            return ", ".join(
                f"{name}={value:.3f}" if isinstance(value, (int, float)) else f"{name}='{value}'"
                for name, value in zip(self.feature_names_in_, x)
            )

    def _call_llm(self, prompt: str) -> str:
        if self.verbose:
            logging.info(f"LLM prompt:\n{prompt}")
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            result = (response.choices[0].message.content or "").strip()

            if self.verbose:
                logging.info(f"LLM result: {result}")
            return result
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

    def _fit_common(self, X, y) -> None:
        if not isinstance(X, pd.DataFrame):
            X, y = check_X_y(X, y)

        self.feature_names_in_ = self._get_feature_names(X)
        self.target_name_ = self._get_target_name(y)
        X_values = X.values if isinstance(X, pd.DataFrame) else X

        formatted_data = self._format_training_data(X_values, y, self.feature_names_in_, self.target_name_)
        self.training_prompt_ = self.prompt_template.format(data=formatted_data)
        self.heuristic_ = self._call_llm(self.training_prompt_)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-pickleable objects
        if "llm_client" in state:
            del state["llm_client"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            warnings.warn(
                "OPENAI_API_KEY is not set. "
                "This PromptEstimator cannot make predictions until the key is available.",
                RuntimeWarning
            )

        openai.api_key = api_key
        self.llm_client = openai.OpenAI()
