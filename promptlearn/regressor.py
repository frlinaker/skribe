import os
import openai
import logging

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

DEFAULT_REGRESSION_PROMPT = """\
You are a principal data scientist. Analyze the following data and output only the final trained regression function (e.g., a linear or nonlinear equation) that best fits the target values from the input features. Consider higher-order polynomials, interactions, and other nonlinear transformations of the input features.

The function must be executable as written — include weights, operations, and any thresholds required to use it as a predictive formula. Your answer should not include explanations, only the final model.

Data:
{data}
"""

class PromptRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        prompt_template=None,
        model="gpt-4",
        temperature=0.0,
        max_tokens=800,
        verbose=False
    ):
        self.prompt_template = prompt_template or DEFAULT_REGRESSION_PROMPT
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.llm_client = openai.OpenAI()

    def fit(self, X, y):
        n_features = X.shape[1]
        headers = [f"x{i+1}" for i in range(n_features)] + ["target"]
        data_rows = ["\t".join(headers)]

        for xi, yi in zip(X, y):
            row = list(map(str, xi)) + [str(yi)]
            data_rows.append("\t".join(row))

        formatted_data = "\n".join(data_rows)
        self.training_prompt_ = self.prompt_template.format(data=formatted_data)

        if self.verbose:
            logging.info("Generated regression prompt:\n" + self.training_prompt_)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.training_prompt_}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            self.regression_formula_ = response.choices[0].message.content.strip()

            if self.verbose:
                logging.info("Learned regression formula:\n" + self.regression_formula_)

        except Exception as e:
            raise RuntimeError(f"LLM failed to generate regression prompt: {e}")

        return self

    def _predict_one(self, x):
        feature_names = [f"x{i+1}" for i in range(len(x))]
        feature_string = ", ".join(f"{name}={value:.3f}" for name, value in zip(feature_names, x))

        inference_prompt = (
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
                temperature=0.0,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip()
            return float(result)
        except Exception as e:
            raise RuntimeError(f"Prediction failed for input {x}: {e}")

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def score(self, X, y):
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)
