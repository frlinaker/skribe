import logging
import numpy as np

from typing import Callable, Optional

from .base import BasePromptEstimator
from .utils import prepare_training_data, generate_feature_dicts, extract_python_code, make_predict_fn, safe_predict

logger = logging.getLogger("promptlearn")

# Updated LLM prompt template with strong type casting and fallback instructions
DEFAULT_PROMPT_TEMPLATE = """
Output a single valid Python function called 'predict' that, given the feature variables (see below), predicts the class as an integer (e.g., 0, 1).

Do NOT use any variable not defined below or present in the provided data. If you need external lookups, include them as Python lists or dicts at the top of your output.

All numeric feature values may be provided as strings or numbers. At the top of your function, coerce ALL numeric variables (e.g., weight_kg, lifespan_years, etc.) to float (or int for integer features) using float(x) or int(x) before calculations or comparisons.

Your function must always return an integer class for any input, even if some features are unknown, missing, or out-of-vocabulary. Use a fallback/default prediction (such as 0) if no match is found.

For categorical inputs, include an really exhaustive list of keys (try to get to 100+) in any mapping you make, i.e. names of countries, animals, colors, fruits, etc.

If there is no data given, analyze the names of the input and output columns (assume the last column is the output or target column) and reason to what will be expected as an outcome, and generate code based on that.

Your function must have signature: def predict(**features): ... (or with explicit arguments).

Only output valid Python code, no markdown or explanations.

Data:
{data}
"""

class PromptClassifier(BasePromptEstimator):
    def __init__(self, model="gpt-4o", verbose: bool = True, max_train_rows: int = 100):
        super().__init__(model=model, verbose=verbose)
        self.max_train_rows = max_train_rows
        self.python_code_: Optional[str] = None
        self.predict_fn: Optional[Callable] = None
        self.target_name_: Optional[str] = None
        self.feature_names_: Optional[list] = None

    def fit(self, X, y):
        data, self.feature_names_, self.target_name_ = prepare_training_data(X, y)

        # Use a small sample for LLM to avoid expensive calls
        if len(data) > self.max_train_rows:
            logger.info(f"Reducing training data from {data.shape[0]:,} to {self.max_train_rows:,} rows for LLM.")
            sample_df = data.sample(self.max_train_rows, random_state=42)
        else:
            sample_df = data

        csv_data = sample_df.to_csv(index=False)

        prompt = DEFAULT_PROMPT_TEMPLATE.format(data=csv_data)
        logger.info(f"[LLM Prompt]\n{prompt}")

        # Call LLM and get code
        code = self._call_llm(prompt)
        if not isinstance(code, str):
            code = str(code)
        logger.info(f"[LLM Output]\n{code}")

        # Remove markdown/code block if present (triple backticks)
        code = extract_python_code(code)
        if not code.strip():
            logger.error("LLM output is empty after removing markdown/code block.")
            raise ValueError("No code to exec from LLM output.")

        self.python_code_ = code
        print(f"the cleaned up code is: [START]{code}[END]")

        # Compile the code into a function
        self.predict_fn = make_predict_fn(code)

        return self

    def predict(self, X) -> np.ndarray:
        if self.predict_fn is None:
            raise RuntimeError("Call fit() before predict().")
        # Use pre-computed self.feature_names_
        results = [
            safe_predict(self.predict_fn, features)
            for features in generate_feature_dicts(X, self.feature_names_)
        ]
        return np.array(results, dtype=int)

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.array(y)
        # Remove None or unknowns from y_pred for scoring (force 0)
        y_pred = np.array([int(v) if v is not None else 0 for v in y_pred])
        return (y_true == y_pred).mean()
