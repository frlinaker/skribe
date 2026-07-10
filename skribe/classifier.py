import logging

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from .base import BaseSkribeEstimator, resolve_model
from .prompt_markers import DATA_MARKER
from .utils import (
    generate_feature_dicts,
    safe_predict,
)

logger = logging.getLogger("skribe")

# Updated LLM prompt template with strong type casting and fallback instructions
DEFAULT_CLASSIFICATION_PROMPT_TEMPLATE = """
Output a single valid Python function called 'predict' that, given the feature variables (see below), predicts the class as an integer (e.g., 0, 1).

Do NOT use any variable not defined below or present in the provided data. If you need external lookups, include them as Python lists or dicts at the top of your output.

All numeric feature values may be provided as strings or numbers. At the top of your function, coerce ALL numeric variables (e.g., weight_kg, lifespan_years, etc.) to float (or int for integer features) using float(x) or int(x) before calculations or comparisons.

Your function must always return an integer class for any input, even if some features are unknown, missing, or out-of-vocabulary. Use a fallback/default prediction if no match is found — see the context block below for which training code to use as that default. Do not default to 0 unless the context block says 0 is the correct default; the codes are not ordered by frequency.

For categorical inputs, aim for complete coverage of all plausible real-world values in any mapping you make — not just the values seen in the data sample.

If there is no data given, analyze the names of the input and output columns (assume the last column is the output or target column) and reason to what will be expected as an outcome, and generate code based on that.

Your function must have signature: def predict(**features): ... (or with explicit arguments).

Every string literal MUST be valid, properly terminated Python. If a dictionary key or value contains an apostrophe (e.g. grevy's zebra), wrap that string in double quotes ("grevy's zebra"); if it contains a double quote, wrap it in single quotes. Never leave an unterminated string literal.

Only output valid Python code, no markdown or explanations.

""" + DATA_MARKER + "\n{data}\n"


class SkribeClassifier(ClassifierMixin, BaseSkribeEstimator):
    def __init__(
        self,
        model=None,
        verbose: bool = True,
        max_train_rows: int | None = None,
        max_retries: int = 2,
        web_search: bool = False,
        context_prepass: bool = True,
        vertex_location: str | None = None,
        llm_timeout: float = 120,
        reasoning_effort: str | None = None,
        reasoning_mode: str | None = None,
    ):
        super().__init__(
            model=resolve_model(model),
            verbose=verbose,
            max_train_rows=max_train_rows,
            max_retries=max_retries,
            web_search=web_search,
            context_prepass=context_prepass,
            vertex_location=vertex_location,
            llm_timeout=llm_timeout,
            reasoning_effort=reasoning_effort,
            reasoning_mode=reasoning_mode,
        )

    def fit(self, X, y, synthetic_features=None, dataset_description=None) -> "SkribeClassifier":
        y = pd.Series(y).reset_index(drop=True)
        # classes_ in sorted order, sklearn-LabelEncoder style. Encoding here
        # (rather than requiring the caller to pre-encode) means skribe always
        # knows the true label for every class code, so the context pre-pass
        # can state it instead of the LLM having to guess what an integer
        # code "should" mean (see test_context_prepass_states_true_label_mapping).
        self.classes_ = np.array(sorted(y.unique(), key=lambda v: (str(type(v)), v)))
        label_names = {i: label for i, label in enumerate(self.classes_)}
        self._code_of_ = {label: i for i, label in label_names.items()}
        y_encoded = y.map(self._code_of_).astype(int)
        y_encoded.name = y.name
        # The prompt template's generic "use a fallback (such as 0)" wording
        # steers the LLM toward always defaulting to code 0 regardless of
        # class frequency — 0 is just whichever class sorts first, not
        # necessarily common. A rule-based/memorized-branch function's
        # fallback executes on every unmatched input, so a bad fallback
        # choice can dominate accuracy even when the matched-branch logic is
        # fine (confirmed live: one cache case went from 0.03 to a
        # replayed 0.54 purely by using the majority class as the fallback
        # instead of code 0). Stating the true majority code removes the
        # guesswork.
        self.majority_class_ = int(y_encoded.value_counts().idxmax()) if len(y_encoded) else 0

        return super()._fit(
            X,
            y_encoded,
            DEFAULT_CLASSIFICATION_PROMPT_TEMPLATE,
            synthetic_features=synthetic_features,
            dataset_description=dataset_description,
            label_names=label_names,
            majority_class=self.majority_class_,
        )

    def predict(self, X) -> np.ndarray:
        if self.predict_fn is None:
            raise RuntimeError("Call fit() before predict().")
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            # Use pre-computed self.feature_names_
            results = [
                safe_predict(self.predict_fn, features)
                for features in generate_feature_dicts(X, self.feature_names_)
            ]
            return np.array(results, dtype=int)
        raise ValueError("X must be a DataFrame or ndarray.")

    def _validate_predict_fn(self, predict_fn, rows, labels=[]):
        super()._validate_predict_fn(predict_fn, rows, labels)
        for row in rows:
            result = predict_fn(**row)
            if not isinstance(result, (int, np.integer)):
                raise ValueError(
                    f"predict() returned {result!r} ({type(result).__name__}) but must "
                    f"return an int. Replace any string class names with their integer "
                    f"codes — e.g. return a dict mapping like "
                    f"{{'mammal': 0, 'bird': 1, ...}}[class_name] and return that integer."
                )

    def score(self, X, y):
        y_pred = self.predict(X)
        # Remove None or unknowns from y_pred for scoring (force 0)
        y_pred = np.array([int(v) if v is not None else 0 for v in y_pred])
        # y may be in original label space (strings, bools, ...) or already
        # the integer codes fit() produced — map through _code_of_ when
        # possible so both are compared in the same (integer) space, falling
        # back to raw values unchanged for any label unseen during fit().
        code_of = getattr(self, "_code_of_", {})
        y_true = np.array([code_of.get(v, v) for v in y])
        return (y_true == y_pred).mean()
