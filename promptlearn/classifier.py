from typing import Optional, List
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array
from .base import BasePromptEstimator

DEFAULT_PROMPT_TEMPLATE = """\\
You are a seasoned data scientist tasked with building a classification prompt for an LLM.

Treat the data as a sample of a much larger problem domain, so don't just memorize the data as-is. Try to minimize prediction error, not just describe dominant patterns. Include exceptions and counterexamples in your logic.

Look at the name of the target column and figure out its meaning. It the input features seem to be text or text entities, it is OK to output a prompt that will ask the LLM to reason by itself what the target value could be, but it should always result in an integer value (if it's a boolean then True=1 and False=0).

Conduct an analysis based on the following data, and output only the final trained classifier (like a decision tree, human-readable instructions, etc) that will be conveyed in the form of an LLM prompt to another system. The rules will be executed as given so you need to have all the weights, equations, thresholds, etc in your output. The classifier should be able to accurately predict the value (class) of the last column based on the data in the other columns.

If you find that a single rule is too broad, break it down into more specific cases to reduce false positives and false negatives.

Data:
{data}
"""

class PromptClassifier(BasePromptEstimator, ClassifierMixin):
    def __init__(self, model: str = "o4-mini", prompt_template: Optional[str] = None,
                 verbose: bool = False, chunk_threshold: int = 300,
                 force_chunking: bool = False, max_chunks: Optional[int] = None):
        super().__init__(model, prompt_template or DEFAULT_PROMPT_TEMPLATE, verbose)
        self.chunk_threshold = chunk_threshold
        self.force_chunking = force_chunking
        self.max_chunks = max_chunks
        self.heuristic_history_: List[str] = []

    def _normalize_target_values(self, y):
        values = pd.Series(y)
        classes = sorted(values.unique())
        self.allowed_classes_ = list(map(int, classes))
        return values.astype(int)

    def _instruction_suffix(self) -> str:
        class_str = ", ".join(str(c) for c in self.allowed_classes_)
        return (
            f"Use only numeric values when predicting {self.target_name_}.\n"
            f"For example: predict {class_str}.\n"
            f"Respond only with one of: {class_str}."
        )

    def fit(self, X, y) -> "PromptClassifier":
        self.target_name_ = self._get_target_name(y)
        y = self._normalize_target_values(y)
        if self.force_chunking or (isinstance(X, pd.DataFrame) and len(X) > self.chunk_threshold):
            if self.verbose:
                print(f"🌀 Switching to chunked fit: {len(X)} rows > threshold {self.chunk_threshold}")
            return self.fit_chunked(X, y, max_chunks=self.max_chunks)
        self._fit_common(X, y)
        return self

    def fit_chunked(self, X, y, chunk_size: int = 100, max_chunks: Optional[int] = None) -> "PromptClassifier":
        if not isinstance(X, pd.DataFrame):
            raise ValueError("fit_chunked requires a pandas DataFrame for X")

        df = X.copy()
        df[self.target_name_] = y.values if hasattr(y, "values") else y

        total_rows = len(df)
        num_chunks = (total_rows - 1) // chunk_size + 1
        if max_chunks:
            num_chunks = min(num_chunks, max_chunks)

        scratchpad = ""

        for i in range(num_chunks):
            chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
            chunk_csv = chunk.to_csv(index=False)

            prompt = f"""You are analyzing a large dataset in sequential windows.

This is the current window (rows {i * chunk_size} to {(i + 1) * chunk_size - 1}):

{chunk_csv}

Your current scratchpad (classifier-in-progress):

{scratchpad}

Update the classifier description based on this window. Use rules, thresholds, decision structures, or conditions.
Do not output predictions for this window. Instead, revise the general classifier logic.
Respond only with the full classifier description in plain text. The classifier should always output an integer value for each possible path (no string returns).
"""

            scratchpad = self._call_llm(prompt)
            self.heuristic_history_.append(scratchpad)
            if self.verbose:
                print(f"🧠 Updated heuristic after chunk {i + 1}:\n{scratchpad}\n")

        self.heuristic_ = scratchpad
        return self

    def _predict_one(self, x) -> int:
        feature_string = self._format_features(x)
        prompt = (
            self.heuristic_ + "\n\n"
            f"Given: {feature_string}\n"
            f"What is the predicted {self.target_name_}?\n"
            f"{self._instruction_suffix()}"
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

    def show_heuristic_evolution(self):
        print("🧠 Heuristic Evolution:\n")
        for i, h in enumerate(self.heuristic_history_):
            print(f"--- After chunk {i+1} ---")
            print(h.strip())
            print()
