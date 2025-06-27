# promptlearn

LLM-powered estimators like `PromptClassifier` and `PromptRegressor`, designed to plug into `scikit-learn` pipelines.

## Installation

```bash
pip install .
```

## Example

```python
from promptlearn import PromptClassifier

clf = PromptClassifier(
    prompt_template="Title: {{title}}\nTL;DR: {{tldr}}\nClassify as -1 or 1",
    llm_client=lambda prompt: "1"
)
clf.fit(X=[], y=[])  # No-op for now
print(clf.predict([{"title": "Test", "tldr": "Summary"}]))
```
