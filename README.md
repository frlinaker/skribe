# promptlearn

**promptlearn** provides `scikit-learn`-compatible estimators powered by LLMs, such as:

- `PromptClassifier` – generates a classifier prompt using data
- `PromptRegressor` – coming soon

These estimators turn structured datasets into LLM prompts that encode classification or regression logic. They're great for exploring interpretable, zero-shot or few-shot predictive models.

---

## 📦 Installation

Coming soon to PyPI.

For now, clone the repo and install in editable mode:

```bash
git clone https://github.com/erkirk/promptlearn.git
cd promptlearn
pip install -e .
```

---

## 🚀 Examples

See the [`examples/`](examples/) folder for runnable demos. Below are two key examples:

<details>
<summary><strong>🪷 Iris Classifier (traditional Scikit-learn example)</strong></summary>

```python
from promptlearn import PromptClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

clf = PromptClassifier(verbose=True)
clf.fit(X_train, y_train)
print(clf.predict(X_test[:1]))
```

## 🧠 What the LLM Sees

During training, the `PromptClassifier` sends a tabular version of the training data to the LLM and asks it to generate a reusable classifier.

Here’s an example of a prediction prompt that the LLM returns (via GPT-4, June 2025):

```text
Given the data, a decision tree classifier can be used to predict the target class.

1. If x4 <= 0.8, then target = 0
2. If x4 > 0.8 and x4 <= 1.75:
   - If x3 <= 4.95:
     - If x4 <= 1.65, then target = 1
     - Else, target = 2
   - Else:
     - If x4 <= 1.55, then target = 2
     - Else:
       - If x1 <= 6.95, then target = 1
       - Else, target = 2
3. If x4 > 1.75, then target = 2

Respond with the predicted target given a feature string.
```

## 🎯 Inference Output

When you pass a new data point into `.predict()`:

```python
x = [[5.1, 3.5, 1.4, 0.2]]
y_pred = clf.predict(x)
```

The LLM appends this data to the prediction prompt along with a few additional hardcoded instructions:

```text
Given: x1=5.100, x2=3.500, x3=1.400, x4=0.200
What is the predicted target class?
Respond only with a number (e.g., 0, 1, 2).
```

The prediction LLM responds:

```text
Prediction result: 0
```

Which let's us decode the prediction to 'setosa' which is correct.

</details>

<details>
<summary><strong>🧠 XOR Logic (from raw data to symbolic inference rules)</strong></summary>

```python
import numpy as np
from promptlearn import PromptClassifier

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR

clf = PromptClassifier(verbose=True)
clf.fit(X, y)
```

The key point of this example is that `fit()` *realizes* that the raw data is in fact a logical XOR. This lets the LLM generate a prediction prompt that is very compact:

```
Based on the provided data, it appears that the target value is the result of a XOR operation on the values of x1 and x2.
Therefore, the classifier can be represented as follows:

IF x1 XOR x2 THEN target = 1
ELSE target = 0
```

It should not be understated how powerful this is! The system has inferred a compact well-established higher-level expression of a pattern only presented to it as raw data.

</details>

## 🧪 Development Status

This is an experimental package. Use it to:

- Build explainable prompt-based classifiers
- Generate natural language decision rules from data
- Evaluate how LLMs evolve as classifiers over time

---

## 📁 License

MIT © 2025 Fredrik Linaker
