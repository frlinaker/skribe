# ⚡️ promptlearn

**promptlearn** supercharges your `scikit-learn` pipelines with the cognitive power of large language models.  
It lets you swap traditional machine learning estimators for LLM-backed alternatives — no retraining, no new APIs, just smarter results.

---

### 🧠 LLMs as Estimators

`promptlearn` provides plug-and-play, `scikit-learn`-compatible estimators that are powered by prompting rather than parameter tuning:

- **`PromptClassifier`** – classifies inputs through intelligent, human-like reasoning  
- **`PromptRegressor`** – uncovers numeric relationships via freeform pattern discovery

These estimators support standard `.fit()`, `.predict()`, and `.score()` methods — just like traditional models — and work seamlessly with the broader `scikit-learn` ecosystem.

---

### 🧩 Introducing the *Heuristic*

What `promptlearn` learns is not a numeric model — it's a **heuristic**: a concise, often symbolic or verbal representation of the underlying pattern in your data.

During `fit()`, the LLM analyzes your dataset and produces a heuristic —  
a human-readable rule, equation, or logic tree like:

> "If the country's flag includes blue — based on common national symbolism — then predict 1, else 0."

Or:

> "If sepal length is less than 5 and petal width is below 0.3, predict 'setosa'."

These heuristics go beyond decision boundaries — they express logic in language.
They can encode symbolic rules, equations, or even reference real-world knowledge the LLM already knows.

---

### 🔮 Beyond Machine Learning

Where standard models see data points, LLMs see **meaning**.

During `fit()`:
- The LLM interprets tabular data as natural information
- Infers structure, causality, and latent world knowledge
- Synthesizes a **heuristic**, not just parameters

During `predict()`:
- The heuristic is applied to new inputs via prompt composition
- The LLM responds with precise predictions based on learned logic

This allows for models that articulate:

> the target is an XOR of x1 and x2  
> y ≈ 2·x + 3 explains this noisy relationship  
> a human-readable decision tree like this [...] works well

---

### 🌍 Embedded World Knowledge

LLMs bring something traditional models can't: **knowledge from the outside world**.

For example:

```
Input: country_name = "Sweden"
Output: has_blue_in_flag = 1
```

This is not feature extraction — it’s *reasoning*.  
LLM-based estimators can draw from their embedded world model to make informed predictions, even when key features are missing.  
Think of it as a **web-scale join**, performed automatically at inference time.

---

### 🕳 Zero-Row Learning

**promptlearn can learn from just the column names.**

When you provide only input and target headers — no rows — the LLM will infer a plausible model from background knowledge.

Example:

```
Input columns: ['country_name']  
Target column: 'has_blue_in_flag'  
Training rows: 0  
Result: a working classifier.
```

See: [examples/zero_row_classifier.py](examples/zero_row_classifier.py)

This is fundamentally impossible in traditional ML — but not for a language model.

---

### 💾 Model Persistence (now with `joblib` support!)

Trained estimators (i.e., heuristics) can be saved and loaded with `joblib`:

```python
import joblib

# Save
joblib.dump(classifier, "model.joblib")

# Load
classifier = joblib.load("model.joblib")
classifier.predict(...)
```

The internal LLM client is safely excluded from the saved state and re-initialized on load.  
You can also inspect `.heuristic_` directly — it’s fully human-readable and portable.

---

## 🔗 Why it matters

`promptlearn` isn’t just a drop-in tool. It’s a paradigm shift:

From **pattern matching** → to **knowledge-aware reasoning**  
From **training on labeled data** → to **prompting from context**  
From **parameter optimization** → to **language-native heuristics**

---

## 📁 License

MIT © 2025 Fredrik Linaker
