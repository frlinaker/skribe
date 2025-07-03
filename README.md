# ⚡️ promptlearn

**promptlearn** brings large language models into your scikit-learn workflow.  
It replaces traditional estimators with language-native reasoning systems that learn, adapt, and describe patterns — using natural language as the model substrate.

---

### 📊 Outperforming Traditional Models with Built-In Knowledge

Because it can inject knowledge into the model — not just extract it from data — `promptlearn` has been shown to outperform classical models like decision trees, random forests, and even logistic regression on tasks where semantics, background facts, or real-world generalization matter.

Consider a benchmark of predicting whether an [animal is a mammal](examples/benchmark_runner.py), given its name, weight, and lifespan. While there are numerical attributes here for traditional ML models to latch onto, `promptlearn` can use its background knowledge to make the right predictions:

| Model               | Accuracy |
|---------------------|----------|
| `promptlearn-o4-mini` | **1.00** ✅ |
| `promptlearn-gpt-4o`  | 0.97     |
| `logistic_regression`| 0.60     |
| `random_forest`      | 0.46     |
| `dummy`              | 0.34     |

Only LLM-based models understood that a `"Whale"` is a mammal and a `"Shark"` is not — despite never seeing those labels in training.

This kind of reasoning cannot be learned purely from numerical patterns. It requires **semantic generalization** — and LLMs bring that to every model.

---

### 🤖 Estimators Powered by Language

`promptlearn` provides scikit-learn-compatible estimators that use LLMs as the modeling engine:

- **`PromptClassifier`** – for predicting classes through generalized reasoning
- **`PromptRegressor`** – for modeling numeric relationships in data

These estimators follow the same API as other `scikit-learn` models (`fit`, `predict`, `score`) but operate via dynamic prompt construction and few-shot abstraction — not parameter optimization.

---

### 📘 What it Learns: The Heuristic

When you call `.fit()`, the LLM reviews your data and writes out an internal heuristic — a compact representation of what it has inferred. This heuristic might describe:

- A relationship between age, hours worked, and income
- How education, gender, and occupation relate to survival rates
- Why one row differs from another

The result is a plain-text model — readable, portable, and expressive. This is stored in `.heuristic_`, and it’s used to power all predictions.

---

### 🧠 Language-Aware Reasoning

Because the models are backed by LLMs, they can reason across both structure and semantics:

- Names of columns matter
- Missing data can be explained or inferred
- World knowledge is available by default

A trained model might use context like:

> “Bachelors” typically correlates with medium income  
> “Private” workclass often means lower capital gain  
> Rows with missing `native-country` likely default to “United States”

This allows reasoning across incomplete, skewed, or lightly structured data — without hand-tuning features.

---

### 🧬 Background Knowledge Included

The LLM brings its internal knowledge graph to the modeling task. For instance:

```
Input: country = "Norway"
Output: has_blue_in_flag = 1
```

Even if there is a complete lack of information in the provided data, the model may still predict correctly by referencing its background information.

This creates a kind of ambient “web join” during training and inference — where external facts are considered part of the input.

---

### 🕳 Zero-Example Learning

If you call `.fit()` with no rows — just column names — `promptlearn` will still return a working model.

This is possible because the LLM can hallucinate a plausible mapping based on:

- Column names
- Prior knowledge
- Type hints or value patterns

This makes rapid prototyping and conceptual modeling trivial. You can build a classifier in seconds based on names alone.

---

### 🧠 Scaling with Chunked Training

To support large datasets, `promptlearn` uses a sliding window training mechanism.

During `.fit()`:
- The dataset is processed in batches (“chunks”)
- The current heuristic is passed forward like a scratchpad
- Each chunk contributes feedback and refinement
- The model evolves with each window

This allows training on limitless rows with a fixed memory budget. You can control chunk size, number of passes, and scratchpad visibility.

The process is fully transparent — if the dataset is large, `fit()` will automatically enable chunked training.

---

### 🧪 Native `.sample()` Support

You can generate synthetic rows directly from any trained model using `.sample(n)`.

This is useful for:

- Understanding what the model believes
- Creating test sets or bootstrapped data
- Building readable examples from internal logic

Example:

```
>>> model.sample(3)
fruit    is_citrus
Lime     1
Banana   0
Orange   1
```

---

### 💾 Save and Reload with `joblib`

Like any scikit-learn model, `promptlearn` estimators can be serialized:

```python
import joblib

joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")
```

The LLM client is excluded from the saved file and re-initialized on load. The heuristic remains intact, interpretable, and ready to use.

---

## 📚 Related Work

### Scikit-LLM

[Scikit-LLM](https://github.com/BeastByteAI/scikit-llm) provides zero- and few-shot classification through template-based prompting.  
It is lightweight and NLP-focused.

**promptlearn** offers a broader modeling philosophy:

| Capability                  | Scikit-LLM         | promptlearn                |
|-----------------------------|--------------------|----------------------------|
| Prompt generated during fit | ❌ No               | ✅ Yes                     |
| Regression support          | ❌ No               | ✅ Yes                     |
| Produces textual heuristics | ❌ No               | ✅ Yes                     |
| Works on tabular data       | ✅ Partial          | ✅ Full                    |
| Generates sample rows       | ❌ No               | ✅ `.sample()`             |

---

## 📁 License

MIT © 2025 Fredrik Linaker
