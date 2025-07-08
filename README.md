
# promptlearn

[![GitHub last commit](https://img.shields.io/github/last-commit/frlinaker/promptlearn)](https://github.com/frlinaker/promptlearn)
[![PyPI - Version](https://img.shields.io/pypi/v/promptlearn)](https://pypi.org/project/promptlearn/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/promptlearn)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/promptlearn)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/promptlearn)
[![Licence](https://img.shields.io/github/license/frlinaker/promptlearn
)](https://mit-license.org/)

**promptlearn** brings large language models into your scikit-learn workflow.  
It replaces traditional estimators with language-native reasoning systems that learn, adapt, and describe patterns using natural language as the model substrate.

---

### 📊 Outperforming Traditional Models with Built-In Knowledge

`promptlearn` allows LLMs to internalize both structure and semantics during training. As a result, the models often exceed the capabilities of classical estimators when the task requires reasoning, real-world knowledge, or symbolic understanding.

Consider a simple binary classification task: predicting whether an [animal is a mammal](examples/data/mammal_train.csv) based on its name, weight, and lifespan.

Traditional models depend solely on the input features. But `promptlearn` models can use their internal understanding of zoology to form highly accurate rules. Even when a label like `"Whale"` is never seen during training, the model knows it belongs to the mammal class.

| Model                 | Accuracy |
|-----------------------|----------|
| `promptlearn-o4-mini` | **1.00** |
| `promptlearn-gpt-4o`  | 0.97     |
| `logistic_regression`| 0.60     |
| `random_forest`       | 0.46     |
| `dummy`               | 0.34     |

This type of semantic generalization is a powerful advantage for LLM-backed models.

---

Now compare performance on a regression task where the data contains [samples of objects falling from different heights, under different gravity](examples/data/fall_train.csv). This is a classic physics problem, with a well-known equation:

```
fall_time_s = sqrt((2 * height_m) / gravity_mps2)
```

Recent `promptlearn` estimators are able to recover this exact formula and use it to generate near-perfect predictions:

| Model                  | MSE       |
|------------------------|-----------|
| `promptlearn-o4-mini`  | **0.00006** |
| `promptlearn-gpt-4o`   | 0.00006   |
| `gradient_boosting`    | 0.035     |
| `linear_regression`    | 0.498     |
| `dummy`                | 5.27      |
| `promptlearn-gpt-4`    | 43.17     |

No feature engineering was performed. No physics constants were added. The model discovered the rule and applied it directly. Classical regressors, by contrast, approximated a curve but missed the exact structure.

These results highlight the practical benefit of reasoning models: they learn compact, expressive heuristics and can outperform traditional systems when symbolic insight or background knowledge is essential.

---

### 🤖 Estimators Powered by Language

`promptlearn` provides scikit-learn-compatible estimators that use LLMs as the modeling engine:

- **`PromptClassifier`** – for predicting classes through generalized reasoning
- **`PromptRegressor`** – for modeling numeric relationships in data

These estimators follow the same API as other `scikit-learn` models (`fit`, `predict`, `score`) but operate via dynamic prompt construction and few-shot abstraction.

---

### 📘 What it Learns: The Heuristic

When you call `.fit()`, the LLM reviews your data and writes out an internal heuristic — a compact representation of what it has inferred. This heuristic might describe:

- A relationship between age, hours worked, and income
- How education, gender, and occupation relate to survival rates
- Why one row differs from another

The result is a plain-text model. It is readable, portable, and expressive. This is stored in `.heuristic_`, and it powers all predictions.

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

This allows reasoning across incomplete, skewed, or lightly structured data without hand-tuning features.

---

### 🧬 Background Knowledge Included

The LLM brings its internal knowledge graph to the modeling task. For instance:

```
Input: country = "Norway"
Output: has_blue_in_flag = 1
```

Even if there is no signal in the data, the model may still predict correctly by referencing background information. This creates a kind of ambient “web join” during training and inference.

---

### 🕳 Zero-Example Learning

If you call `.fit()` with no rows — just column names — `promptlearn` will still return a working model.

This is possible because the LLM can hallucinate a plausible mapping based on:

- Column names
- Prior knowledge
- Type hints or value patterns

This makes rapid prototyping and conceptual modeling trivial.

---

### 🧠 Scaling with Chunked Training

To support large datasets, `promptlearn` uses a sliding window training mechanism.

During `.fit()`:
- The dataset is processed in batches (“chunks”)
- The current heuristic is passed forward like a scratchpad
- Each chunk contributes feedback and refinement
- The model evolves with each window

This allows training on limitless rows using a fixed memory budget. The process is transparent. If the dataset is large, chunked training activates automatically.

---

### 🧪 Native `.sample()` Support

You can generate synthetic rows directly from any trained model using `.sample(n)`:

```
>>> model.sample(3)
fruit    is_citrus
Lime     1
Banana   0
Orange   1
```

This is useful for:

- Understanding what the model believes
- Creating test sets or bootstrapped data
- Building readable examples from internal logic

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
