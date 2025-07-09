
# promptlearn

[![GitHub last commit](https://img.shields.io/github/last-commit/frlinaker/promptlearn)](https://github.com/frlinaker/promptlearn)
[![PyPI - Version](https://img.shields.io/pypi/v/promptlearn)](https://pypi.org/project/promptlearn/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/promptlearn)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/promptlearn)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/promptlearn)
[![Licence](https://img.shields.io/github/license/frlinaker/promptlearn
)](https://mit-license.org/)

**promptlearn** brings large language models into your scikit-learn workflow. It replaces traditional estimators with language-native reasoning systems that learn, adapt, and describe patterns using natural language as the model substrate. The output is directly executable and portable Python code that is executed in a safe sandbox environment during `predict()` calls.

---

### 📊 Outperforming Traditional Models with Built-In Knowledge

`promptlearn` allows LLMs to internalize both structure and semantics during training. As a result, the models often exceed the capabilities of classical estimators when the task requires reasoning, real-world knowledge, or symbolic understanding.

Consider a simple binary classification task: predicting whether an [animal is a mammal](examples/benchmark_classifier.py) based on its name, weight, and lifespan.

Traditional models depend solely on the input features. But `promptlearn` models can use their internal understanding of zoology to form highly accurate rules. Even when a label like `"Whale"` is never seen during training, the model knows it belongs to the mammal class.

                    model  accuracy  fit_time_sec  predict_time_sec
      promptlearn_o3-mini      0.94     49.114336          0.002808
      promptlearn_o4-mini      0.86     60.961045          0.002417
promptlearn_gpt-3.5-turbo      0.66     20.246616          0.002738
       promptlearn_gpt-4o      0.66     43.930959          0.002250
      logistic_regression      0.60      0.016565          0.000962
            decision_tree      0.53      0.001409          0.000529
        gradient_boosting      0.53      0.020737          0.001094
        promptlearn_gpt-4      0.40     12.494963          0.002196
                    dummy      0.34      0.000554          0.000120
            random_forest      0.28      0.010656          0.001659

This type of semantic generalization is a powerful advantage for LLM-backed models.

---

Now compare performance on a regression task where the data contains [samples of objects falling from different heights, under different gravity](examples/benchmark_regressor.py). This is a classic physics problem, with a well-known equation:

```
fall_time_s = sqrt((2 * height_m) / gravity_mps2)
```

Recent `promptlearn` estimators are able to recover this exact formula and use it to generate near-perfect predictions:

                    model     mse  fit_time_sec  predict_time_sec
       promptlearn_gpt-4o   0.000         2.924             0.001
      promptlearn_o3-mini   0.000        10.801             0.001
      promptlearn_o4-mini   0.000         7.959             0.001
            random_forest   0.028         0.013             0.002
        gradient_boosting   0.035         0.011             0.001
            decision_tree   0.067         0.001             0.000
        linear_regression   0.498         0.001             0.000
                    dummy   5.273         0.001             0.000
promptlearn_gpt-3.5-turbo  18.193         3.009             0.002
        promptlearn_gpt-4 855.445         2.428             0.001

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

When you call `.fit()`, the LLM reviews your data and generates executable Python code that realizes the found relationships.

The result is thus a plain-text, human-readable, piece of code. It is readable, portable, and expressive. This is stored in `.heuristic_`, and it powers all predictions.

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

Even if there is no signal in the data, the model may still predict correctly by referencing background information. This creates a kind of ambient “web join” during training that gets materialized as an explicit list or dictionary that expands all categorical values that are encountered during training, to cover unseen cases. This can include countries, flags, animals, and more.

---

### 🕳 Zero-Example Learning

If you call `.fit()` with no rows — just column names — `promptlearn` will still return a working model.

This is possible because the LLM can hallucinate a plausible mapping based on:

- Column names
- Prior knowledge
- Type hints or value patterns

This makes rapid prototyping and conceptual modeling trivial.

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
| Produces runnable Python code | ❌ No               | ✅ Yes                     |
| Regression support          | ❌ No               | ✅ Yes                     |

---

## 📁 License

MIT © 2025 Fredrik Linaker
