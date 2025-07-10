
# promptlearn

[![GitHub last commit](https://img.shields.io/github/last-commit/frlinaker/promptlearn)](https://github.com/frlinaker/promptlearn)
[![PyPI - Version](https://img.shields.io/pypi/v/promptlearn)](https://pypi.org/project/promptlearn/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/promptlearn)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/promptlearn)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/promptlearn)
[![Licence](https://img.shields.io/github/license/frlinaker/promptlearn
)](https://mit-license.org/)

**promptlearn** brings large language models into your scikit-learn workflow. It is able to look at data, reason about the meaning of inputs and outputs, relate it to and identify relevant knowledge of the world, automatically building standalone executable Python code that augments the relationships of the original data with relevant materialized world-knowledge about categorical variables.

---

### 📊 Outperforming Traditional Models with Built-In Knowledge

Consider a simple binary classification task: predicting whether an [animal is a mammal](examples/benchmark_classifier.py) given things like its name, weight, and lifespan.

Traditional models depend solely on the input features. But `promptlearn` models can use their internal understanding of zoology to form highly accurate rules, pulling in data about known mammals, and making that knowledge available in explicit reference tables for subsequent predictions.

|             model             | accuracy (higher is better) | fit_time_sec | predict_time_sec |
|:------------------------------|---------:|-------------:|-----------------:|
|      promptlearn_o3-mini      |   0.94   |   49.11      |      0.0028      |
|      promptlearn_o4-mini      |   0.86   |   60.96      |      0.0024      |
| promptlearn_gpt-3.5-turbo     |   0.66   |   20.25      |      0.0027      |
|      promptlearn_gpt-4o       |   0.66   |   43.93      |      0.0023      |
|      logistic_regression      |   0.60   |   0.02       |      0.0010      |
|        decision_tree          |   0.53   |   0.0014     |      0.0005      |
|      gradient_boosting        |   0.53   |   0.02       |      0.0011      |
|      promptlearn_gpt-4        |   0.40   |   12.49      |      0.0022      |
|            dummy              |   0.34   |   0.0006     |      0.0001      |
|        random_forest          |   0.28   |   0.01       |      0.0017      |

This type of semantic generalization is a powerful advantage for LLM-backed models.

---

Now compare performance on a regression task where the data contains samples of [objects falling from different heights, under different gravity](examples/benchmark_regressor.py). This is a classic physics problem, with a well-known equation:

```
fall_time_s = sqrt((2 * height_m) / gravity_mps2)
```

`promptlearn` estimators are able to recover this exact formula, using just the dataframe itself, and use it to generate perfect predictions:

|              model             |   mse (lower is better)  | fit_time_sec | predict_time_sec |
|:-------------------------------|--------:|-------------:|-----------------:|
|      promptlearn_gpt-4o        |  0.000  |     2.92     |      0.001       |
|     promptlearn_o3-mini        |  0.000  |    10.80     |      0.001       |
|     promptlearn_o4-mini        |  0.000  |     7.96     |      0.001       |
|        random_forest           |  0.028  |     0.01     |      0.002       |
|      gradient_boosting         |  0.035  |     0.01     |      0.001       |
|        decision_tree           |  0.067  |     0.001    |      0.000       |
|      linear_regression         |  0.498  |     0.001    |      0.000       |
|             dummy              |  5.273  |     0.001    |      0.000       |
| promptlearn_gpt-3.5-turbo      | 18.193  |     3.01     |      0.002       |
|      promptlearn_gpt-4         | 855.445 |     2.43     |      0.001       |

No feature engineering was performed. No physics constants were added. The model discovered the rule and applied it directly. Classical regressors, by contrast, approximated a curve but missed the exact structure.

These results highlight the practical benefit of reasoning models: they learn compact, expressive heuristics and can outperform traditional systems when symbolic insight or background knowledge is essential.

---

### 🤖 Estimators Powered by Language

`promptlearn` provides scikit-learn-compatible estimators that use LLMs as the modeling engine:

- **`PromptClassifier`** – for predicting classes through generalized reasoning
- **`PromptRegressor`** – for modeling numeric relationships in data

These estimators follow the same API as other `scikit-learn` models (`fit`, `predict`, `score`) but operate via dynamic prompt construction and few-shot abstraction.

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
