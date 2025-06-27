# ⚡️ promptlearn

**promptlearn** supercharges your `scikit-learn` pipelines with the cognitive power of large language models.  
It lets you swap traditional machine learning estimators for LLM-backed alternatives — no retraining, no new APIs, just smarter results.

### 🧠 LLMs as Estimators

`promptlearn` provides plug-and-play, `scikit-learn`-compatible estimators that are powered by prompting rather than parameter tuning:

- **`PromptClassifier`** – classifies inputs through intelligent, human-like reasoning  
- **`PromptRegressor`** – uncovers numeric relationships via freeform pattern discovery

Just like their traditional counterparts, these estimators support `.fit()` and `.predict()`, and work seamlessly with the entire `scikit-learn` ecosystem.  
But what happens under the hood is fundamentally different.

---

### 🔮 Beyond Machine Learning

Where standard models see data points, LLMs see **meaning**.

During `fit()`, the model:
- Parses tabular data as natural information
- Infers mathematical structure, logic, and latent world knowledge
- Condenses it into a symbolic or verbal model

During `predict()`, it:
- Applies this model (a prompt!) to new inputs
- Produces answers that reflect **not just statistical learning, but understanding**

Imagine a model that learns, and is able to express in plain English that:  
> the target is an XOR of x1 and x2  
> y ≈ 2·x + 3 explains this noisy relationship  
> a human readable decision tree like this [...] is a good prediction approach

---

### 🌍 Embedded World Knowledge

One of `promptlearn`’s most astonishing capabilities is its ability to fill in missing data using *embedded global knowledge*.  

For example:
```
Input: country_name = "Sweden"
Output: has_blue_in_flag = 1
```

This is not feature extraction. It is *reasoning*.  
Where traditional models fail on incomplete data, LLM-based estimators can infer, enrich, and generalize — effectively performing a **web-scale join** against the model's internal representation of the world.

---

### 🕳 Zero-Row Learning

**promptlearn can train on zero examples.**  
You read that right: it can learn a functioning model with *no data at all*, as long as it knows the **names** of the inputs and the target.

Example:

```
Input columns: ['country_name']
Target column: 'has_blue_in_flag'
Training rows: 0
Result: a working classifier.
```

Try it: [examples/zero_row_classifier.py](examples/zero_row_classifier.py)

This is fundamentally impossible with traditional ML. With LLMs, it's just inference.

---

## 🔗 Why it matters

`promptlearn` isn’t just a drop-in tool. It’s a paradigm shift:  
From **pattern matching** to **knowledge-aware inference**.  
From **training on data** to **prompting on context**.  
From **model parameters** to **language-native logic**.

---

## 📁 License

MIT © 2025 Fredrik Linaker
