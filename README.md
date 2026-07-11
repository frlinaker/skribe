
# skribe

[![GitHub last commit](https://img.shields.io/github/last-commit/frlinaker/skribe)](https://github.com/frlinaker/skribe)
[![PyPI - Version](https://img.shields.io/pypi/v/skribe)](https://pypi.org/project/skribe/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/skribe)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/skribe)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/skribe)
[![Licence](https://img.shields.io/github/license/frlinaker/skribe
)](https://mit-license.org/)

**skribe compiles a large language model's reasoning and world knowledge into a small, fast, auditable scikit-learn model.** The LLM writes a standalone Python `predict()` function once, at `fit()` time. After that, predictions just run that code — no API calls, no network, no per-row LLM cost.

The resulting model:

- **knows things your columns don't encode** — a country's GDP, an animal's number of legs, whether a flag contains blue, a date's weekday — because the LLM bakes that world knowledge into explicit lookup tables in the generated code;
- **predicts like plain Python** — microsecond latency, deterministic, offline, no API key needed at inference time;
- **is fully inspectable** — read the generated source, or call `.explain()` for a plain-English description of the rule;
- **is a scikit-learn citizen** — `fit` / `predict` / `score`, plus `Pipeline`, `GridSearchCV`, `MultiOutputRegressor`, `clone`, and `joblib`.

In short: the reasoning of an LLM, with the cost, latency, and transparency of classical code.

---

## Install

```bash
pip install skribe
```

Set an API key for your provider (`OPENAI_API_KEY` by default — see [Providers](#-choose-your-provider)). The LLM is called **only at `fit()` time**.

---

## 60-second example

Fit on **column names alone — no training rows** — and let the model reason from world knowledge:

```python
import pandas as pd
from skribe import SkribeClassifier

clf = SkribeClassifier()
clf.fit(
    pd.DataFrame(columns=["country_name"]),          # no rows, just the schema
    pd.Series(name="has_blue_in_flag", dtype=int),
)

clf.predict(pd.DataFrame([{"country_name": "France"}]))   # -> [1]  (blue in flag)
clf.predict(pd.DataFrame([{"country_name": "Japan"}]))    # -> [0]  (no blue)

print(clf.python_code_)   # the exact Python the model runs — inspect it
print(clf.explain())      # plain-English description of the learned rule
```

At `fit()` the LLM wrote a `predict()` function (including a country→flag-colors table). Every prediction afterwards is pure Python — fast, deterministic, and offline.

---

## Why skribe?

There are three ways to put an LLM near tabular data. Only one leaves you with a deployable artifact:

| | Classical ML (XGBoost, logreg) | Per-row LLM calls (e.g. prompt-per-row) | **skribe** |
|---|:--:|:--:|:--:|
| Uses world knowledge beyond the columns | ❌ | ✅ | ✅ |
| Cost after fitting | free | \$ per prediction | **free** |
| Inference latency | µs | network round-trip | **µs** |
| Runs offline / no key at predict time | ✅ | ❌ | ✅ |
| Deterministic predictions | ✅ | ❌ | ✅ |
| Auditable artifact | partial | ❌ (just a prompt) | ✅ (readable code) |
| Training data required | lots | none | **little or none** |

**What's genuinely new:** the LLM is used as a *one-time program synthesizer*, not an inference engine. skribe turns "what an LLM would predict here" into a compact program you can read, version, serialize, and run anywhere — and it **materializes** the model's world knowledge into explicit tables instead of leaving it implicit in weights or a prompt.

**Reach for it when** your data has semantically meaningful columns (places, names, products, categories, dates) where outside knowledge helps, you have little labeled data, or you need an interpretable model that's cheap to serve. See [When *not* to use it](#when-not-to-use-it).

---

## Proof: every new model generation is a free accuracy upgrade

We ran 10 model generations — from `gpt-4o-mini` (mid-2024) to `GPT-5.6 Sol` with web search (today) — against the same 16 OpenML classification datasets and the same three classical baselines (logistic regression, XGBoost, TabPFN). No dataset changed, no baseline changed, only the model name string changed:

| model | released | mean accuracy |
| --- | --- | ---: |
| `gpt-4o-mini` | 2024 | 0.527 |
| `gpt-4.1` | 2025 | 0.690 |
| `gpt-5.4-mini` | 2025 | 0.693 |
| `gpt-5.5` | 2025 | 0.852 |
| `GPT-5.6 Sol` +web | 2026 | **0.868** |
| — | | |
| `logreg` (baseline) | — | 0.868 |
| `xgboost` (baseline) | — | 0.900 |
| `tabpfn` (baseline) | — | 0.913 |

Two years ago the direct classifier barely beat a coin flip. Today `GPT-5.6 Sol` *ties* logistic regression on average and is closing on XGBoost and TabPFN — **without a single line of skribe's own code changing.** Swap the model string, hit retrain, and the ceiling moves. Reproduce with [`benchmarks/run_openml_benchmark.py`](benchmarks/run_openml_benchmark.py).

**Standout wins:** `GPT-5.6 Sol` beats every baseline outright on `kr-vs-kp` (chess endgames, 0.486 → **1.000** in one year) and `tic-tac-toe` (**1.000**), and ties the best baseline on `lymph` and `monks-2`. `nursery` alone moved from 0.312 to 0.990.

<details>
<summary><b>Where it still trails, and why</b> (two systematic failure modes, not twelve unrelated ones)</summary>

skribe still trails the best baseline on 12 of 16 datasets, but the gaps cluster into two fixable patterns:

- **Trusts books rather than observations.** On `hepatitis` (0.692 vs. 0.872) and `credit-g` (0.592 vs. 0.768), the model names the right variables and the right direction of effect, but substitutes a generic textbook threshold for the number these specific datasets actually need — despite web search being available, zero searches were issued on either. Right markers, wrong cutoffs: a calibration problem, not a knowledge problem.
- **Proprietary, undocumented features.** On `spotify-genre` (0.508 vs. 0.649), columns like `speechiness` and `energy` are opaque proprietary audio scores with no semantic anchor anywhere on the public web — the model fell back on hardcoding artist names into genre buckets rather than looking anything up, because there was nothing to find.

Neither failure needs a more capable model to fix — both are fixed by giving `fit()` a smarter process: a classical optimizer to calibrate thresholds, or a real retrieval step instead of a guess.

**Honest caveat:** all 16 datasets are decades-old and public, so we can't fully rule out memorization inflating some of this trend — read it as directional, not as controlled science. A genuinely new, unpublished dataset would settle the question; that's the natural next benchmark to add.
</details>

---

## The estimators

All three follow the standard scikit-learn API (`fit` / `predict` / `score`) and are interchangeable with any other sklearn component.

- **`SkribeClassifier`** — predicts classes by reasoning over the schema, the data, and world knowledge.
- **`SkribeRegressor`** — models numeric relationships, including ones with exact closed-form structure (see below).
- **`SkribeFeatureEngineer`** — a transformer that derives new, world-knowledge-rich features for a downstream classical model.

### Feature engineering (`SkribeFeatureEngineer`)

At `fit()` the LLM writes a `transform()` function that derives new features from semantically meaningful columns (mapping a country to its GDP tier, parsing a date into `is_weekend`, bucketing ages). At `transform()` it just runs that code — **no per-row LLM calls** — and appends the engineered columns, so it drops straight into a `Pipeline` before any classical model:

```python
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from skribe import SkribeFeatureEngineer

# SkribeFeatureEngineer appends engineered columns to the original frame, so a
# downstream linear model still wants the categoricals one-hot encoded.
encode = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), selector(dtype_exclude="number"))],
    remainder="passthrough",
)
pipe = Pipeline([
    ("features", SkribeFeatureEngineer()),  # LLM-generated feature code
    ("encode", encode),
    ("model", LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
```

Routing through a classical model this way is the fix for the "opaque, undocumented features" failure mode in the benchmark above (e.g. `spotify-genre`): a fast, interpretable linear model lifted by the LLM's world knowledge, rather than asking the LLM to classify directly from columns it can't interpret.

### It recovers exact structure, not just correlations

Give `SkribeRegressor` samples of objects falling from various heights under various gravities, and it recovers the physics rather than approximating a curve:

```
fall_time_s = sqrt((2 * height_m) / gravity_mps2)
```

No feature engineering, no constants supplied — the model identifies the closed-form equation and applies it directly, where classical regressors only fit an approximate surface. Try it: `python examples/quickstart.py --demo compare --dataset fall`.

---

## More capabilities

**Explain the rule.** `.explain()` returns a plain-English description (global by default and cached, so it's deterministic; `explain(X)` describes a single prediction). It's an `Explanation` object with `meta`/`data` dicts, JSON round-trippable via `to_json()` / `from_json()`.

```python
>>> print(model.explain())
Predicts 1 (adult) when `age` is at least 18, otherwise 0.
>>> model.explain().features_used
['age']
```

**Generate synthetic rows.** `.sample(n)` asks the fitted model to emit example rows — handy for sanity-checking what it believes or bootstrapping test data.

**Save and reload with joblib.** Estimators serialize like any sklearn model. The compiled function is dropped on dump and recompiled on load, so the file is tiny and contains **only code and metadata — never an API key or LLM client**:

```python
import joblib
joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")   # ready to predict, no LLM needed
```

**Zero-example learning.** Call `.fit()` with just column names (no rows) and the model infers a rule from the schema and its prior knowledge — ideal for rapid prototyping (see the 60-second example).

---

## 🔌 Choose your provider

The provider is selected by the `model` string and resolved via [LiteLLM](https://github.com/BerriAI/litellm), so you aren't locked into OpenAI:

```python
SkribeClassifier(model="gpt-5.5")            # OpenAI (the default)
SkribeClassifier(model="claude-sonnet-4-6")  # Anthropic
SkribeClassifier(model="ollama:llama3.1")    # local Ollama
```

API keys are read from the usual per-provider environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …); local providers like Ollama need none. To change the default without touching code, set `SKRIBE_MODEL` (e.g. `export SKRIBE_MODEL=gpt-5.4-mini` for faster, cheaper runs); an explicit `model=` argument always wins.

---

## How it works

`fit()` runs a short pipeline, all at training time:

1. **Synthesize.** The LLM is prompted with the column names and a sample of rows, and returns a standalone `predict()` (or `transform()`) function.
2. **Validate & retry.** The generated code is compiled and run over the training sample. If it raises, the error is fed back to the LLM and it tries again (up to `max_retries`).
3. **Extend.** A second pass expands any categorical lookup tables with more real-world keys, so the model generalizes beyond the rows it saw.
4. **Compile.** The validated function is stored on the estimator.

`predict()` simply calls that compiled function. The LLM is never invoked at inference, and serialization captures only the code and fitted metadata.

---

## When *not* to use it

- **Opaque / encoded features.** When columns carry no semantic meaning (hashed IDs, cryptic codes), the LLM has no knowledge to exploit and direct classification can underperform a trained model — use `SkribeFeatureEngineer` + a classical model, or just a classical model.
- **Abundant data, no semantic columns.** A gradient-boosted tree is likely simpler and stronger.
- **High-stakes decisions without review.** The generated code is readable *precisely so you can audit it* — do so before relying on it.
- **Need a deterministic `fit`.** `fit()` calls an LLM and is non-deterministic; the *resulting model* is fully deterministic, but two fits may produce different code.

---

## 🚀 Try it

Everything runnable lives in a single guided tour, [`examples/quickstart.py`](examples/quickstart.py) — a menu of self-contained demos. Each makes live LLM calls, so run them one at a time:

```bash
python examples/quickstart.py --list                              # see all the demos
python examples/quickstart.py --demo zero_row                     # fit on column names only
python examples/quickstart.py --demo feature_engineer             # LLM feature engineering
python examples/quickstart.py --demo compare --dataset mammal     # skribe vs sklearn/XGBoost
python examples/quickstart.py --demo titanic --dump artifacts/    # deep tour: code, explain(), joblib
```

Demos cover zero-row fitting, `.sample()`, joblib round-tripping, world-knowledge reasoning, linear/nonlinear/multi-output regression, XOR, `GridSearchCV`, a large OpenML dataset, feature engineering, the side-by-side `compare`, and a deep `titanic` walkthrough. The `compare` demo uses the reusable `skribe.compare_models(models, X_train, y_train, X_test, y_test)` helper, which works with any mix of skribe and sklearn/XGBoost estimators.

---

## Related work

[Scikit-LLM](https://github.com/BeastByteAI/scikit-llm) brings LLMs to scikit-learn via template-based zero-/few-shot prompting — lightweight and NLP-focused, with an LLM call **per prediction**. skribe takes a different stance: the LLM writes code **once**, and that code does the predicting.

| Capability | Scikit-LLM | skribe |
|---|:--:|:--:|
| Produces runnable Python code | ❌ | ✅ |
| LLM calls at inference time | per row | **none** |
| Regression support | ❌ | ✅ |
| Feature-engineering transformer | ❌ | ✅ |
| Built-in explanations / serialization | ❌ | ✅ |

---

## 🛠 Development

```bash
pip install -r requirements-dev.txt
pre-commit install
```

The pre-commit hooks run [black](https://github.com/psf/black) and the full test suite, both of which must pass before a commit is allowed. The suite makes live LLM calls, so it needs a provider key (e.g. `OPENAI_API_KEY`); it runs against `gpt-5.4-mini` by default for speed. Release steps are documented in [`RELEASING.md`](RELEASING.md).

---

## License

MIT © 2025-2026 Fredrik Linaker
