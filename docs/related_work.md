# Related Work and Novelty

This document situates skribe in the existing literature and gives proper attribution to prior work. It is intended to inform future academic writing, README positioning, and design discussions.

---

## Core idea

skribe treats the LLM as a **one-time code synthesizer**: at `fit()` the LLM reads a sample of training data and writes a standalone Python `predict(**features)` function. At inference, only that compiled function runs — no LLM API calls, no network, no per-row cost. The function is the complete predictor; nothing else runs on top of it.

---

## Closest prior work

### FeatLLM
Han et al., "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning", [arXiv 2404.09491](https://arxiv.org/abs/2404.09491), ICML 2024.

The closest single prior work to skribe's core idea. FeatLLM generates Python functions (`extracting_features_no(df_input)`) that apply if/elif-style lambda conditions to create binary feature columns. No LLM is called at inference — the generated functions run as a feature transformation step. Feature names are the semantic hook: the LLM sees column names like `Age`, `Sex` and writes threshold rules directly referencing them.

**Key difference:** FeatLLM generates *feature engineering functions* (transformations to binary columns), then trains a separate linear classifier on top of the engineered features. The generated code is not the entire predictor. Not sklearn-compatible (`BaseEstimator`). No context pre-pass, extend pass, error-feedback retry, SHAP explain, or joblib serialization.

### CAAFE
Hollmann et al., "Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering", [arXiv 2305.03403](https://arxiv.org/abs/2305.03403), NeurIPS 2023.

LLM iteratively generates Python code for new feature columns, guided by cross-validation feedback. The generated code is feature transformation code; a separate sklearn classifier (default: TabPFN) does the actual prediction and runs at every inference. Provides a `CAAFEClassifier` sklearn wrapper.

CAAFE was the first to explicitly identify the feature-name semantic gap that AutoML had missed — that LLMs can exploit column names to apply domain knowledge.

**Key difference:** CAAFE requires a downstream classifier at inference. The LLM is not called at inference but the sklearn model is. No standalone predict function; no context pre-pass; no code-string serialization.

### OCTree
Nam et al., "Optimized Feature Generation for Tabular Data via LLMs with Decision Tree Reasoning", [arXiv 2406.08527](https://arxiv.org/abs/2406.08527), NeurIPS 2024.

LLM generates new feature column rules guided by decision-tree reasoning feedback from prior iterations. Same paradigm as CAAFE: feature engineering code plus a separate tree-based downstream model.

### Talking Trees
Yandex Research, "Talking Trees: Reasoning-Assisted Induction of Decision Trees for Tabular Data", [arXiv 2509.21465](https://arxiv.org/abs/2509.21465), 2025.

An LLM agent constructs a decision tree at training time using a tool loop. Inference is LLM-free — just feature comparisons on the tree. Feature names and descriptions are given to the LLM, so it can apply prior knowledge when selecting splits. Sklearn tree conversion is supported.

**Key difference:** The artifact is a decision tree object, not a Python code string. Cannot represent arbitrary scoring logic, lookup tables, or multi-feature interactions the way a generated `predict()` function can. No context pre-pass, no extend pass, no SHAP explain, no regression support.

### "From Stochastic Answers to Verifiable Reasoning"
"From Stochastic Answers to Verifiable Reasoning: Interpretable Decision-Making with LLM-Generated Code", [arXiv 2603.13287](https://arxiv.org/abs/2603.13287), 2026.

LLM generates Python lambda expressions encoding binary classification rules over structured dictionaries, then feeds them into logistic regression. No LLM at inference; field names are the semantic hooks.

**Key difference:** Domain-specific (founder screening). The generated code is a set of lambda rules feeding into logistic regression, not a standalone predict function. Not sklearn-compatible. No fit/predict API, no context pre-pass, no serialization.

### SemPipes
Ovcharenko et al., "SemPipes: Optimizable Semantic Data Operators for Tabular Machine Learning Pipelines", [arXiv 2602.05134](https://arxiv.org/abs/2602.05134), 2026.

LLM synthesizes Python implementations of semantic data operators declared in natural language, guided by MCTS evolutionary search. Generates sklearn-pipeline-compatible operator implementations.

**Key difference:** Synthesizes data transformation operators within a search loop, not a standalone predict function. Requires many LLM calls during search. No world-knowledge-driven context pre-pass.

---

## LLM-as-direct-predictor (inference-time LLM — architecturally opposite)

All of the following call the LLM on every inference row. They are the dominant prior paradigm and represent the tradeoff that skribe specifically avoids.

- **LIFT** — Dinh, Zeng et al., "LIFT: Language-Interfaced Fine-Tuning for Non-language Machine Learning Tasks", [NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/4ce7fe1d2730f53cb3857032952cd1b8-Abstract-Conference.html). Fine-tunes LLM to predict tabular labels; LLM runs at inference.
- **TABLET** — Slack et al., "TABLET: Learning From Instructions For Tabular Data", [arXiv 2304.13188](https://arxiv.org/abs/2304.13188). LLM called per row with natural-language instructions.
- **TabLLM** — Hegselmann et al., "TabLLM: Few-shot Classification of Tabular Data with Large Language Models", [arXiv 2210.10723](https://arxiv.org/abs/2210.10723), AISTATS 2023. Row serialized to text; LLM called per row.
- **UniPredict** — "UniPredict: Large Language Models are Universal Tabular Classifiers", [arXiv 2310.03266](https://arxiv.org/abs/2310.03266). LLM maps rows to class probabilities via text generation.
- **Scikit-LLM** — [github.com/iryna-kondr/scikit-llm](https://github.com/iryna-kondr/scikit-llm), [PyPI](https://pypi.org/project/scikit-llm/). Sklearn `fit`/`predict` API wrapping GPT-4; every `.predict()` is an LLM API call.

### TabPFN
Hollmann et al., [arXiv 2207.01848](https://arxiv.org/abs/2207.01848) (ICLR 2023); v2: "Accurate predictions on small data with a tabular foundation model", [Nature 637:319–326, 2025](https://www.nature.com/articles/s41586-024-08328-6).

Transformer pre-trained on synthetic data; uses in-context learning at inference (the entire training set is passed as context for each test batch). No LLM call but requires running a neural network with O(n²) complexity in training rows. Not code synthesis. Not a compiled Python function. Strong baseline across small tabular datasets.

---

## Symbolic regression and program synthesis

- **PySR** — Cranmer, [github.com/MilesCranmer/PySR](https://github.com/MilesCranmer/PySR), 2023. Evolves symbolic mathematical expressions using genetic programming. Outputs sklearn-compatible `PySRRegressor`/`PySRClassifier`. Generates code (sympy expressions exportable to Python/C). No LLM; uses evolutionary search, not world knowledge. Cannot handle named categorical features or semantic domain knowledge.
- **TPOT** — [github.com/EpistasisLab/tpot](https://github.com/EpistasisLab/tpot). AutoML using genetic programming to optimize sklearn pipelines. Selects and chains sklearn primitives; no predict-logic code generation.
- **LaSR** — Grayeli et al., "Symbolic Regression with a Learned Concept Library", [arXiv 2409.09359](https://arxiv.org/abs/2409.09359), 2024. Combines LLM suggestions with PySR's evolutionary search for symbolic regression. Targets numeric regression formulas, not if/elif classification logic on named categorical features.

---

## What is novel in skribe

The following combination of properties has no prior art that bundles all of them:

**1. The LLM writes a complete standalone `predict(**features)` that is the entire model.**
FeatLLM and CAAFE come closest but both generate feature-engineering functions that feed a separate downstream classifier. The generated code in skribe *is* the predictor — nothing else runs on top of it.

**2. True zero-cost inference.**
The compiled function is pure Python: no imports beyond the standard library, no network, no GPU. Inference is microseconds per row regardless of training set size. FeatLLM eliminates the per-row LLM call but still runs a linear classifier; CAAFE runs a full sklearn model.

**3. Full sklearn `BaseEstimator` contract.**
`Pipeline`, `GridSearchCV`, `clone()`, and `joblib` serialization all work without modification. Scikit-LLM provides the same API surface but calls the LLM on every `.predict()`. CAAFE wraps a base classifier with sklearn compatibility but the synthesized code is not itself a `BaseEstimator`.

**4. Serialization via code string.**
`__getstate__`/`__setstate__` drop the compiled function object (not serializable) and recreate it from the `python_code_` string on load. Joblib files contain only the generated Python source — no model weights, no API keys, no runtime state.

**5. Multi-pass generation pipeline at fit time.**
Three sequential LLM calls: (a) context pre-pass summarizes the dataset and decodes opaque column names/values; (b) code generation with error-feedback retry loop (compile error or validation failure is fed back to the LLM for correction); (c) extend pass expands categorical lookup tables in the generated code. No prior system has this structured multi-pass pipeline.

**6. Web search at fit time.**
The LLM can query the web during `fit()` to look up domain schemas — ICD codes, airport codes, chess notation, NAICS codes — and build richer lookup tables in the generated function. No prior tabular ML system does this.

**7. `explain()` and `explain_comparison()`.**
SHAP KernelExplainer over the compiled predict function, with LLM-generated narrative. `explain_comparison()` runs contrastive SHAP importance across multiple fitted models (including non-skribe baselines) and generates a plain-English explanation of *why* they differ. This is novel as a post-hoc interpretability layer specifically designed for synthesized-code predictors.

---

## Summary table

| System | Standalone predict fn | No LLM at inference | sklearn BaseEstimator | Code-string serialization | Multi-pass fit | Web search at fit | explain_comparison |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **skribe** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| FeatLLM | — | ✓ | — | — | — | — | — |
| CAAFE | — | partial | ✓ (wrapper) | — | — | — | — |
| Talking Trees | — | ✓ | partial | — | — | — | — |
| Scikit-LLM | — | — | ✓ | — | — | — | — |
| PySR | ✓ | ✓ | ✓ | — | — | — | — |
| TabPFN | — | ✓ | ✓ | — | — | — | — |
| LIFT / TABLET / TabLLM | — | — | — | — | — | — | — |

---

## Citation notes for future use

When writing a paper or extended README, these are the works that should be cited as the most relevant prior art:

- FeatLLM ([arXiv 2404.09491](https://arxiv.org/abs/2404.09491)) — closest to the code-synthesis idea; credit for LLM-generated Python functions from tabular data
- CAAFE ([arXiv 2305.03403](https://arxiv.org/abs/2305.03403)) — credit for identifying the feature-name semantic gap in AutoML
- TabPFN ([arXiv 2207.01848](https://arxiv.org/abs/2207.01848)) — the strongest baseline on small tabular datasets; important to benchmark against
- Talking Trees ([arXiv 2509.21465](https://arxiv.org/abs/2509.21465)) — LLM-at-train-time, inference-free, but tree artifact not Python code
- Scikit-LLM ([GitHub](https://github.com/iryna-kondr/scikit-llm)) — the most prominent prior sklearn-API LLM wrapper; contrast case for inference cost
- PySR ([GitHub](https://github.com/MilesCranmer/PySR)) — sklearn-compatible code synthesis via evolution; contrast case for non-LLM program synthesis
