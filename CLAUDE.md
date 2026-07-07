# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dev dependencies
pip install -r requirements-dev.txt
pre-commit install

# Run tests
pytest                                          # full suite
pytest tests/test_classifier.py                # single file
pytest tests/test_classifier.py::test_fit_predict_dataframe  # single test
pytest --cov=skribe --cov-report=html     # with coverage

# Lint / format
black skribe/ tests/
ruff check skribe/ tests/
isort skribe/ tests/
mypy skribe/

# Build
python -m build
```

Pre-commit hooks enforce Black formatting and the full pytest suite on every commit. The pytest hook calls `.venv/bin/python -m pytest`, so the venv must exist.

Tests in `conftest.py` set `SKRIBE_MODEL=gpt-5.4-mini` so CI uses a cheap model. Tests that make live LLM calls require an API key (`OPENAI_API_KEY` by default).

## Architecture

**Core principle:** the LLM is a one-time code synthesizer, not an inference engine. At `fit()` the LLM writes a standalone Python `predict()` function once. At `predict()` only that compiled function runs — no API calls, no network, no per-row cost.

### Class hierarchy

```
BaseSkribeEstimator (base.py)          ← all LLM interaction + code generation lives here
├── SkribeClassifier (classifier.py)   ← thin wrapper, classification prompt template
├── SkribeRegressor (regressor.py)     ← thin wrapper, regression prompt template
└── SkribeFeatureEngineer              ← generates transform(**features) -> dict
    └── AdaptiveSkribeEngineer        ← skips FE if probe CV shows no improvement
```

`BaseSkribeEstimator` inherits `sklearn.base.BaseEstimator` so all scikit-learn protocols (Pipeline, GridSearchCV, clone, joblib) work automatically.

### Fit-time code generation pipeline (`base.py`)

1. **`_fit()`** — normalizes column names via `prepare_training_data()`, downsamples to `max_train_rows` (seed=42, deterministic), builds prompt from template + CSV sample.
2. **`_generate_code()`** — calls LLM, extracts Python from markdown fences (`extract_python_code()`), compiles and validates the function against pre-computed `validation_rows`. On failure, feeds the error back to the LLM and retries up to `max_retries` times.
3. **`_extend_code()`** — second LLM pass asking it to expand categorical lookup tables in the generated code.
4. **`make_predict_fn()`** (`utils.py`) — `exec()`s the final code string and extracts the first callable named `predict` or `transform`.

### Inference

Predictions call `safe_predict()` / `safe_regress()` (utils.py) which wrap the compiled function, coerce output types, and return a default (0 / 0.0) on any exception rather than raising.

`SkribeFeatureEngineer.transform()` validates that every row returns the same dict keys; inconsistency triggers a retry at fit time.

### LLM routing (`base._call_llm`)

Model selection: explicit `model=` arg > `SKRIBE_MODEL` env var > `DEFAULT_MODEL` (`"gpt-5.5"`).

All calls go through LiteLLM. Web search is available for GPT-5+ models (OpenAI Responses API) and Gemini Vertex AI models (grounding). Pass `web_search=True` to `fit()` to enable it.

### Serialization

`__getstate__` drops `predict_fn` (compiled function, not serializable). `__setstate__` re-execs `python_code_` to recreate it. Joblib files contain only the code string and metadata — no API keys embedded.

### Feature name normalization

`normalize_feature_name()` (utils.py) converts column names to valid Python identifiers (lowercase, non-alphanumeric → `_`). The LLM-generated code uses these normalized names. Original column names are never seen by the generated function.

### Key attributes set after `fit()`

| Attribute | Type | Contents |
|---|---|---|
| `python_code_` | `str` | The generated Python source |
| `predict_fn` | `callable` | Compiled function (not serialized) |
| `feature_names_` | `list[str]` | Normalized column names |
| `explanation_` | `Explanation` | Cached after first `.explain()` call |

### AdaptiveSkribeEngineer

Two-stage decision before any LLM call: (1) size guard — skip if `n_rows < min_rows` (default 200); (2) probe CV — fit FE on a stratified subset, compare logreg CV accuracy with/without engineered features. If `probe_delta_ <= min_delta` (default 0.0), returns `X` unchanged and sets `skip_reason_`.
