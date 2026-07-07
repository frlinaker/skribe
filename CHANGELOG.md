# Changelog

All notable changes to skribe are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.5.0] — 2026-06-23 — feature engineering & benchmarks
### Added
- `SkribeFeatureEngineer`: a scikit-learn `TransformerMixin` that uses the LLM
  to generate a standalone `transform()` function deriving new, world-knowledge-
  rich features from semantically meaningful columns. It validates/retries the
  generated code like the estimators, makes no per-row LLM calls, appends the
  engineered columns, and drops into a `Pipeline` before any classical model
- Benchmark harness (`benchmarks/run_openml_benchmark.py`) comparing skribe,
  skribe + feature engineering, logistic regression, and XGBoost across 10
  OpenML datasets, with a JSON results cache. Results published in the README:
  on `gpt-5.5`, `SkribeFeatureEngineer` + logistic regression reaches 0.937 mean
  accuracy — ahead of XGBoost (0.925) and plain logistic regression (0.878)

### Changed
- Extracted the code-generation + validation/retry loop into
  `BaseSkribeEstimator._generate_code`, now shared by the estimators and the
  feature engineer; `make_predict_fn` resolves a `predict` or `transform` entry
  point
- `compare_models` passes pre-built `Pipeline` instances through untouched, so a
  feature-engineering pipeline isn't re-wrapped in the one-hot encoder

---

## [0.4.1] — 2026-06-23 — packaging & test ergonomics
### Added
- `SKRIBE_MODEL` environment variable to override the default model
  without code changes (an explicit `model=` argument still wins). The test
  suite uses it to run the live-LLM gate against a fast, cheap model

### Changed
- Code-generation prompts now require valid, properly terminated string
  literals and instruct the model to wrap apostrophe-bearing keys (e.g.
  `"grevy's zebra"`) in double quotes, reducing first-attempt syntax failures

### Fixed
- Packaging no longer ships the `tests` (or `examples`) directory as an
  installable top-level package; only `skribe` is published
- Added a `MANIFEST.in` that keeps local-only and secret-bearing files
  (`.env`, `.envrc`, `.cursorrules`, `.claude/`) out of the source distribution

---

## [0.4.0] — 2026-06-23 — multi-provider, reliability & explainability
### Added
- Multi-provider LLM support via [litellm](https://github.com/BerriAI/litellm):
  the `model` string now routes to OpenAI (`gpt-5.5`), Anthropic
  (`claude-sonnet-4-6`), or local Ollama (`ollama:llama3.1`) (#1)
- Code validation with retry: `fit()` now runs the generated function over the
  training sample and, on failure, feeds the error back to the LLM and retries
  up to `max_retries` times (default 2) before raising (#2)
- `explain()` method returning a plain-English `Explanation` of the fitted
  heuristic. Bare `explain()` gives a cached, global explanation; `explain(X)`
  gives a local explanation of a single prediction. The `Explanation` object
  carries `meta`/`data` dicts (with attribute access) and is JSON
  round-trippable; `explain()` on an unfitted estimator raises
  `NotFittedError` (#3)

- `compare_models(models, X_train, y_train, X_test, y_test)` helper that fits
  any mix of skribe and sklearn/XGBoost estimators on one dataset and
  returns a side-by-side metrics table plus a row-by-row predictions table

### Changed
- Replaced the hardcoded OpenAI client with litellm; API keys are now resolved
  lazily per-provider from the usual environment variables, so constructing an
  estimator no longer requires `OPENAI_API_KEY`
- Consolidated all of `examples/` into a single guided tour,
  `examples/quickstart.py`, exposing every example as a demo behind a `--demo`
  selector (zero_row, sample, joblib, linear, nonlinear, xor, world_knowledge,
  multioutput, gridsearch, large_dataset, compare, titanic). This replaces the
  ~20 individual scripts, drops ones that were broken or relied on removed
  parameters / missing data files / an external benchmark corpus, and is built
  on the reusable `compare_models` helper for the side-by-side benchmark

### Fixed
- scikit-learn ≥1.6 compatibility: the estimators now inherit `BaseEstimator`
  (with `ClassifierMixin` / `RegressorMixin`), so they expose `__sklearn_tags__`
  and once again work inside meta-estimators such as `GridSearchCV` and
  `MultiOutputRegressor`, which previously raised `AttributeError`

---

## [0.3.1] — 2026-06-22 — benchmark polish
### Changed
- Benchmark output now highlights winning models per metric
- Benchmark code reformatted with black and redundancies removed
- Cleaned up unused functionality across estimators and benchmark runner

### Fixed
- Pylance type warnings in benchmark module

---

## [0.3.0] — 2025-07-12 — second-pass generalisation
### Added
- Second LLM pass that refines and extends the generated Python heuristic
- Riddle-style example demonstrating how the second pass improves edge cases
- Kaggle dataset example (added in 0.2.3, documented here)

---

## [0.2.3] — 2025-07-12 — housekeeping
### Added
- Kaggle dataset example

### Removed
- Unused dependencies trimmed from requirements

---

## [0.2.2] — 2025-07-10 — test coverage
### Added
- Full test suite covering classifier, regressor, and edge cases
- Test coverage for missing data, zero-row fitting, and boolean targets

---

## [0.2.1] — 2025-07-09 — refactor and sample
### Added
- Reintroduced `.sample(n)` method for generating synthetic rows from a fitted model

### Changed
- Classifier refactored for clarity and reduced duplication
- Common base logic simplified and consolidated
- Codebase reformatted with black

### Fixed
- Model benchmark code corrected

---

## [0.2.0] — 2025-07-08 — Python heuristics (major internal change)
### Changed
- Estimators now generate a **standalone Python function** as the heuristic instead of
  returning raw LLM text. Predictions run the generated function directly — no LLM call
  at predict time. This is the core architectural shift that makes inference fast and
  the model serializable.
- Regressor updated to output Python heuristics, benchmark fixed accordingly
- README updated to reflect Python heuristic approach

### Fixed
- Broken joblib test removed
- Classifier test corrected

---

## [0.1.0] — initial release
### Added
- `SkribeClassifier` — sklearn-compatible classifier backed by LLM reasoning
- `SkribeRegressor` — sklearn-compatible regressor backed by LLM reasoning
- Zero-row fitting: pass column names only, model infers from world knowledge
- `.sample(n)` method for generating synthetic rows from the fitted heuristic
- `joblib` serialization support (LLM client excluded, heuristic preserved)
- Pandas DataFrame support for both estimators
- Sliding window chunking for large datasets
- Benchmarks: mammal classification and falling-object regression vs traditional models
- Examples: Iris, Titanic, census, country flag colours, XOR, multioutput
- Type annotations throughout
- PyPI packaging
