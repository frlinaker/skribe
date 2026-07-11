# Changelog

All notable changes to skribe are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] — 2026-07-11 — reliability, reasoning controls & fit-time diagnostics
### Added
- `llm_timeout`, `reasoning_effort`, and `reasoning_mode` constructor params on
  the estimators, threaded through to the underlying litellm/Responses API
  calls (`reasoning_mode` supports OpenAI's "pro" tier; `reasoning_effort`
  supports `"max"` via the Responses API)
- `fit_log_`: captures fit-time retry/error history and, when `web_search=True`,
  the search evidence (citations, search-call counts) used while generating code
- Static AST check for unresolved names in generated code, with "did you mean"
  suggestions and argument runtime types surfaced in fit-time retry feedback
- Automatic retry on `litellm.RateLimitError` instead of sleeping then failing

### Changed
- The classifier's fallback/default prediction now states the true majority
  class (regressor: median) in the prompt instead of always defaulting to
  code `0` — the class codes aren't ordered by frequency, so a generic
  "default to 0" instruction could steer the LLM toward a fallback that
  fires on every unmatched input and dominates accuracy even when the
  matched-branch logic is correct
- Context pre-pass no longer fabricates column/value semantics or class-label
  mappings it wasn't given; ambiguous label mappings and uncapped
  high-cardinality columns are now handled explicitly
- `gpt-5.4-mini` context-window truncation now uses the API-reported limit
  instead of a hardcoded estimate
- Entire codebase reformatted with black/isort plus safe ruff auto-fixes

### Fixed
- `safe_exec_fn` no longer corrupts free-text feature values that happen to
  look numeric

---

## [0.1.0] — 2026-07-06 — rename to skribe (PyPI v0.1.0)
### Changed
- Package renamed from `promptlearn` to `skribe`. All public classes renamed:
  `SkribeClassifier`, `SkribeRegressor`, `SkribeFeatureEngineer`,
  `AdaptiveSkribeEngineer`, `BaseSkribeEstimator`
- Environment variable renamed: `SKRIBE_MODEL` (was `PROMPTLEARN_MODEL`)
- CLI entry point renamed: `skribe` (was `promptlearn`)
- Package metadata migrated from `setup.py` to `pyproject.toml` (PEP 621)
- Version reset to `0.1.0` for fresh PyPI project at pypi.org/project/skribe/
- A final `promptlearn` 0.6.0 shim published on PyPI that depends on `skribe`
  and redirects existing users

### Added
- `explain_comparison()`: contrastive SHAP-based feature importance across
  multiple fitted models (including non-skribe baselines) with LLM-generated
  plain-English narrative of why the models differ
- Web search at fit time: pass `web_search=True` to `fit()` to let the LLM
  query the web for domain schemas (ICD codes, airport codes, NAICS codes, etc.)
  when building lookup tables. Supported on GPT-5+ and Gemini Vertex AI models
- `AdaptiveSkribeEngineer`: two-stage guard before any LLM call — size check
  then probe CV — skips feature engineering if it doesn't improve accuracy
- Benchmark suite expanded to 16 datasets (13 OpenML + spotify-genre,
  heart-statlog, zoo); results cached per model/dataset with progression tracking
- `docs/related_work.md`: prior art survey covering FeatLLM, CAAFE, OCTree,
  Talking Trees, Scikit-LLM, TabPFN, PySR and seven novelty claims for skribe

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
