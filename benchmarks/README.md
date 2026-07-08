# Benchmarks

Compares `SkribeClassifier` (across multiple LLMs) against `logreg`, `xgboost`,
and `tabpfn` baselines on 16 OpenML classification datasets.

## Setup

```bash
pip install -r requirements-benchmark.txt
```

Environment variables (only needed for the parts you actually run):

| Var | Needed for |
|---|---|
| `OPENAI_API_KEY` | `skribe` fits using a `gpt-*` model |
| `GOOGLE_APPLICATION_CREDENTIALS`, `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` | `skribe` fits using a `gemini-*` model |
| `TABPFN_TOKEN` | `tabpfn` baseline (hosted TabPFN API, not local inference) |

## Running everything in one go

```bash
benchmarks/run_all_models.sh
```

Runs baselines (logreg, xgboost, tabpfn) across all 16 datasets, then every
`(LLM model, dataset)` pair for `SkribeClassifier`, then `collate.py` to produce
summary tables and charts. Already-cached `(model, dataset)` results are
skipped automatically — safe to re-run or resume after an interruption.

Useful flags:

| Flag | Effect |
|---|---|
| `--workers=N` | Parallel workers for the LLM section (default 2) |
| `--no-cache` | Force re-run everything, ignoring existing cache |
| `--no-collate` | Skip the final `collate.py` step |
| `--skip-baselines` | Skip straight to the LLM section |
| `--baselines-only` | Run only baselines, then exit (implies `--no-collate`) |

## Running as two blocks in sequence

```bash
# Block 1: baselines only
benchmarks/run_all_models.sh --baselines-only

# Block 2: LLM variants only (baselines already cached, so this is fast to reach)
benchmarks/run_all_models.sh --skip-baselines --workers=4
```

## How the LLM section is scheduled

Every `(model, dataset)` pair across **all** LLM models (OpenAI + Google, `+web`
variants excluded) is flattened into a single work queue, ordered
dataset-outer / model-inner with datasets sorted smallest-to-largest by row
count. `--workers` workers pull whatever pair is next off that queue — no
model or provider has to fully finish before the next one starts, so a slow
straggler on one model never leaves other workers idle. Smallest-dataset-first
means you see real results within the first minute or two of a fresh run.

## Running one `(model, dataset)` pair directly

```bash
# Baseline
.venv/bin/python benchmarks/run_openml_fit.py --model xgboost --dataset credit-g

# Skribe with a specific LLM
.venv/bin/python benchmarks/run_openml_fit.py --model skribe --llm gpt-5.5 --dataset adult

# List valid --llm values
.venv/bin/python benchmarks/run_openml_fit.py --model skribe --dataset zoo --list-models
```

Each invocation fits exactly one model on one dataset and writes a single
cache JSON to `artifacts/benchmark_results/cache/`. `--no-cache` forces a
re-fit even if a cache file already exists.

## Monitoring a running benchmark

```bash
benchmarks/monitor.sh
```

Polls `artifacts/benchmark_results/cache/` and `artifacts/benchmark_results/run_llm.log`
every 30s and prints cache-file counts per model plus the most recent log lines.

## Collating results

```bash
.venv/bin/python benchmarks/collate.py
```

Reads every JSON in `artifacts/benchmark_results/cache/`, prints a model ×
dataset accuracy table, and writes charts. Runs automatically at the end of
`run_all_models.sh` unless `--no-collate` was passed. Supports `--datasets`
and `--llms` to filter, and `--output-dir` to point at a different cache
location.

## Adding a dataset or model

Datasets, LLM models, and baseline learners are all defined once in
`benchmarks/config.yaml` — every script reads from there via
`benchmark_utils.py` (`DEFAULT_DATASETS`, `MODEL_PROGRESSION`,
`BASELINE_MODELS`/`BASELINE_META`), so there's nothing to update elsewhere.

- **Dataset**: add an entry under `datasets:` — `openml_name` + `version` for
  an OpenML dataset, or `csv_path` (relative to `benchmarks/`) + `target_col`
  + `description` for a CSV-backed one.
- **Model**: add an entry under `models:` — only base (non-`+web`) models are
  listed; a `+web` sibling is generated automatically for any entry with
  `supports_web: true` (label suffixed " +web", color lightened unless
  `web_color` is given explicitly).
- **Baseline learner**: add an entry under `baselines:` with `name`, `label`,
  `color`. Also requires a corresponding classifier factory in
  `benchmark_utils.py` and a case in `run_openml_fit.py`'s baseline dispatch.

## Other scripts

- `benchmark_utils.py` — shared dataset loaders, `DEFAULT_DATASETS`,
  `MODEL_PROGRESSION`, plotting utilities. Not run directly.
- `build_skribe_inspector.py` — generates a self-contained HTML page for
  browsing generated prompts/code per `(dataset, model)` from cache files.
- `run_adaptive_fe_benchmark.py` — separate benchmark for
  `AdaptiveSkribeEngineer` (feature engineering), independent of the
  classifier benchmark above.
- `test_large_prompt.py` — stress test for very large fit prompts against
  OpenAI/Gemini.
