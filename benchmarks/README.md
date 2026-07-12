# Benchmarks

Compares `SkribeClassifier` (across multiple LLMs) against `logreg`, `xgboost`,
and `tabpfn` baselines on 16 OpenML classification datasets.

## Setup

```bash
pip install -e .
pip install -r requirements-benchmark.txt
```

`requirements-benchmark.txt` only pulls in skribe's runtime dependencies (via
`-r requirements.txt`) plus benchmark-only libraries — it does not install
the `skribe` package itself, so `pip install -e .` must come first or
`run_openml_fit.py --model skribe ...` fails with `ModuleNotFoundError`.

14 of the 16 datasets are fetched automatically from OpenML at run time (via
`sklearn.datasets.fetch_openml`, which caches to disk after the first fetch —
no setup needed). The 15th, `spotify-genre`, is `csv_path`-backed
(`examples/external_data/spotify_genre.csv`) and is **not** included in the
repo — `examples/external_data/` is gitignored, so a fresh clone is missing
the file and any run touching `spotify-genre` (including the default
`run_all_models.sh`) fails with `FileNotFoundError`. Reproduce it before
running the benchmarks:

```bash
mkdir -p examples/external_data
curl -sL -o /tmp/spotify_songs_raw.csv \
  "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
```

```python
import pandas as pd

FEATURE_COLS = [
    "track_name", "track_artist", "track_popularity",
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms",
]

# The TidyTuesday CSV has one row per (track, playlist) pairing, so the same
# track appears once per playlist it's on -- with track_popularity and the
# audio features identical across its duplicates, but playlist_genre/
# playlist_name/etc. varying. config.yaml's dataset description promises
# "each row is a unique track", so this has to collapse back down to one row
# per track before it's usable as a classification dataset.
raw = pd.read_csv("/tmp/spotify_songs_raw.csv").rename(columns={"playlist_genre": "genre"})

# A handful of rows have nulls in feature columns (bad API scrapes upstream);
# drop them before dedup so a null-valued duplicate can't survive by being
# picked as the "first" occurrence of a (name, artist) pair.
raw = raw.dropna(subset=FEATURE_COLS + ["genre"])

# keep="first" reproduces examples/external_data/spotify_genre.csv exactly
# (verified row-for-row) -- there's nothing principled about "first" over
# "last", it's just whichever occurrence the original file happened to keep.
deduped = raw.drop_duplicates(subset=["track_name", "track_artist"], keep="first")

deduped[FEATURE_COLS + ["genre"]].to_csv(
    "examples/external_data/spotify_genre.csv", index=False
)
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

## AdaptiveSkribeEngineer (feature engineering) benchmark

```bash
benchmarks/run_afe_benchmark.sh
```

For every dataset, fits `logreg` and `xgboost` both with and without
`AdaptiveSkribeEngineer` applied first (via `run_openml_fit.py --fe-model`),
then runs `plot_afe_lift.py` to print a per-dataset delta table and save a
lift chart. Same cache as everything else, so it's safe to re-run or resume.
Defaults to the latest base OpenAI model in `config.yaml`; override with
`--fe-model=gpt-5.5`.

## Other scripts

- `benchmark_utils.py` — shared dataset loaders, `DEFAULT_DATASETS`,
  `MODEL_PROGRESSION`, plotting utilities. Not run directly.
- `build_skribe_inspector.py` — generates a self-contained HTML page for
  browsing generated prompts/code per `(dataset, model)` from cache files.
- `plot_afe_lift.py` — reporting step for the AFE benchmark above; reads
  cache files, doesn't fit anything itself.
- `test_large_prompt.py` — stress test for very large fit prompts against
  OpenAI/Gemini.
