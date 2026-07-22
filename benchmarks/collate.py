#!/usr/bin/env python
"""Collate cached benchmark results and produce summary tables and charts.

Reads all JSON files from --output-dir/cache/ and merges them into a unified
DataFrame.  All cache files share the same format:

  ``{dataset}-{model_id}-{hash}.json``

Each file must contain ``dataset`` and ``model_id`` keys at the top level.
Baseline files (logreg / xgboost / tabpfn) set ``model_id`` to the learner
name and store metrics under that same key.  Skribe files set ``model_id`` to
the LLM model ID and store metrics under the ``"skribe"`` key.

Results are merged via ``build_summary_df()`` and passed to
``print_summary_table()`` (model × dataset accuracy grid) and
``plot_progression()`` (timeline / heatmap / bar charts).

Examples
--------
    # collate everything in the default output dir
    python benchmarks/collate.py

    # filter to a subset of datasets and/or LLM model IDs
    python benchmarks/collate.py --datasets adult credit-g --llms gpt-5.5 gpt-5.5-web

    # write charts to a different directory
    python benchmarks/collate.py --output-dir /tmp/bench_out
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Allow running from repo root: python benchmarks/collate.py
sys.path.insert(0, os.path.dirname(__file__))

from benchmark_utils import (
    BASELINE_MODELS,
    build_summary_df,
    plot_progression,
    print_summary_table,
)

logger = logging.getLogger("skribe.progression")


def load_cache_results(
    cache_dir: Path,
    dataset_filter: list[str] | None = None,
    llm_filter: list[str] | None = None,
    exclude_web: bool = False,
) -> list[dict]:
    """Read all JSON cache files from *cache_dir* and return a flat list of dicts.

    Parameters
    ----------
    cache_dir:
        Directory containing the JSON cache files (``cache/`` subdirectory of
        the output dir).
    dataset_filter:
        When given, only include records whose dataset name is in this list.
    llm_filter:
        When given, only include skribe records whose ``model_id`` is in this
        list.  Baseline records (logreg / xgboost / tabpfn) are always included.
    exclude_web:
        When true, skip skribe records whose ``model_id`` is a web-search
        variant (``<base>-web``, per ``_build_model_progression``'s naming
        convention — the cache file itself carries no separate web_search
        flag to check). Baseline records are always included.
    """
    if not cache_dir.exists():
        logger.warning("Cache directory does not exist: %s", cache_dir)
        return []

    results: list[dict] = []

    for path in sorted(cache_dir.glob("*.json")):
        fname = path.name
        if fname == "metrics_all.json":
            continue

        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
            continue

        dataset_name = data.get("dataset")
        model_id = data.get("model_id")

        if not dataset_name or not model_id:
            logger.debug("Skipping %s — missing dataset or model_id key", fname)
            continue

        if dataset_filter and dataset_name not in dataset_filter:
            continue

        is_baseline = model_id in BASELINE_MODELS
        if not is_baseline and llm_filter and model_id not in llm_filter:
            continue
        if not is_baseline and exclude_web and model_id.endswith("-web"):
            continue

        results.append(data)

    logger.info("Loaded %d cache records from %s", len(results), cache_dir)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmark_results",
        help="Directory containing the cache/ subdirectory and where charts are saved.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset keys to include (default: all available in cache).",
    )
    parser.add_argument(
        "--llms",
        nargs="*",
        default=None,
        metavar="MODEL_ID",
        help=(
            "LLM model IDs to include in the skribe rows "
            "(default: all available in cache).  Baselines are always included."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip chart generation (print table only).",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Exclude web-search-enabled (+web) skribe variants from the table and charts.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logger.setLevel(logging.INFO)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"

    results = load_cache_results(
        cache_dir,
        dataset_filter=args.datasets,
        llm_filter=args.llms,
        exclude_web=args.no_web,
    )

    if not results:
        print(
            f"No cache files found in {cache_dir}.  "
            "Run run_openml_fit.py (or run_all_models.sh for the full suite) first."
        )
        return 1

    df = build_summary_df(results)

    if df.empty:
        print("No usable rows in the loaded cache files.")
        return 1

    print_summary_table(df)

    if not args.no_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            plot_progression(df, output_dir)
            print(f"\nCharts saved to {output_dir}/", flush=True)
        except Exception as e:
            logger.warning("Chart generation failed: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
