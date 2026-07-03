#!/usr/bin/env python
"""Collate cached benchmark results and produce summary tables and charts.

Reads all JSON files from --output-dir/cache/ and merges them into a unified
DataFrame.  Handles two file naming conventions:

  promptlearn files : ``{dataset}-{model_id}-{hash}.json``
      Must contain ``dataset`` and ``model_id`` keys at the top level.

  baseline files    : ``baselines-{dataset}-{hash}.json``
      Contain learner-name keys (``logreg``, ``xgboost``, ``tabpfn``) but no
      ``dataset`` / ``model_id`` keys.  The dataset name is inferred from the
      file name (second dash-separated segment).

Both file types are merged via ``build_summary_df()`` and passed to
``print_summary_table()`` (model × dataset accuracy grid) and
``plot_progression()`` (timeline / heatmap / bar charts).

Examples
--------
    # collate everything in the default output dir
    python benchmarks/collate.py

    # filter to a subset of datasets and/or LLM model IDs
    python benchmarks/collate.py --datasets adult credit-g --llms gpt-5.5 gpt-5.5+web

    # write charts to a different directory
    python benchmarks/collate.py --output-dir /tmp/bench_out
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Allow running from repo root: python benchmarks/collate.py
sys.path.insert(0, os.path.dirname(__file__))

from benchmark_utils import (
    DEFAULT_DATASETS,
    MODEL_PROGRESSION,
    build_summary_df,
    plot_progression,
    print_summary_table,
)

logger = logging.getLogger("promptlearn.progression")

_BASELINE_RE = re.compile(r"^baselines-(.+)-[0-9a-f]{16}\.json$")
_PROMPTLEARN_RE = re.compile(r"^(.+?)-(.+)-[0-9a-f]{16}\.json$")


def load_cache_results(
    cache_dir: Path,
    dataset_filter: list[str] | None = None,
    llm_filter: list[str] | None = None,
) -> list[dict]:
    """Read all JSON cache files from *cache_dir* and return a flat list of dicts.

    Baseline files are augmented with a ``"dataset"`` key inferred from the
    file name so that ``build_summary_df()`` can process them uniformly.

    Parameters
    ----------
    cache_dir:
        Directory containing the JSON cache files (``cache/`` subdirectory of
        the output dir).
    dataset_filter:
        When given, only include records whose dataset name is in this list.
    llm_filter:
        When given, only include promptlearn records whose ``model_id`` is in
        this list.  Baseline records are always included (they are
        model-independent).
    """
    if not cache_dir.exists():
        logger.warning("Cache directory does not exist: %s", cache_dir)
        return []

    results: list[dict] = []

    for path in sorted(cache_dir.glob("*.json")):
        fname = path.name

        # ── baseline file ────────────────────────────────────────────────────
        m = _BASELINE_RE.match(fname)
        if m:
            dataset_name = m.group(1)
            if dataset_filter and dataset_name not in dataset_filter:
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                # Inject dataset so build_summary_df can emit baseline rows.
                data["dataset"] = dataset_name
                results.append(data)
            except Exception as e:
                logger.warning("Failed to read %s: %s", path, e)
            continue

        # ── promptlearn file ─────────────────────────────────────────────────
        # Skip files that start with "baselines-" (already handled above) and
        # any aggregation files like metrics_all.json.
        if fname == "metrics_all.json" or fname.startswith("baselines-"):
            continue

        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
            continue

        # Promptlearn files contain dataset / model_id at the top level.
        dataset_name = data.get("dataset")
        model_id = data.get("model_id")

        if not dataset_name or not model_id:
            logger.debug("Skipping %s — missing dataset or model_id key", fname)
            continue

        if dataset_filter and dataset_name not in dataset_filter:
            continue
        if llm_filter and model_id not in llm_filter:
            continue

        results.append(data)

    logger.info(
        "Loaded %d cache records from %s", len(results), cache_dir
    )
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
            "LLM model IDs to include in the promptlearn rows "
            "(default: all available in cache).  Baselines are always included."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip chart generation (print table only).",
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
    )

    if not results:
        print(
            f"No cache files found in {cache_dir}.  "
            "Run run_baselines.py and/or run_promptlearn.py first."
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
