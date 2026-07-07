#!/usr/bin/env python
"""Run skribe on the curated OpenML suite for one LLM model.

Results are written to --output-dir/cache/ as JSON files named
``{dataset}-{model_id}-{hash}.json``.  These files are later read by
collate.py to build the summary table and charts.

This script does NOT run any baselines.  Use run_baselines.py for those.

Examples
--------
    # run GPT-5.5 on the full suite
    python benchmarks/run_skribe.py --llm gpt-5.5

    # run GPT-5.5 with web search enabled on two datasets
    python benchmarks/run_skribe.py --llm gpt-5.5+web --datasets adult credit-g

    # use AdaptiveSkribeEngineer with a different model
    python benchmarks/run_skribe.py --llm gpt-5.5 --fe-model gpt-5.4-mini

    # see all available model IDs
    python benchmarks/run_skribe.py --list-models
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# benchmark_utils must be importable; add benchmarks/ to sys.path when running
# from repo root so `import benchmark_utils` works without installing the package.
sys.path.insert(0, os.path.dirname(__file__))

from benchmark_utils import (
    DEFAULT_DATASETS,
    MODEL_PROGRESSION,
    _cache_key,
    _rich_metrics,
    load_dataset,
)

import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger("skribe.progression")

_MODEL_LOOKUP = {m["model_id"]: m for m in MODEL_PROGRESSION}


_print_lock = threading.Lock()


def _print(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def run_dataset_model(
    dataset: str,
    spec: tuple,
    model_id: str,
    max_rows: int | None,
    cache_dir: Path | None,
    vertex_region: str | None = None,
    fe_model: str | None = None,
    web_search: bool = False,
    base_model_id: str | None = None,
    skip_cache_read: bool = False,
    skip_context: bool = False,
) -> dict:
    """Run skribe on one (dataset, model) cell.

    Parameters
    ----------
    dataset:
        Short name used in cache keys and log messages.
    spec:
        ``(openml_name, version)`` tuple from DEFAULT_DATASETS.
    model_id:
        The canonical model key from MODEL_PROGRESSION (e.g. ``"gpt-5.5+web"``).
    max_rows:
        Cap on training+test rows combined (sampled deterministically).
    cache_dir:
        Directory for JSON cache files.  Pass ``None`` to disable all caching.
    vertex_region:
        Overrides ``VERTEXAI_LOCATION`` env var for this call (restored afterwards).
    fe_model:
        LLM model id for AdaptiveSkribeEngineer.  ``None`` disables FE.
    web_search:
        Pass ``web_search=True`` to SkribeClassifier.fit().
    base_model_id:
        Actual LLM model id when ``model_id`` is a synthetic key like
        ``"gpt-5.5+web"``.  Stripped from model_id when ``None`` and
        ``web_search`` is True.
    skip_cache_read:
        When True, ignore any existing cache file and re-run, but still write
        the new result to cache.  Used by ``--no-cache``.

    Returns
    -------
    dict
        Metrics dict with keys ``dataset``, ``model_id``, ``skribe``, etc.
    """
    # For web-search variants the model_id is a synthetic key (e.g. "gpt-5.5+web");
    # the actual LLM call uses base_model_id when provided, else strips the +web suffix.
    actual_model_id = base_model_id or (
        model_id.removesuffix("+web") if web_search else model_id
    )

    tag = f"[{dataset} × {model_id}]"

    safe_model_id = model_id.replace("/", "-")
    cache_file = (
        cache_dir
        / f"{dataset}-{safe_model_id}-{_cache_key(dataset, model_id, max_rows, fe_model=fe_model, web_search=web_search)}.json"
        if cache_dir
        else None
    )
    if cache_file and cache_file.exists() and not skip_cache_read:
        _print(f"{tag} cached — skipping")
        with open(cache_file) as f:
            return json.load(f)

    openml_name, version = spec[0], spec[1]
    csv_path = spec[2] if len(spec) > 2 else None
    target_col = spec[3] if len(spec) > 3 else None
    spec_description = spec[4] if len(spec) > 4 else None
    _print(f"{tag} loading dataset…")
    X, y, class_map, description = load_dataset(
        openml_name, version, max_rows,
        csv_path=csv_path, target_col=target_col, description=spec_description,
        require_description=not skip_context,
    )
    n_classes = len(class_map)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    _print(f"{tag} {len(X)} rows  {X.shape[1]} cols  {n_classes} classes")

    result = {
        "dataset": dataset,
        "model_id": model_id,
        "n_rows": len(X),
        "n_cols": X.shape[1],
        "n_classes": n_classes,
        "class_map": class_map,
    }

    # Lazy import — only run_skribe.py touches skribe.
    from skribe import AdaptiveSkribeEngineer, SkribeClassifier

    # Optional feature engineering pass before the classifier.
    if fe_model:
        _print(f"{tag} running AdaptiveSkribeEngineer ({fe_model})…")
        try:
            fe_step = AdaptiveSkribeEngineer(model=fe_model, verbose=False)
            X_train = fe_step.fit_transform(X_train, y_train)
            X_test = fe_step.transform(X_test)
            skip = getattr(fe_step, "skip_reason_", None)
            if skip:
                _print(f"{tag} AdaptiveFE skipped: {skip}")
            else:
                _print(
                    f"{tag} AdaptiveFE done — {X_train.shape[1]} cols  "
                    f"delta={getattr(fe_step, 'probe_delta_', float('nan')):.3f}"
                )
        except Exception as e:
            _print(f"{tag} AdaptiveFE FAILED: {e}  (using original features)")

    # ── skribe ───────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        clf = SkribeClassifier(
            model=actual_model_id,
            verbose=False,
            web_search=web_search,
            vertex_location=vertex_region or None,
            context_prepass=not skip_context,
        )

        # Patch _call_llm to print each LLM sub-step and accumulate per-stage timing.
        _prepass_time = [0.0]
        _fit_llm_time = [0.0]  # code-gen + extend + retries
        _codegen_count = [0]
        _orig_call_llm = clf._call_llm  # bound method

        def _instrumented_call_llm(prompt: str, web_search: bool = False) -> str:
            is_prepass = "preparing a structured dataset summary" in prompt
            is_extend = "extend any such mappings" in prompt
            if is_prepass:
                step = "context pre-pass"
            elif is_extend:
                step = "extend pass"
            else:
                _codegen_count[0] += 1
                step = "code generation" if _codegen_count[0] == 1 else f"retry #{_codegen_count[0] - 1}"
            ws_note = " [+web]" if web_search else ""
            _print(f"{tag}   → {step}{ws_note}…")
            t_llm = time.time()
            result_text = _orig_call_llm(prompt, web_search=web_search)
            dt = time.time() - t_llm
            if is_prepass:
                _prepass_time[0] += dt
            else:
                _fit_llm_time[0] += dt
            _print(f"{tag}   ✓ {step} done  ({len(result_text):,} chars  {dt:.1f}s)")
            return result_text

        clf._call_llm = _instrumented_call_llm

        _print(f"{tag} fitting…")
        t_fit = time.time()
        clf.fit(X_train, y_train, dataset_description=description or None)
        fit_elapsed = time.time() - t_fit
        _print(f"{tag} fit done  ({fit_elapsed:.1f}s)  code={len(clf.raw_python_code_ or ''):,} chars")

        t_predict = time.time()
        y_pred = clf.predict(X_test)
        predict_elapsed = time.time() - t_predict
        y_proba = None
        if hasattr(clf, "predict_proba"):
            try:
                y_proba = clf.predict_proba(X_test)
            except Exception:
                pass
        result["skribe"] = _rich_metrics(
            np.array(y_test), y_pred, y_proba, n_classes
        )
        result["skribe"]["fit_time_s"] = round(fit_elapsed, 2)
        result["skribe"]["prepass_time_s"] = round(_prepass_time[0], 2)
        result["skribe"]["predict_time_s"] = round(predict_elapsed, 4)
        result["skribe"]["generated_code"] = clf.raw_python_code_
        result["skribe"]["fit_prompt"] = getattr(clf, "fit_prompt_", None)
        result["skribe"]["context_prepass_prompt"] = getattr(clf, "context_prepass_prompt_", None)
        result["skribe"]["context_summary"] = getattr(clf, "context_summary_", None)
        acc = result["skribe"]["accuracy"]
        _print(f"{tag} accuracy={acc:.3f}  fit={fit_elapsed:.1f}s  predict={predict_elapsed:.4f}s  ✓")
    except Exception as e:
        elapsed = time.time() - t0
        _print(f"{tag} FAILED after {elapsed:.1f}s: {e}")
        result["skribe"] = {"error": str(e)}

    # Only cache successful results — errors are not cached so re-running the
    # script automatically retries failed datasets without manual cache cleanup.
    pl = result.get("skribe", {})
    if cache_file and not (isinstance(pl, dict) and pl.get("error")):
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        _print(f"{tag} cached → {cache_file.name}")

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm",
        metavar="MODEL_ID",
        help=(
            "LLM model_id from MODEL_PROGRESSION to run "
            "(e.g. gpt-5.5, gpt-5.5+web, vertex_ai/gemini-2.5-pro).  "
            "Use --list-models to see all valid values."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print all valid --llm values and exit.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Dataset keys to run (default: full suite).",
    )
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap dataset rows before split (default: no cap — use full dataset).")
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmark_results",
        help="Directory for cached results (default: artifacts/benchmark_results).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing cache files and re-run, but still write results to cache.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel dataset workers (default: 4). Each makes its own LLM calls.",
    )
    parser.add_argument(
        "--skip-context",
        action="store_true",
        help=(
            "Disable the dataset context pre-pass. "
            "WARNING: this produces lower-quality results and should only be used "
            "for debugging. Datasets with no description will fail without this flag."
        ),
    )
    parser.add_argument(
        "--fe-model",
        default=None,
        metavar="MODEL_ID",
        help=(
            "LLM to use for AdaptiveSkribeEngineer (e.g. gpt-5.5). "
            "Applied before SkribeClassifier.  Omit to disable FE."
        ),
    )
    args = parser.parse_args(argv)

    if args.list_models:
        print("Valid --llm values (MODEL_PROGRESSION):")
        for m in MODEL_PROGRESSION:
            ws = " [+web]" if m.get("web_search") else ""
            print(f"  {m['model_id']:<45}  {m['label']}{ws}")
        return 0

    if not args.llm:
        parser.error("--llm is required (use --list-models to see valid values)")

    if args.llm not in _MODEL_LOOKUP:
        valid = [m["model_id"] for m in MODEL_PROGRESSION]
        parser.error(
            f"Unknown --llm value {args.llm!r}.\n"
            f"Valid model IDs: {valid}\n"
            f"Use --list-models to see all options."
        )

    # Route our own logger to stdout; suppress skribe's internal logger
    # (key events are printed directly via _print instead).
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logging.basicConfig(handlers=[handler], level=logging.INFO)
    logging.getLogger("skribe").setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"

    meta = _MODEL_LOOKUP[args.llm]
    vertex_region = meta.get("vertex_region")
    web_search = meta.get("web_search", False)
    base_model_id = meta.get("base_model_id")
    label = meta.get("label", args.llm)

    unknown_datasets = [d for d in args.datasets if d not in DEFAULT_DATASETS]
    if unknown_datasets:
        logger.warning("Unknown datasets (not in DEFAULT_DATASETS): %s", unknown_datasets)
    datasets_to_run = [d for d in args.datasets if d in DEFAULT_DATASETS]

    if not datasets_to_run:
        print("No valid datasets to run.")
        return 1

    print(
        f"Running skribe  model={label!r}  datasets={datasets_to_run}  "
        f"max_rows={args.max_rows}  fe_model={args.fe_model or 'none'}  "
        f"cache={cache_dir}{'  (skip-read)' if args.no_cache else ''}",
        flush=True,
    )

    n_total = len(datasets_to_run)
    n_workers = min(args.workers, n_total)
    _print(f"Parallelism: {n_workers} workers  ({n_total} datasets)")

    results = []
    completed = [0]  # mutable for closure

    def _run_one(dataset: str) -> dict:
        spec = DEFAULT_DATASETS[dataset]
        idx = datasets_to_run.index(dataset) + 1
        _print(f"\n── {idx}/{n_total}: {dataset} starting ──")
        return run_dataset_model(
            dataset,
            spec,
            args.llm,
            args.max_rows,
            cache_dir,
            vertex_region=vertex_region,
            fe_model=args.fe_model,
            web_search=web_search,
            base_model_id=base_model_id,
            skip_cache_read=args.no_cache,
            skip_context=args.skip_context,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, ds): ds for ds in datasets_to_run}
        for future in as_completed(futures):
            dataset = futures[future]
            completed[0] += 1
            try:
                r = future.result()
                results.append(r)
                pl = r.get("skribe", {})
                if pl.get("error"):
                    _print(
                        f"\n✗ {dataset} FAILED ({completed[0]}/{n_total} done): {pl['error']}"
                    )
                else:
                    _print(
                        f"\n✓ {dataset} accuracy={pl.get('accuracy', float('nan')):.3f}"
                        f"  fit={pl.get('fit_time_s', '?')}s"
                        f"  ({completed[0]}/{n_total} done)"
                    )
            except Exception as e:
                _print(f"\n✗ {dataset} FATAL ({completed[0]}/{n_total} done): {e}")

    succeeded = [r for r in results if not r.get("skribe", {}).get("error")]
    failed = [r for r in results if r.get("skribe", {}).get("error")]
    _print(f"\n{'─'*60}")
    _print(f"Done: {len(succeeded)}/{n_total} succeeded  {len(failed)} failed  model={label!r}")
    if succeeded:
        accs = [r["skribe"]["accuracy"] for r in succeeded]
        _print(f"Accuracy — mean={sum(accs)/len(accs):.3f}  min={min(accs):.3f}  max={max(accs):.3f}")
    if failed:
        _print("Failed datasets:")
        for r in failed:
            _print(f"  {r['dataset']}: {r['skribe']['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
