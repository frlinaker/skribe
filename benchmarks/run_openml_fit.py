#!/usr/bin/env python
"""Fit one model on one dataset and write a single cache file.

Produces one JSON file under --output-dir/cache/.  The shell orchestrator
(run_all_models.sh) calls this script once per (model, dataset) combination.

--model choices
---------------
  logreg   LogisticRegression (one-hot + standard scaling)
  xgboost  XGBClassifier      (ordinal encoding)
  tabpfn   TabPFNClassifier   (ordinal encoding)
  skribe   SkribeClassifier   (requires --llm)

Examples
--------
    python benchmarks/run_openml_fit.py --model logreg --dataset adult
    python benchmarks/run_openml_fit.py --model xgboost --dataset credit-g
    python benchmarks/run_openml_fit.py --model skribe --llm gpt-5.5 --dataset adult
    python benchmarks/run_openml_fit.py --model skribe --list-models
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from benchmark_utils import (
    BASELINE_META,
    DEFAULT_DATASETS,
    MODEL_PROGRESSION,
    _baseline_cache_key,
    _cache_key,
    _rich_metrics,
    _tabpfn_classifier,
    _xgb_classifier,
    load_dataset,
)

logger = logging.getLogger("skribe.benchmark")

BASELINE_MODELS = list(BASELINE_META)
ALL_MODELS = BASELINE_MODELS + ["skribe"]
_MODEL_LOOKUP = {m["model_id"]: m for m in MODEL_PROGRESSION}


def _build_baseline_pipeline(model: str, X_train):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    transformers = []

    if model == "logreg":
        if cat_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    cat_cols,
                )
            )
        if num_cols:
            transformers.append(
                (
                    "num",
                    Pipeline([("imp", SimpleImputer(strategy="mean")), ("scl", StandardScaler())]),
                    num_cols,
                )
            )
        clf = LogisticRegression(max_iter=1000)
    else:
        if cat_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            (
                                "enc",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value", unknown_value=-1
                                ),
                            ),
                        ]
                    ),
                    cat_cols,
                )
            )
        if num_cols:
            transformers.append(("num", SimpleImputer(strategy="mean"), num_cols))
        clf = _xgb_classifier() if model == "xgboost" else _tabpfn_classifier()
        if clf is None:
            raise ImportError(f"{model} is not installed")

    preproc = ColumnTransformer(transformers, remainder="passthrough")
    return Pipeline([("pre", preproc), ("clf", clf)])


def run_one_baseline(
    dataset: str,
    spec: tuple,
    model: str,
    max_rows: int | None,
    cache_dir: Path | None,
    fe_model: str | None = None,
    skip_cache_read: bool = False,
) -> dict:
    cache_file = (
        cache_dir
        / f"{dataset}-{model}-{_baseline_cache_key(dataset, max_rows, fe_model=fe_model)}.json"
        if cache_dir
        else None
    )

    if cache_file and cache_file.exists() and not skip_cache_read:
        logger.info("[%s] %s cached — skipping", dataset, model)
        with open(cache_file) as f:
            return json.load(f)

    openml_name, version = spec[0], spec[1]
    csv_path = spec[2] if len(spec) > 2 else None
    target_col = spec[3] if len(spec) > 3 else None
    description = spec[4] if len(spec) > 4 else None

    X, y, class_map, _, _ = load_dataset(
        openml_name,
        version,
        max_rows,
        csv_path=csv_path,
        target_col=target_col,
        description=description,
        require_description=False,
    )
    n_classes = len(class_map)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    result = {
        "dataset": dataset,
        "model_id": model,
        "n_rows": len(X),
        "n_cols": X.shape[1],
        "n_classes": n_classes,
    }

    tag = f"[{dataset} × {model}]"
    if fe_model:
        from skribe import AdaptiveSkribeEngineer

        print(f"{tag} running AdaptiveSkribeEngineer ({fe_model})…", flush=True)
        try:
            fe_step = AdaptiveSkribeEngineer(model=fe_model, verbose=False)
            X_train = fe_step.fit_transform(X_train, y_train)
            X_test = fe_step.transform(X_test)
            skip = getattr(fe_step, "skip_reason_", None)
            result["fe_model"] = fe_model
            result["fe_skip_reason"] = skip
            result["fe_probe_delta"] = getattr(fe_step, "probe_delta_", None)
            if skip:
                print(f"{tag} AdaptiveFE skipped: {skip}", flush=True)
            else:
                print(f"{tag} AdaptiveFE done — {X_train.shape[1]} cols", flush=True)
        except Exception as e:
            print(f"{tag} AdaptiveFE FAILED: {e}  (using original features)", flush=True)
            result["fe_model"] = fe_model
            result["fe_error"] = str(e)

    t0 = time.time()
    try:
        pipe = _build_baseline_pipeline(model, X_train)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
        metrics = _rich_metrics(np.array(y_test), y_pred, y_proba, n_classes)
        metrics["fit_time_s"] = round(time.time() - t0, 2)
        metrics["status"] = "ok"
        result[model] = metrics
        print(f"  {model:<10} {dataset:<20} accuracy={metrics['accuracy']:.3f}", flush=True)
    except Exception as e:
        logger.warning("[%s] %s failed: %s", dataset, model, e)
        result[model] = {"error": str(e), "status": "error"}

    if cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result


def run_one_skribe(
    dataset: str,
    spec: tuple,
    model_id: str,
    max_rows: int | None,
    cache_dir: Path | None,
    vertex_region: str | None = None,
    web_search: bool = False,
    base_model_id: str | None = None,
    fe_model: str | None = None,
    skip_cache_read: bool = False,
    skip_context: bool = False,
    reasoning_effort: str | None = None,
) -> dict:
    actual_model_id = base_model_id or (model_id.removesuffix("-web") if web_search else model_id)
    tag = f"[{dataset} × {model_id}]"
    safe_model_id = model_id.replace("/", "-")
    # The effort suffix is only added to the filename when explicitly set, so
    # existing cache files (which predate reasoning_effort) keep their names
    # and a plain default-effort run still resolves to the same file.
    effort_suffix = f"-effort_{reasoning_effort}" if reasoning_effort else ""
    cache_file = (
        cache_dir
        / (
            f"{dataset}-{safe_model_id}{effort_suffix}-"
            f"{_cache_key(dataset, model_id, max_rows, fe_model=fe_model, web_search=web_search, reasoning_effort=reasoning_effort)}.json"
        )
        if cache_dir
        else None
    )
    if cache_file and cache_file.exists() and not skip_cache_read:
        with open(cache_file) as f:
            cached = json.load(f)
        # A cached record from a run that errored out (timeout, rate-limit,
        # etc.) is stored with accuracy=0.0 so it counts against the model in
        # aggregate charts (see plot_progression) instead of silently
        # dropping out of the average — but it must still be retried on the
        # next run rather than treated as a permanent result. Older cache
        # files predate the explicit "status" field, so fall back to the
        # presence of "error" for those.
        cached_skribe = cached.get("skribe")
        is_error = isinstance(cached_skribe, dict) and (
            cached_skribe.get("status") == "error"
            if "status" in cached_skribe
            else bool(cached_skribe.get("error"))
        )
        if not is_error:
            print(f"{tag} cached — skipping", flush=True)
            return cached
        print(
            f"{tag} cached result was a failure ({cached_skribe['error']!r}) — retrying", flush=True
        )

    openml_name, version = spec[0], spec[1]
    csv_path = spec[2] if len(spec) > 2 else None
    target_col = spec[3] if len(spec) > 3 else None
    spec_description = spec[4] if len(spec) > 4 else None

    print(f"{tag} loading dataset…", flush=True)
    X, y, class_map, description, y_str = load_dataset(
        openml_name,
        version,
        max_rows,
        csv_path=csv_path,
        target_col=target_col,
        description=spec_description,
        require_description=not skip_context,
    )
    n_classes = len(class_map)
    X_train, X_test, y_train, y_test, y_str_train, _ = train_test_split(
        X, y, y_str, test_size=0.25, random_state=42, stratify=y
    )
    print(f"{tag} {len(X)} rows  {X.shape[1]} cols  {n_classes} classes", flush=True)

    result = {
        "dataset": dataset,
        "model_id": model_id,
        "n_rows": len(X),
        "n_cols": X.shape[1],
        "n_classes": n_classes,
        "class_map": class_map,
    }
    if reasoning_effort:
        result["reasoning_effort"] = reasoning_effort

    from skribe import AdaptiveSkribeEngineer, SkribeClassifier

    if fe_model:
        print(f"{tag} running AdaptiveSkribeEngineer ({fe_model})…", flush=True)
        try:
            fe_step = AdaptiveSkribeEngineer(model=fe_model, verbose=False)
            X_train = fe_step.fit_transform(X_train, y_train)
            X_test = fe_step.transform(X_test)
            skip = getattr(fe_step, "skip_reason_", None)
            if skip:
                print(f"{tag} AdaptiveFE skipped: {skip}", flush=True)
            else:
                print(
                    f"{tag} AdaptiveFE done — {X_train.shape[1]} cols  delta={getattr(fe_step, 'probe_delta_', float('nan')):.3f}",
                    flush=True,
                )
        except Exception as e:
            print(f"{tag} AdaptiveFE FAILED: {e}  (using original features)", flush=True)

    t0 = time.time()
    try:
        clf = SkribeClassifier(
            model=actual_model_id,
            verbose=False,
            web_search=web_search,
            vertex_location=vertex_region or None,
            context_prepass=not skip_context,
            reasoning_effort=reasoning_effort,
        )

        _prepass_time = [0.0]
        _codegen_count = [0]
        _orig_call_llm = clf._call_llm

        def _instrumented_call_llm(prompt: str, web_search: bool = False, **kwargs) -> str:
            is_prepass = "preparing a structured dataset summary" in prompt
            is_extend = "extend any such mappings" in prompt
            if is_prepass:
                step = "context pre-pass"
            elif is_extend:
                step = "extend pass"
            else:
                _codegen_count[0] += 1
                step = (
                    "code generation"
                    if _codegen_count[0] == 1
                    else f"retry #{_codegen_count[0] - 1}"
                )
            ws_note = " [+web]" if web_search else ""
            print(f"{tag}   → {step}{ws_note}…", flush=True)
            t_llm = time.time()
            result_text = _orig_call_llm(prompt, web_search=web_search, **kwargs)
            dt = time.time() - t_llm
            if is_prepass:
                _prepass_time[0] += dt
            print(f"{tag}   ✓ {step} done  ({len(result_text):,} chars  {dt:.1f}s)", flush=True)
            return result_text

        clf._call_llm = _instrumented_call_llm

        print(f"{tag} fitting…", flush=True)
        t_fit = time.time()
        # Pass the original string labels (not the pre-encoded y_train) so
        # SkribeClassifier's own internal encoding knows the true class names
        # and can state them in the context pre-pass, instead of the pre-pass
        # LLM having to guess what a bare integer code means. Its classes_
        # mapping is guaranteed to match class_map (both are
        # sorted(unique-labels)), so clf.predict()'s integer output stays
        # directly comparable to y_test below.
        clf.fit(X_train, y_str_train, dataset_description=description or None)
        fit_elapsed = time.time() - t_fit
        print(
            f"{tag} fit done  ({fit_elapsed:.1f}s)  code={len(clf.raw_python_code_ or ''):,} chars",
            flush=True,
        )

        t_predict = time.time()
        y_pred = clf.predict(X_test)
        predict_elapsed = time.time() - t_predict
        y_proba = None
        if hasattr(clf, "predict_proba"):
            try:
                y_proba = clf.predict_proba(X_test)
            except Exception:
                pass

        result["skribe"] = _rich_metrics(np.array(y_test), y_pred, y_proba, n_classes)
        result["skribe"]["fit_time_s"] = round(fit_elapsed, 2)
        result["skribe"]["prepass_time_s"] = round(_prepass_time[0], 2)
        result["skribe"]["predict_time_s"] = round(predict_elapsed, 4)
        result["skribe"]["generated_code"] = clf.python_code_
        result["skribe"]["generated_code_raw"] = clf.raw_python_code_
        result["skribe"]["fit_prompt"] = getattr(clf, "fit_prompt_", None)
        result["skribe"]["context_prepass_prompt"] = getattr(clf, "context_prepass_prompt_", None)
        result["skribe"]["context_summary"] = getattr(clf, "context_summary_", None)
        result["skribe"]["fit_log"] = getattr(clf, "fit_log_", [])
        result["skribe"]["status"] = "ok"
        acc = result["skribe"]["accuracy"]
        print(
            f"{tag} accuracy={acc:.3f}  fit={fit_elapsed:.1f}s  predict={predict_elapsed:.4f}s  ✓",
            flush=True,
        )
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{tag} FAILED after {elapsed:.1f}s: {e}", flush=True)
        result["skribe"] = {
            "error": str(e),
            "status": "error",
            "fit_log": getattr(locals().get("clf"), "fit_log_", []),
        }

    # Cache failures too (accuracy=0.0, see build_summary_df) so aggregate
    # charts penalize a model that can't even produce a classifier for this
    # dataset, instead of silently omitting it from the average. The
    # cache-read path above detects the "error" key and retries these on the
    # next run rather than treating them as a permanent result.
    if cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        pl = result.get("skribe", {})
        if isinstance(pl, dict) and pl.get("error"):
            print(f"{tag} cached failure → {cache_file.name}", flush=True)
        else:
            print(f"{tag} cached → {cache_file.name}", flush=True)

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        choices=ALL_MODELS,
        required=True,
        help="Model to fit: logreg | xgboost | tabpfn | skribe",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DEFAULT_DATASETS),
        metavar="DATASET",
        help=f"Dataset to run. One of: {', '.join(DEFAULT_DATASETS)}",
    )
    parser.add_argument(
        "--llm",
        metavar="MODEL_ID",
        help="LLM model_id (required when --model skribe). Use --list-models to see valid values.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print valid --llm values and exit.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap dataset rows before split (default: no cap).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmark_results",
        help="Directory for cached results (default: artifacts/benchmark_results).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing cache files and re-run, but still write results.",
    )
    parser.add_argument(
        "--skip-context",
        action="store_true",
        help="Disable skribe context pre-pass (debugging only).",
    )
    parser.add_argument(
        "--fe-model",
        default=None,
        metavar="MODEL_ID",
        help="LLM for AdaptiveSkribeEngineer applied before fitting --model (logreg/xgboost/skribe).",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        metavar="EFFORT",
        help="Reasoning effort for --model skribe (e.g. low/medium/high/xhigh, provider-"
        "dependent). Included in the cache filename/key when set, so different-effort "
        "runs of the same dataset+model don't collide.",
    )
    args = parser.parse_args(argv)

    if args.list_models:
        print("Valid --llm values:")
        for m in MODEL_PROGRESSION:
            print(f"  {m['model_id']:<45}  {m['label']}")
        return 0

    if args.model == "skribe":
        if not args.llm:
            parser.error(
                "--llm is required when --model skribe (use --list-models to see valid values)"
            )
        if args.llm not in _MODEL_LOOKUP:
            parser.error(f"Unknown --llm {args.llm!r}. Use --list-models to see valid values.")
    elif args.llm:
        parser.error("--llm is only valid when --model skribe")

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.getLogger("skribe").setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    dataset = args.dataset

    label = args.model if args.model != "skribe" else _MODEL_LOOKUP[args.llm].get("label", args.llm)
    print(
        f"model={label}  dataset={dataset}  "
        f"max_rows={args.max_rows}  cache={cache_dir}{'  (skip-read)' if args.no_cache else ''}",
        flush=True,
    )

    if args.model in BASELINE_MODELS:
        try:
            run_one_baseline(
                dataset,
                DEFAULT_DATASETS[dataset],
                args.model,
                args.max_rows,
                cache_dir,
                fe_model=args.fe_model,
                skip_cache_read=args.no_cache,
            )
        except Exception as e:
            logger.warning("[%s] failed: %s", dataset, e)
            return 1
        return 0

    # skribe
    meta = _MODEL_LOOKUP[args.llm]
    run_one_skribe(
        dataset,
        DEFAULT_DATASETS[dataset],
        args.llm,
        args.max_rows,
        cache_dir,
        vertex_region=meta.get("vertex_region"),
        web_search=meta.get("web_search", False),
        base_model_id=meta.get("base_model_id"),
        fe_model=args.fe_model,
        skip_cache_read=args.no_cache,
        skip_context=args.skip_context,
        reasoning_effort=args.reasoning_effort,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
