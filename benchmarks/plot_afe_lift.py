#!/usr/bin/env python
"""Report AdaptiveSkribeEngineer (AFE) lift: with-FE vs without-FE accuracy,
per dataset, for logreg and xgboost.

Reads cache files already produced by run_openml_fit.py (via
run_afe_benchmark.sh) — one pair of files per (dataset, learner): one fit
without --fe-model, one fit with it. Does not fit anything itself.

Usage
-----
    benchmarks/run_afe_benchmark.sh          # runs the fits, then calls this
    .venv/bin/python benchmarks/plot_afe_lift.py --fe-model gpt-5.5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_utils import DEFAULT_DATASETS

LEARNERS = ("logreg", "xgboost")

# Datasets whose feature names are abstract (a1, a2, ...) rather than
# semantically meaningful — an LLM has no world knowledge to exploit for
# these, so AFE is expected to help less here.
ABSTRACT_DATASETS = {"tic-tac-toe", "kr-vs-kp", "monks-2"}


def load_records(cache_dir: Path, fe_model: str) -> list[dict]:
    """One record per dataset: {dataset, skip_reason, afe_delta: {logreg, xgboost}}."""
    records = []
    for dataset in DEFAULT_DATASETS:
        record = {"dataset": dataset, "skip_reason": None, "afe_delta": {}}
        for learner in LEARNERS:
            without_acc = None
            with_acc = None
            skip_reason = None
            for path in sorted(cache_dir.glob(f"{dataset}-{learner}-*.json")):
                data = json.loads(path.read_text())
                if data.get("dataset") != dataset or data.get("model_id") != learner:
                    continue
                metrics = data.get(learner, {})
                acc = metrics.get("accuracy")
                if data.get("fe_model") == fe_model:
                    with_acc = acc
                    skip_reason = data.get("fe_skip_reason")
                elif not data.get("fe_model"):
                    without_acc = acc
            if with_acc is not None and without_acc is not None:
                record["afe_delta"][learner] = with_acc - without_acc
            if skip_reason:
                record["skip_reason"] = skip_reason
        records.append(record)
    return records


def print_table(records: list[dict]):
    print(f"\n{'dataset':<18} {'skip reason':<45} {'lr_Δ':>7} {'xgb_Δ':>7}")
    print("-" * 82)
    for r in sorted(
        records, key=lambda r: r["afe_delta"].get("logreg", float("-inf")), reverse=True
    ):
        lr_d = r["afe_delta"].get("logreg", float("nan"))
        xgb_d = r["afe_delta"].get("xgboost", float("nan"))
        skip = r["skip_reason"] or "-"
        print(f"{r['dataset']:<18} {skip[:45]:<45} {lr_d:>+7.3f} {xgb_d:>+7.3f}")


def plot_lift(records: list[dict], out_path: Path, fe_model: str):
    datasets = [r["dataset"] for r in records]
    lr_delta = [r["afe_delta"].get("logreg", 0.0) for r in records]
    xgb_delta = [r["afe_delta"].get("xgboost", 0.0) for r in records]
    skipped = [r["skip_reason"] is not None for r in records]

    order = sorted(range(len(datasets)), key=lambda i: lr_delta[i], reverse=True)
    datasets = [datasets[i] for i in order]
    lr_delta = [lr_delta[i] for i in order]
    xgb_delta = [xgb_delta[i] for i in order]
    skipped = [skipped[i] for i in order]

    def bar_color(ds, val, is_skipped):
        if is_skipped:
            return "#aaaaaa"
        if ds in ABSTRACT_DATASETS:
            return "#9467bd" if val >= 0 else "#c5b0d5"
        return "#2ca02c" if val >= 0 else "#d62728"

    x = np.arange(len(datasets))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, (ds, ld, xd, sk) in enumerate(zip(datasets, lr_delta, xgb_delta, skipped)):
        ax.bar(i - w / 2, ld, w, color=bar_color(ds, ld, sk), alpha=0.85)
        ax.bar(i + w / 2, xd, w, color=bar_color(ds, xd, sk), alpha=0.55)

    ax.axhline(0, color="black", linewidth=0.8)

    for i, (ld, xd, sk) in enumerate(zip(lr_delta, xgb_delta, skipped)):
        for val, offset in [(ld, -w / 2), (xd, w / 2)]:
            if math.isnan(val) or (sk and val == 0.0):
                continue
            ax.text(
                i + offset,
                val + (0.005 if val >= 0 else -0.008),
                f"{val:+.2f}" if not sk else "skip",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=7,
            )

    for i, (ds, sk, r) in enumerate(zip(datasets, skipped, [records[j] for j in order])):
        if sk:
            reason = r["skip_reason"] or ""
            short = "n_rows" if "n_rows" in reason else "probe_delta"
            ax.text(
                i, 0.005, f"skip\n({short})",
                ha="center", va="bottom", fontsize=6.5, color="#666666",
            )

    ax.set_xticks(x)
    labels = [
        f"{ds}\n({'abstract' if ds in ABSTRACT_DATASETS else 'semantic'})"
        for ds in datasets
    ]
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Accuracy delta (AFE − no FE)")
    ax.set_title(
        f"Per-dataset AdaptiveFE lift ({fe_model} SkribeFeatureEngineer)\n"
        "Dark bars = logreg  ·  Light bars = xgboost  ·  Grey = skipped by pre-flight  ·  Purple = abstract",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ca02c", alpha=0.85, label="logreg +AFE (semantic)"),
        Patch(facecolor="#2ca02c", alpha=0.55, label="xgboost +AFE (semantic)"),
        Patch(facecolor="#9467bd", alpha=0.85, label="logreg +AFE (abstract)"),
        Patch(facecolor="#9467bd", alpha=0.55, label="xgboost +AFE (abstract)"),
        Patch(facecolor="#aaaaaa", alpha=0.85, label="skipped (pre-flight)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--fe-model", required=True, metavar="MODEL_ID",
        help="fe_model value the with-FE cache files were run with.",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/benchmark_results",
        help="Directory containing cache/ (default: artifacts/benchmark_results).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"

    records = load_records(cache_dir, args.fe_model)
    print_table(records)
    plot_lift(records, output_dir / "fe_per_dataset_lift_afe.png", args.fe_model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
