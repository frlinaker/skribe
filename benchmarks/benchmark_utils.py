"""Shared utilities for the skribe model-progression benchmarks.

This module is imported by run_baselines.py, run_skribe.py, and collate.py.
It must NOT import from skribe itself — only the runner scripts do that.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from adjustText import adjust_text
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

logger = logging.getLogger("skribe.progression")

CACHE_SCHEMA = "progression-v1"

# Ordered oldest → newest. release_date is approximate; used as the x-axis value.
MODEL_PROGRESSION = [
    # OpenAI
    {
        "model_id": "gpt-4o",
        "label": "GPT-4o",
        "release_date": date(2024, 5, 13),
        "family": "GPT-4",
        "provider": "openai",
    },
    {
        "model_id": "gpt-4o-mini",
        "label": "GPT-4o mini",
        "release_date": date(2024, 7, 18),
        "family": "GPT-4",
        "provider": "openai",
    },
    {
        "model_id": "gpt-4o-mini+web",
        "base_model_id": "gpt-4o-mini",
        "label": "GPT-4o mini +web",
        "release_date": date(2024, 7, 18),
        "family": "GPT-4",
        "provider": "openai",
        "web_search": True,
    },
    {
        "model_id": "gpt-4.1",
        "label": "GPT-4.1",
        "release_date": date(2025, 4, 14),
        "family": "GPT-4.1",
        "provider": "openai",
    },
    {
        "model_id": "gpt-4.1+web",
        "base_model_id": "gpt-4.1",
        "label": "GPT-4.1 +web",
        "release_date": date(2025, 4, 14),
        "family": "GPT-4.1",
        "provider": "openai",
        "web_search": True,
    },
    {
        "model_id": "gpt-5.4-mini",
        "label": "GPT-5.4 mini",
        "release_date": date(2026, 3, 17),  # GA: Mar 17 2026
        "family": "GPT-5",
        "provider": "openai",
    },
    {
        "model_id": "gpt-5.4-mini+web",
        "base_model_id": "gpt-5.4-mini",
        "label": "GPT-5.4 mini +web",
        "release_date": date(2026, 3, 17),
        "family": "GPT-5",
        "provider": "openai",
        "web_search": True,
    },
    {
        "model_id": "gpt-5.5",
        "label": "GPT-5.5",
        "release_date": date(2026, 4, 23),  # GA: Apr 23 2026
        "family": "GPT-5",
        "provider": "openai",
    },
    {
        "model_id": "gpt-5.5+web",
        "base_model_id": "gpt-5.5",
        "label": "GPT-5.5 +web",
        "release_date": date(2026, 4, 23),  # GA: Apr 23 2026
        "family": "GPT-5",
        "provider": "openai",
        "web_search": True,
    },
    # Google Gemini (via Vertex AI) — ordered oldest GA → newest
    {
        "model_id": "vertex_ai/gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
        "release_date": date(2025, 5, 20),  # GA: May 20 2025
        "family": "Gemini 2.5",
        "provider": "google",
        "vertex_region": "us-central1",
    },
    {
        "model_id": "vertex_ai/gemini-2.5-flash+web",
        "base_model_id": "vertex_ai/gemini-2.5-flash",
        "label": "Gemini 2.5 Flash +web",
        "release_date": date(2025, 5, 20),
        "family": "Gemini 2.5",
        "provider": "google",
        "vertex_region": "us-central1",
        "web_search": True,
    },
    {
        "model_id": "vertex_ai/gemini-2.5-pro",
        "label": "Gemini 2.5 Pro",
        "release_date": date(2025, 6, 5),  # GA: Jun 5 2025
        "family": "Gemini 2.5",
        "provider": "google",
        "vertex_region": "us-central1",
    },
    {
        "model_id": "vertex_ai/gemini-2.5-pro+web",
        "base_model_id": "vertex_ai/gemini-2.5-pro",
        "label": "Gemini 2.5 Pro +web",
        "release_date": date(2025, 6, 5),
        "family": "Gemini 2.5",
        "provider": "google",
        "vertex_region": "us-central1",
        "web_search": True,
    },
    {
        "model_id": "vertex_ai/gemini-2.5-flash-lite",
        "label": "Gemini 2.5 Flash Lite",
        "release_date": date(2025, 7, 22),  # GA: Jul 22 2025
        "family": "Gemini 2.5",
        "provider": "google",
        "vertex_region": "us-central1",
    },
    {
        "model_id": "vertex_ai/gemini-2.5-flash-lite+web",
        "base_model_id": "vertex_ai/gemini-2.5-flash-lite",
        "label": "Gemini 2.5 Flash Lite +web",
        "release_date": date(2025, 7, 22),
        "family": "Gemini 2.5",
        "provider": "google",
        "vertex_region": "us-central1",
        "web_search": True,
    },
    {
        "model_id": "vertex_ai/gemini-3.5-flash",
        "label": "Gemini 3.5 Flash",
        "release_date": date(2026, 5, 19),
        "family": "Gemini 3",
        "provider": "google",
        "vertex_region": "asia-southeast1",
    },
    {
        "model_id": "vertex_ai/gemini-3.5-flash+web",
        "base_model_id": "vertex_ai/gemini-3.5-flash",
        "label": "Gemini 3.5 Flash +web",
        "release_date": date(2026, 5, 19),
        "family": "Gemini 3",
        "provider": "google",
        "vertex_region": "asia-southeast1",
        "web_search": True,
    },
]

DEFAULT_DATASETS = {
    "adult": ("adult", 2),
    "credit-g": ("credit-g", 1),
    "bank-marketing": ("bank-marketing", 1),
    "mushroom": ("mushroom", 1),
    "car": ("car", 3),
    "nursery": ("nursery", 3),
    "vote": ("vote", 1),
    "tic-tac-toe": ("tic-tac-toe", 1),
    "kr-vs-kp": ("kr-vs-kp", 1),
    "monks-2": ("monks-problems-2", 1),
    "soybean": ("soybean", 1),
    "hepatitis": ("hepatitis", 1),
    "lymph": ("lymph", 1),
    "zoo": ("zoo", 1),
    "heart-statlog": ("heart-statlog", 1),
    "spotify-genre": (
        None, None,
        Path(__file__).parent.parent / "examples" / "external_data" / "spotify_genre.csv",
        "genre",
        (
            "Spotify track genre classification. Each row is a unique track. "
            "Target is the playlist genre: edm, latin, pop, r&b, rap, rock. "
            "Features include track_name and track_artist (text identifiers useful "
            "for world-knowledge lookup), track_popularity (0-100 Spotify score), "
            "and audio features: danceability (0-1, rhythmic suitability for dancing), "
            "energy (0-1, intensity/activity), key (musical key 0-11), "
            "loudness (dB, typically -60 to 0), mode (1=major, 0=minor), "
            "speechiness (0-1, presence of spoken words), "
            "acousticness (0-1, confidence track is acoustic), "
            "instrumentalness (0-1, predicts no vocals), "
            "liveness (0-1, presence of live audience), "
            "valence (0-1, musical positiveness), tempo (BPM), "
            "duration_ms (track length in milliseconds)."
        ),
    ),
}


def load_dataset(openml_name, version, max_rows: int | None, csv_path=None, target_col=None, description=None, require_description=True):
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        y = df[target_col].astype(str)
        X = df.drop(columns=[target_col])
        resolved_description = description or ""
    else:
        bunch = fetch_openml(
            name=openml_name, version=version, as_frame=True, parser="auto"
        )
        X = bunch.data.copy()
        y = pd.Series(np.asarray(bunch.target)).astype(str)
        resolved_description = description or getattr(bunch, "DESCR", None) or ""
    if require_description and not resolved_description:
        raise ValueError(
            f"Dataset has no description — the context pre-pass cannot run. "
            f"Add a description string to the DEFAULT_DATASETS entry, "
            f"or pass --skip-context to explicitly disable the pre-pass."
        )
    classes = {c: i for i, c in enumerate(sorted(y.unique()))}
    y = y.map(classes).astype(int)
    if max_rows and len(X) > max_rows:
        X = X.sample(max_rows, random_state=42)
        y = y.loc[X.index]
    return X.reset_index(drop=True), y.reset_index(drop=True), classes, resolved_description


def _rich_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None, n_classes: int
) -> dict:
    """Compute a broad set of classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "error_rate": float(1 - accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None:
        try:
            if n_classes == 2:
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
                lb = LabelBinarizer().fit(y_true)
                y_bin = lb.transform(y_true)
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            pass
    return metrics


def _cache_key(
    dataset: str,
    model_id: str,
    max_rows: int | None,
    fe_model: str | None = None,
    web_search: bool = False,
) -> str:
    raw = f"{CACHE_SCHEMA}|{dataset}|{model_id}|{max_rows}|fe={fe_model or ''}|ws={web_search}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _baseline_cache_key(dataset: str, max_rows: int | None) -> str:
    raw = f"{CACHE_SCHEMA}|baselines|{dataset}|{max_rows}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _xgb_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=4, verbosity=0
    )


def _tabpfn_classifier():
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return None
    return TabPFNClassifier(ignore_pretraining_limits=True)


def build_summary_df(results: list[dict]) -> pd.DataFrame:
    """Long-form DataFrame: one row per (dataset, learner) with all metrics.

    skribe learner names are qualified as "skribe[<llm-label>]" so they
    are never confused with the LLM model dimension.  Baseline learners
    (logreg, xgboost, tabpfn) appear once per dataset with no LLM association.

    Accepts both skribe cache dicts (which have "dataset" and "model_id" keys)
    and baseline-only cache dicts (which have "logreg"/"xgboost"/"tabpfn" keys but
    no "dataset"/"model_id").  Callers must inject a "dataset" key into baseline-only
    dicts before passing them here (collate.py does this).
    """
    rows = []
    model_meta = {m["model_id"]: m for m in MODEL_PROGRESSION}
    seen_baselines: set[tuple] = set()  # (dataset, learner) — emit baselines once

    for r in results:
        dataset = r.get("dataset")
        model_id = r.get("model_id")

        # skribe rows carry both dataset and model_id
        if dataset and model_id:
            meta = model_meta.get(model_id, {})
            llm_label = meta.get("label", model_id)

            if "skribe" in r and "error" not in r["skribe"]:
                m = r["skribe"]
                web_search = meta.get("web_search", False)
                row = {
                    "dataset": dataset,
                    "model_id": model_id,
                    "llm_label": llm_label,
                    "release_date": str(meta.get("release_date", "")),
                    "family": meta.get("family", ""),
                    "provider": meta.get("provider", "openai"),
                    "web_search": web_search,
                    "learner": f"skribe[{llm_label}]",
                    "n_rows": r.get("n_rows"),
                    "n_cols": r.get("n_cols"),
                    "n_classes": r.get("n_classes"),
                }
                row.update({k: v for k, v in m.items() if k not in ("fit_time_s",)})
                row["fit_time_s"] = m.get("fit_time_s")
                rows.append(row)

        # Emit baselines from any dict that has a dataset key (including skribe
        # cache files that embed baselines from the old run_model_progression flow).
        if dataset:
            for learner in ("logreg", "xgboost", "tabpfn"):
                key = (dataset, learner)
                if key in seen_baselines:
                    continue
                if learner not in r or "error" in r[learner]:
                    continue
                seen_baselines.add(key)
                m = r[learner]
                row = {
                    "dataset": dataset,
                    "model_id": None,
                    "llm_label": None,
                    "release_date": None,
                    "family": None,
                    "provider": None,
                    "learner": learner,
                    "n_rows": r.get("n_rows"),
                    "n_cols": r.get("n_cols"),
                    "n_classes": r.get("n_classes"),
                }
                row.update({k: v for k, v in m.items() if k not in ("fit_time_s",)})
                row["fit_time_s"] = m.get("fit_time_s")
                rows.append(row)

    return pd.DataFrame(rows)


def plot_progression(df: pd.DataFrame, output_dir: Path):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    import matplotlib as mpl
    mpl.rcParams["grid.alpha"] = 0.18
    mpl.rcParams["grid.color"] = "#b0b0b0"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── shared prep ──────────────────────────────────────────────────────────
    # skribe rows have a release_date; baseline rows do not.
    pl_df = df[df["learner"].str.startswith("skribe[")].copy()
    pl_df["release_date"] = pd.to_datetime(pl_df["release_date"])

    if "web_search" not in pl_df.columns:
        pl_df["web_search"] = False
    pl_df["web_search"] = pl_df["web_search"].fillna(False)

    pl_summary = (
        pl_df.groupby(
            [
                "model_id",
                "llm_label",
                "release_date",
                "learner",
                "provider",
                "web_search",
            ]
        )["accuracy"]
        .mean()
        .reset_index()
        .sort_values("release_date")
    )

    pl_data = pl_summary.copy()

    lr_data = df[df["learner"] == "logreg"]
    xgb_data = df[df["learner"] == "xgboost"]
    tabpfn_data = df[df["learner"] == "tabpfn"]

    n_datasets = df["dataset"].nunique()

    # ── 1. Timeline: mean accuracy vs model release date ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    _baseline_ys: list[float] = []

    if not lr_data.empty:
        lr_mean = lr_data["accuracy"].mean()
        _baseline_ys.append(lr_mean)
        ax.axhline(
            lr_mean,
            color="#4878CF",
            linewidth=1.8,
            linestyle="--",
            label=f"Logistic Regression  ({lr_mean:.3f})",
        )

    if not xgb_data.empty:
        xgb_mean = xgb_data["accuracy"].mean()
        _baseline_ys.append(xgb_mean)
        ax.axhline(
            xgb_mean,
            color="#6ACC65",
            linewidth=1.8,
            linestyle="--",
            label=f"XGBoost  ({xgb_mean:.3f})",
        )

    if not tabpfn_data.empty:
        tabpfn_mean = tabpfn_data["accuracy"].mean()
        _baseline_ys.append(tabpfn_mean)
        ax.axhline(
            tabpfn_mean,
            color="#FF7F0E",
            linewidth=1.8,
            linestyle="--",
            label=f"TabPFN  ({tabpfn_mean:.3f})",
        )

    # skribe — one solid line per provider (base models) + one dashed line
    # per provider (+web models).  Cumulative-max envelope so weaker models
    # don't cause visual dips.
    provider_styles = {
        "openai": {"color": "#D65F5F", "marker": "o", "label": "skribe / OpenAI GPT"},
        "google": {
            "color": "#4285F4",
            "marker": "s",
            "label": "skribe / Google Gemini",
        },
    }
    if "web_search" not in pl_data.columns:
        pl_data["web_search"] = False

    _annotation_texts: list = []
    _scatter_objects: list = []

    if not pl_data.empty:
        # Split standard vs web-search rows
        pl_standard = pl_data[~pl_data["web_search"].fillna(False)].copy()
        pl_web = pl_data[pl_data["web_search"].fillna(False)].copy()

        for provider, grp in pl_standard.groupby("provider"):
            grp = grp.sort_values("release_date").reset_index(drop=True)
            style = provider_styles.get(
                provider,
                {"color": "#999", "marker": "o", "label": f"skribe / {provider}"},
            )
            color = style["color"]

            # Cumulative-max envelope line (the "best so far" trajectory).
            grp["best_so_far"] = grp["accuracy"].cummax()
            final_acc = grp["best_so_far"].iloc[-1]
            ax.plot(
                grp["release_date"],
                grp["best_so_far"],
                color=color,
                linewidth=2.5,
                linestyle="-",
                label=f"{style['label']} ({final_acc:.3f})",
                zorder=3,
            )

            # Individual model dots — weaker models shown slightly faded.
            for _, row in grp.iterrows():
                is_best = abs(row["accuracy"] - row["best_so_far"]) < 1e-9
                alpha = 1.0 if is_best else 0.55
                sc = ax.scatter(
                    row["release_date"],
                    row["accuracy"],
                    marker="o",
                    s=60,
                    color=color,
                    alpha=alpha,
                    zorder=4,
                )
                _scatter_objects.append(sc)
                txt = ax.text(
                    row["release_date"],
                    row["accuracy"],
                    row["llm_label"],
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=color,
                    alpha=alpha,
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                )
                _annotation_texts.append(txt)

        # Web-search variants: separate dashed envelope line per provider,
        # star markers, annotations offset to the right to avoid overlap.
        for provider, grp in pl_web.groupby("provider"):
            grp = grp.sort_values("release_date").reset_index(drop=True)
            style = provider_styles.get(
                provider,
                {"color": "#999", "marker": "o", "label": f"skribe / {provider}"},
            )
            color = style["color"]
            web_label = f"skribe / {'OpenAI GPT' if provider == 'openai' else 'Google Gemini'} +web"

            grp["best_so_far"] = grp["accuracy"].cummax()
            final_acc_web = grp["best_so_far"].iloc[-1]
            ax.plot(
                grp["release_date"],
                grp["best_so_far"],
                color=color,
                linewidth=2.0,
                linestyle=":",
                label=f"{web_label} ({final_acc_web:.3f})",
                zorder=3,
                alpha=0.85,
            )

            for _, row in grp.iterrows():
                is_best = abs(row["accuracy"] - row["best_so_far"]) < 1e-9
                alpha = 0.9 if is_best else 0.5
                sc = ax.scatter(
                    row["release_date"],
                    row["accuracy"],
                    marker="o",
                    s=60,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=1.5,
                    alpha=alpha,
                    zorder=5,
                )
                _scatter_objects.append(sc)
                txt = ax.text(
                    row["release_date"],
                    row["accuracy"],
                    row["llm_label"],
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=color,
                    alpha=alpha,
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                )
                _annotation_texts.append(txt)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    fig.autofmt_xdate(rotation=30)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    all_acc = pl_data["accuracy"]
    # Extra top margin so labels for the top-right cluster have room to spread.
    ax.set_ylim(max(0.0, all_acc.min() - 0.08), min(1.05, all_acc.max() + 0.20))

    if _annotation_texts:
        # Collect scatter x/y coords so adjust_text can repel labels from points.
        _pt_x = [sc.get_offsets()[:, 0].tolist() for sc in _scatter_objects]
        _pt_x = [x for sub in _pt_x for x in sub]
        _pt_y = [sc.get_offsets()[:, 1].tolist() for sc in _scatter_objects]
        _pt_y = [y for sub in _pt_y for y in sub]

        # Add phantom points along each baseline so labels are repelled from them.
        if _baseline_ys and _pt_x:
            import numpy as _np
            x_min, x_max = min(_pt_x), max(_pt_x)
            _phantom_x = _np.linspace(x_min, x_max, 30).tolist()
            for _by in _baseline_ys:
                _pt_x.extend(_phantom_x)
                _pt_y.extend([_by] * 30)

        adjust_text(
            _annotation_texts,
            x=_pt_x,
            y=_pt_y,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.7),
            expand=(2.0, 2.0),
            force_text=(1.5, 1.5),
            force_points=(2.0, 2.0),
            avoid_self=True,
            only_move={"text": "xy", "points": "xy"},
        )
    ax.set_xlabel("Model release date", fontsize=12)
    ax.set_ylabel(f"Mean accuracy ({n_datasets} datasets)", fontsize=12)
    ax.set_title(
        "skribe accuracy grows with LLM evolution\n"
        "Classical ML baselines shown as dashed horizontals",
        fontsize=13,
    )
    # Sort legend entries by the numeric value embedded in the label (highest first).
    handles, labels = ax.get_legend_handles_labels()
    import re as _re
    def _legend_sort_key(hl):
        m = _re.search(r"\((\d+\.\d+)\)", hl[1])
        return -float(m.group(1)) if m else 0.0
    handles, labels = zip(*sorted(zip(handles, labels), key=_legend_sort_key)) if handles else (handles, labels)
    ax.legend(handles, labels, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = output_dir / "model_progression.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved timeline chart → %s", out)
    plt.close(fig)

    # ── 2. Per-dataset heatmap: datasets × LLM models, skribe accuracy ──
    # Order columns by release date.
    col_order = (
        pl_data.sort_values("release_date")["llm_label"].tolist()
        if not pl_data.empty
        else None
    )
    pl_pivot = pl_df.pivot_table(
        index="dataset", columns="llm_label", values="accuracy"
    )
    if col_order:
        pl_pivot = pl_pivot.reindex(
            columns=[c for c in col_order if c in pl_pivot.columns]
        )
    # Sort rows by mean accuracy ascending so weakest datasets sit at the top.
    pl_pivot = pl_pivot.loc[pl_pivot.mean(axis=1).sort_values().index]

    if not pl_pivot.empty:
        fig2, ax2 = plt.subplots(
            figsize=(
                max(8, len(pl_pivot.columns) * 1.8),
                max(5, len(pl_pivot) * 0.7 + 1.5),
            )
        )
        sns.heatmap(
            pl_pivot,
            ax=ax2,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0.5,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Accuracy", "shrink": 0.8},
        )
        ax2.set_title("skribe accuracy per dataset × model", fontsize=12, pad=12)
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        ax2.tick_params(axis="x", rotation=30)
        ax2.tick_params(axis="y", rotation=0)
        fig2.tight_layout()
        out2 = output_dir / "per_dataset_heatmap.png"
        fig2.savefig(out2, dpi=150)
        logger.info("Saved heatmap → %s", out2)
        plt.close(fig2)

    # ── 3. All-learner bar chart: two rows (no-web / +web), columns aligned ──
    # Columns = base model labels ordered by release date; gap where no web variant.
    _provider_bar_color = {"openai": "#D65F5F", "google": "#4285F4"}
    if not pl_data.empty:
        pl_bar_all = (
            pl_data.groupby(["llm_label", "release_date", "provider", "web_search"])["accuracy"]
            .mean()
            .reset_index()
            .sort_values("release_date")
        )
        # Determine column order from base (no-web) models only.
        base_order = (
            pl_bar_all[~pl_bar_all["web_search"].fillna(False)]
            .sort_values("release_date")["llm_label"]
            .tolist()
        )
        # Map base label → web label (strip " +web" suffix from web rows).
        # Web row label is "<base_label> +web" by convention.
        web_rows = pl_bar_all[pl_bar_all["web_search"].fillna(False)].copy()
        web_rows["base_label"] = web_rows["llm_label"].str.replace(r"\s*\+web$", "", regex=True)
        web_by_base: dict = {row["base_label"]: row for _, row in web_rows.iterrows()}

        n_cols = len(base_order)
        x = np.arange(n_cols)

        baseline_styles_3 = [
            ("logreg", lr_data, "#4878CF", "Logistic Regression"),
            ("xgboost", xgb_data, "#6ACC65", "XGBoost"),
            ("tabpfn", tabpfn_data, "#FF7F0E", "TabPFN"),
        ]
        baseline_means_3 = {
            lbl: bdata["accuracy"].mean()
            for _, bdata, _, lbl in baseline_styles_3
            if not bdata.empty
        }

        fig3, (ax3_top, ax3_bot) = plt.subplots(
            2, 1,
            figsize=(max(10, n_cols * 1.5), 10),
            sharex=True,
        )

        base_lookup = {
            row["llm_label"]: row
            for _, row in pl_bar_all[~pl_bar_all["web_search"].fillna(False)].iterrows()
        }

        def _draw_bar_row(ax, lookup, title_suffix, is_web_row=False):
            _legend_seen: set[str] = set()
            for i, lbl in enumerate(base_order):
                row = lookup.get(lbl)
                if row is None:
                    continue
                prov = row["provider"]
                color = _provider_bar_color.get(prov, "#999999")
                prov_name = "OpenAI GPT" if prov == "openai" else "Google Gemini"
                label_str = f"skribe / {prov_name}"
                bar = ax.bar(
                    i, row["accuracy"], color=color,
                    label=label_str if label_str not in _legend_seen else "_nolegend_",
                    zorder=3,
                )
                _legend_seen.add(label_str)
                ax.text(
                    i, row["accuracy"] + 0.005,
                    f"{row['accuracy']:.2f}",
                    ha="center", va="bottom", fontsize=8,
                )
            for lbl, val in baseline_means_3.items():
                color_map = {
                    "Logistic Regression": "#4878CF",
                    "XGBoost": "#6ACC65",
                    "TabPFN": "#FF7F0E",
                }
                c = color_map.get(lbl, "#888")
                ax.axhline(val, color=c, linewidth=1.8, linestyle="--",
                           label=f"{lbl}  ({val:.3f})", zorder=4)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
            ax.set_ylim(0, 1.12)
            ax.set_ylabel(f"Mean accuracy ({n_datasets} datasets)", fontsize=11)
            ax.set_title(title_suffix, fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.18)
            handles_l, labels_l = ax.get_legend_handles_labels()
            def _lk(hl):
                m = _re.search(r"\((\d+\.\d+)\)", hl[1])
                return -float(m.group(1)) if m else 0.0
            if handles_l:
                handles_l, labels_l = zip(*sorted(zip(handles_l, labels_l), key=_lk))
            ax.legend(handles_l, labels_l, fontsize=9, loc="lower right")

        _draw_bar_row(ax3_top, base_lookup, "Without web search")
        _draw_bar_row(ax3_bot, web_by_base, "With web search  (+web variants only; gaps = no web support)")

        ax3_bot.set_xticks(x)
        ax3_bot.set_xticklabels(base_order, rotation=30, ha="right", fontsize=9)
        ax3_bot.set_xlabel("LLM model (oldest → newest)", fontsize=12)

        fig3.suptitle(
            "skribe vs baselines: mean accuracy by LLM generation",
            fontsize=13, y=1.01,
        )
        fig3.tight_layout()
        out3 = output_dir / "all_learners_bar.png"
        fig3.savefig(out3, dpi=150, bbox_inches="tight")
        logger.info("Saved grouped bar chart → %s", out3)
        plt.close(fig3)

    # ── 4. Gap-to-baseline chart: skribe vs best baseline per LLM ───────
    if not pl_data.empty:
        best_baseline_acc = max(
            (
                bdata["accuracy"].mean()
                for bdata in [lr_data, xgb_data, tabpfn_data]
                if not bdata.empty
            ),
            default=None,
        )
        if best_baseline_acc is not None:
            pl_bar2 = (
                pl_data.groupby(["llm_label", "release_date", "provider", "web_search"])["accuracy"]
                .mean()
                .reset_index()
                .sort_values("release_date")
            )
            pl_bar2["gap"] = pl_bar2["accuracy"] - best_baseline_acc

            n_gap = len(pl_bar2)
            fig4, ax4 = plt.subplots(figsize=(max(12, n_gap * 1.1), 5))
            # Solid provider color when above baseline, desaturated when below.
            # +web bars get hatching.
            for i, (_, row) in enumerate(pl_bar2.iterrows()):
                base = _provider_bar_color.get(row["provider"], "#999999")
                is_web = bool(row.get("web_search", False))
                color = base if row["gap"] >= 0 else base + "80"
                ax4.bar(
                    i, row["gap"], color=color,
                    hatch="//" if is_web else "",
                    edgecolor="white" if not is_web else base,
                    linewidth=0.5,
                    zorder=3,
                )
            ax4.set_xticks(range(n_gap))
            ax4.set_xticklabels(pl_bar2["llm_label"].tolist(), rotation=25, ha="right", fontsize=9)
            for i, (_, row) in enumerate(pl_bar2.iterrows()):
                ax4.text(
                    i,
                    row["gap"] + (0.004 if row["gap"] >= 0 else -0.008),
                    f"{row['gap']:+.3f}",
                    ha="center",
                    va="bottom" if row["gap"] >= 0 else "top",
                    fontsize=9,
                )
            ax4.set_xlabel("LLM model (oldest → newest)", fontsize=12)
            ax4.set_ylabel("Accuracy gap vs best baseline", fontsize=12)
            ax4.set_title(
                "skribe gap to best baseline (logreg / XGBoost / TabPFN)\n"
                "Solid = above baseline  ·  Faded = below  ·  // = +web search",
                fontsize=12,
            )
            ax4.yaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1.0, decimals=1)
            )
            ax4.grid(axis="y", alpha=0.18)
            fig4.tight_layout()
            out4 = output_dir / "gap_to_baseline.png"
            fig4.savefig(out4, dpi=150)
            logger.info("Saved gap chart → %s", out4)
            plt.close(fig4)

    # ── 5. Per-dataset timelines ──────────────────────────────────────────────
    datasets = sorted(df["dataset"].unique())
    if datasets and not pl_data.empty:
        ncols = 2
        nrows = (len(datasets) + ncols - 1) // ncols
        fig5, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 6, nrows * 3.8), squeeze=False
        )

        # Baseline means are dataset-specific here (not cross-dataset).
        for idx, dataset in enumerate(datasets):
            ax = axes[idx // ncols][idx % ncols]
            ds_df = df[df["dataset"] == dataset].copy()
            ds_df["release_date"] = pd.to_datetime(ds_df["release_date"])
            ds_df = ds_df.sort_values("release_date")

            _ds_baseline_ys: list[float] = []

            for learner, color, ls, lw in [
                ("logreg", "#4878CF", "--", 1.5),
                ("xgboost", "#6ACC65", "--", 1.5),
                ("tabpfn", "#FF7F0E", "--", 1.5),
            ]:
                ld = ds_df[ds_df["learner"] == learner]
                if ld.empty:
                    continue
                val = ld["accuracy"].mean()
                pl_dates = ds_df[ds_df["learner"].str.startswith("skribe[")][
                    "release_date"
                ].dropna()
                if pl_dates.empty:
                    continue
                x_min = pl_dates.min()
                x_max = pl_dates.max()
                ax.plot(
                    [x_min, x_max],
                    [val, val],
                    color=color,
                    linewidth=lw,
                    linestyle=ls,
                    label=f"{learner} ({val:.3f})",
                )
                _ds_baseline_ys.append(val)

            # skribe — solid envelope line per provider (base), dotted for +web.
            # Final accuracy shown in legend label; no inline text annotations.
            pl_ds = ds_df[ds_df["learner"].str.startswith("skribe[")].reset_index(drop=True).copy()
            if "web_search" not in pl_ds.columns:
                pl_ds["web_search"] = False
            pl_ds["web_search"] = pl_ds["web_search"].fillna(False).astype(bool)
            pl_ds["release_date"] = pd.to_datetime(pl_ds["release_date"])
            ds_provider_styles = {
                "openai": {"color": "#D65F5F", "label": "OpenAI GPT"},
                "google": {"color": "#4285F4", "label": "Gemini"},
            }
            pl_ds_base = pl_ds[~pl_ds["web_search"]].copy()
            pl_ds_web = pl_ds[pl_ds["web_search"]].copy()

            ds_acc = ds_df["accuracy"]
            ax.set_ylim(max(0.0, ds_acc.min() - 0.08), min(1.05, ds_acc.max() + 0.08))

            for provider, grp in pl_ds_base.groupby("provider"):
                grp = grp.sort_values("release_date").reset_index(drop=True)
                pstyle = ds_provider_styles.get(provider, {"color": "#999", "label": provider})
                grp["best_so_far"] = grp["accuracy"].cummax()
                final_acc = grp["accuracy"].iloc[-1]
                ax.plot(
                    grp["release_date"],
                    grp["best_so_far"],
                    color=pstyle["color"],
                    linewidth=2.2,
                    linestyle="-",
                    label=f"skribe/{pstyle['label']} ({final_acc:.3f})",
                    zorder=3,
                )
                for _, row in grp.iterrows():
                    is_best = abs(row["accuracy"] - row["best_so_far"]) < 1e-9
                    ax.scatter(
                        row["release_date"],
                        row["accuracy"],
                        marker="o",
                        s=40,
                        color=pstyle["color"],
                        alpha=1.0 if is_best else 0.45,
                        zorder=4,
                    )

            for provider, grp in pl_ds_web.groupby("provider"):
                grp = grp.sort_values("release_date").reset_index(drop=True)
                pstyle = ds_provider_styles.get(provider, {"color": "#999", "label": provider})
                color = pstyle["color"]
                grp["best_so_far"] = grp["accuracy"].cummax()
                final_acc = grp["accuracy"].iloc[-1]
                ax.plot(
                    grp["release_date"],
                    grp["best_so_far"],
                    color=color,
                    linewidth=1.8,
                    linestyle=":",
                    label=f"skribe/{pstyle['label']} +web ({final_acc:.3f})",
                    zorder=3,
                    alpha=0.85,
                )
                for _, row in grp.iterrows():
                    is_best = abs(row["accuracy"] - row["best_so_far"]) < 1e-9
                    ax.scatter(
                        row["release_date"],
                        row["accuracy"],
                        marker="o",
                        s=40,
                        facecolors="white",
                        edgecolors=color,
                        linewidths=1.2,
                        alpha=0.85 if is_best else 0.4,
                        zorder=5,
                    )

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=5))
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
            ax.set_title(dataset, fontsize=11, fontweight="bold")
            _h, _lb = ax.get_legend_handles_labels()
            if _h:
                _h, _lb = zip(*sorted(zip(_h, _lb), key=lambda hl: (
                    -float(m.group(1)) if (m := _re.search(r"\((\d+\.\d+)\)", hl[1])) else 0.0
                )))
            ax.legend(_h, _lb, fontsize=7.5, loc="lower right")
            ax.tick_params(axis="x", rotation=25, labelsize=7.5)
            ax.grid(True, alpha=0.18)

        # Hide unused subplots.
        for idx in range(len(datasets), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig5.suptitle(
            "skribe accuracy per dataset across model generations\n"
            "Dashed = classical ML baselines",
            fontsize=13,
            y=1.01,
        )
        fig5.tight_layout()
        out5 = output_dir / "per_dataset_timelines.png"
        fig5.savefig(out5, dpi=150, bbox_inches="tight")
        logger.info("Saved per-dataset timelines → %s", out5)
        plt.close(fig5)


def print_summary_table(df: pd.DataFrame):
    """Print the full model × dataset accuracy grid.

    LLM models are rows (ordered by release date), datasets are columns, with a
    MEAN column on the right.  Baselines (tabpfn, logreg, xgboost) appear as
    rows below the LLMs.
    """
    print("\n## Model progression — accuracy grid (model × dataset)\n")

    all_datasets = sorted(df["dataset"].unique())

    # ── LLM rows ─────────────────────────────────────────────────────────────
    pl_rows = df[df["learner"].str.startswith("skribe[")].copy()
    if not pl_rows.empty:
        pl_rows["release_date"] = pd.to_datetime(pl_rows["release_date"])
        # Pivot: rows = llm_label (sorted by release_date), cols = dataset
        pivot = pl_rows.pivot_table(
            index=["llm_label", "release_date"], columns="dataset", values="accuracy"
        )
        pivot = pivot.sort_values("release_date")
        pivot.index = pivot.index.get_level_values("llm_label")
        pivot = pivot.reindex(columns=all_datasets)
        pivot["MEAN"] = pivot.mean(axis=1)
        print(pivot.to_string(float_format="%.3f"))
        print()

    # ── Baseline rows ─────────────────────────────────────────────────────────
    print("--- baselines ---")
    for learner in ("tabpfn", "logreg", "xgboost"):
        brows = df[df["learner"] == learner]
        if brows.empty:
            continue
        row_data = {ds: brows[brows["dataset"] == ds]["accuracy"].mean()
                    for ds in all_datasets}
        row_data["MEAN"] = brows["accuracy"].mean()
        row_series = pd.Series(row_data, name=learner)
        print(f"\n{learner}")
        print(row_series.to_string(float_format="%.3f"))
        print()
