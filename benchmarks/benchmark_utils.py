"""Shared utilities for the skribe model-progression benchmarks.

This module is imported by run_baselines.py, run_skribe.py, and collate.py.
It must NOT import from skribe itself — only the runner scripts do that.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import textalloc as ta
import yaml
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

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _lighten(hex_color: str, amount: float = 0.35) -> str:
    """Blend hex_color toward white by amount (0-1). Used to auto-derive a
    "+web" variant's color from its base model's color."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _build_model_progression(config: dict) -> list[dict]:
    """Expand config["models"] (base models only) into the flat, ordered
    list every script expects: each base model followed by its "-web"
    sibling (auto-derived) when supports_web is true.
    """
    progression = []
    for entry in config["models"]:
        release_date = entry["release_date"]
        if isinstance(release_date, str):
            release_date = datetime.strptime(release_date, "%Y-%m-%d").date()

        base = {
            "model_id": entry["model_id"],
            "label": entry["label"],
            "release_date": release_date,
            "family": entry["family"],
            "provider": entry["provider"],
            "color": entry["color"],
        }
        if "vertex_region" in entry:
            base["vertex_region"] = entry["vertex_region"]
        if "api_base" in entry:
            base["api_base"] = entry["api_base"]
        progression.append(base)

        if entry.get("supports_web"):
            web = dict(base)
            web["model_id"] = f"{entry['model_id']}-web"
            web["base_model_id"] = entry["model_id"]
            web["label"] = f"{entry['label']} +web"
            web["web_search"] = True
            web["color"] = entry.get("web_color") or _lighten(entry["color"])
            progression.append(web)

    return progression


def _build_default_datasets(config: dict) -> dict:
    """Expand config["datasets"] into the {name: spec_tuple} shape every
    script expects: (openml_name, version) for OpenML datasets, or
    (None, None, csv_path, target_col, description) for CSV-backed ones.
    csv_path (if given) is resolved relative to the benchmarks/ directory.
    """
    datasets = {}
    for entry in config["datasets"]:
        if "csv_path" in entry:
            csv_path = (Path(__file__).parent / entry["csv_path"]).resolve()
            datasets[entry["name"]] = (
                None,
                None,
                csv_path,
                entry["target_col"],
                entry.get("description", ""),
            )
        else:
            datasets[entry["name"]] = (entry["openml_name"], entry["version"])
    return datasets


_CONFIG = _load_config()

# Baseline learner names. Cache files for these set model_id to the learner
# name itself and store metrics under that same key, e.g. r["logreg"] when
# r["model_id"] == "logreg" — distinct from skribe cache files, where
# model_id is the LLM model ID and metrics live under the "skribe" key.
BASELINE_MODELS = {b["name"] for b in _CONFIG["baselines"]}

# name -> {label, color}, for scripts that need baseline display metadata
# (e.g. build_skribe_inspector.py).
BASELINE_META = {
    b["name"]: {"label": b["label"], "color": b["color"]} for b in _CONFIG["baselines"]
}

# Ordered oldest → newest. release_date is approximate; used as the x-axis value.
MODEL_PROGRESSION = _build_model_progression(_CONFIG)

DEFAULT_DATASETS = _build_default_datasets(_CONFIG)


def load_dataset(
    openml_name,
    version,
    max_rows: int | None,
    csv_path=None,
    target_col=None,
    description=None,
    require_description=True,
):
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        y_str = df[target_col].astype(str)
        X = df.drop(columns=[target_col])
        resolved_description = description or ""
    else:
        bunch = fetch_openml(name=openml_name, version=version, as_frame=True, parser="auto")
        X = bunch.data.copy()
        y_str = pd.Series(np.asarray(bunch.target)).astype(str)
        resolved_description = description or getattr(bunch, "DESCR", None) or ""
    if require_description and not resolved_description:
        raise ValueError(
            "Dataset has no description — the context pre-pass cannot run. "
            "Add a description string to the DEFAULT_DATASETS entry, "
            "or pass --skip-context to explicitly disable the pre-pass."
        )
    classes = {c: i for i, c in enumerate(sorted(y_str.unique()))}
    y = y_str.map(classes).astype(int)
    if max_rows and len(X) > max_rows:
        X = X.sample(max_rows, random_state=42)
        y = y.loc[X.index]
        y_str = y_str.loc[X.index]
    # y: int-coded (sorted(classes) order) for baselines/metrics, which need
    # numeric targets and must stay directly comparable across skribe and
    # baseline runs. y_str: the original string labels, in the same row
    # order/index — SkribeClassifier.fit() now does its own internal integer
    # encoding, so passing y_str (rather than the pre-encoded y) lets its
    # context pre-pass state the true class labels instead of having to
    # guess what an already-bare integer means.
    return (
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        classes,
        resolved_description,
        y_str.reset_index(drop=True),
    )


def _rich_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None, n_classes: int
) -> dict:
    """Compute a broad set of classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
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
    reasoning_effort: str | None = None,
    reasoning_mode: str | None = None,
) -> str:
    raw = f"{CACHE_SCHEMA}|{dataset}|{model_id}|{max_rows}|fe={fe_model or ''}|ws={web_search}"
    # Only mixed in when explicitly set, so pre-existing cache files (hashed
    # before reasoning_effort/reasoning_mode existed) keep resolving to the
    # same key.
    if reasoning_effort:
        raw += f"|re={reasoning_effort}"
    if reasoning_mode:
        raw += f"|rm={reasoning_mode}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _baseline_cache_key(dataset: str, max_rows: int | None, fe_model: str | None = None) -> str:
    # No fe_model: keep the exact pre-existing formula so already-cached
    # baseline runs (hashed without any "|fe=" component) stay valid.
    raw = f"{CACHE_SCHEMA}|baselines|{dataset}|{max_rows}"
    if fe_model:
        raw += f"|fe={fe_model}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def find_failed_skribe_cache_entries(cache_dir: Path) -> list[tuple[str, str]]:
    """Scan cache_dir for skribe cache files whose result errored out (timeout,
    rate-limit, etc.), returning (model_id, dataset) pairs suitable for
    re-running via run_openml_fit.py --model skribe --llm <model_id> --dataset
    <dataset> --no-cache.

    Uses the same is_error derivation as build_summary_df: an explicit
    "status" field when present (post reasoning_effort/status-field work),
    else the presence of an "error" key for cache files that predate it.
    Baseline-model cache files (logreg/xgboost/tabpfn) are skipped -- they
    have no --llm to retry with and run_all_models.sh already re-runs them
    unconditionally every time since they're cheap and cache-checked inside
    run_one_baseline itself.
    """
    import json

    pairs = []
    for f in sorted(Path(cache_dir).glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        model_id = d.get("model_id")
        dataset = d.get("dataset")
        if not model_id or not dataset or model_id in BASELINE_MODELS:
            continue
        skribe = d.get("skribe")
        if not isinstance(skribe, dict):
            continue
        is_error = skribe.get("status") == "error" if "status" in skribe else "error" in skribe
        if is_error:
            pairs.append((model_id, dataset))
    return pairs


def _xgb_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=4, verbosity=0)


def _tabpfn_classifier():
    try:
        from tabpfn_client import TabPFNClassifier, set_access_token
    except ImportError:
        return None
    token = os.environ.get("TABPFN_TOKEN")
    if token:
        set_access_token(token)
    return TabPFNClassifier()


def build_summary_df(results: list[dict]) -> pd.DataFrame:
    """Long-form DataFrame: one row per (dataset, learner) with all metrics.

    skribe learner names are qualified as "skribe[<llm-label>]" so they
    are never confused with the LLM model dimension.  Baseline learners
    (logreg, xgboost, tabpfn) appear once per dataset with no LLM association.

    Each cache dict has top-level "dataset" and "model_id" keys. For skribe
    runs, model_id is the LLM model ID and metrics live under the "skribe"
    key. For baseline runs, model_id is the learner name itself
    (logreg/xgboost/tabpfn) and metrics live under that same key, e.g.
    r["logreg"] when r["model_id"] == "logreg".
    """
    rows = []
    model_meta = {m["model_id"]: m for m in MODEL_PROGRESSION}

    for r in results:
        dataset = r.get("dataset")
        model_id = r.get("model_id")

        if not dataset or not model_id:
            continue

        if model_id in BASELINE_MODELS:
            m = r.get(model_id, {})
            is_error = m.get("status") == "error" if "status" in m else "error" in m
            if is_error:
                continue
            row = {
                "dataset": dataset,
                "model_id": None,
                "llm_label": None,
                "release_date": None,
                "family": None,
                "provider": None,
                "learner": model_id,
                "n_rows": r.get("n_rows"),
                "n_cols": r.get("n_cols"),
                "n_classes": r.get("n_classes"),
            }
            row.update({k: v for k, v in m.items() if k not in ("fit_time_s",)})
            row["fit_time_s"] = m.get("fit_time_s")
            rows.append(row)
            continue

        meta = model_meta.get(model_id, {})
        llm_label = meta.get("label", model_id)
        reasoning_mode = r.get("reasoning_mode")
        reasoning_effort = r.get("reasoning_effort")
        # A non-default reasoning_mode/effort (e.g. "pro", "xhigh") shares
        # model_id with the plain run, so without this suffix its rows would
        # collide with the plain run's in groupby/cummax (same provider, same
        # release_date) -- distinguishing the label keeps point annotations
        # from overlapping while `provider`/`learner` stay unchanged so both
        # still merge into the same best-so-far envelope, same treatment as
        # +web today.
        _variant_tags = [t for t in (reasoning_mode, reasoning_effort) if t]
        if _variant_tags:
            llm_label = f"{llm_label} ({', '.join(_variant_tags)})"

        if "skribe" in r:
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
                "reasoning_mode": reasoning_mode,
                "reasoning_effort": reasoning_effort,
                "learner": f"skribe[{llm_label}]",
                "n_rows": r.get("n_rows"),
                "n_cols": r.get("n_cols"),
                "n_classes": r.get("n_classes"),
            }
            is_error = m.get("status") == "error" if "status" in m else "error" in m
            if is_error:
                # A run that errored out (timeout, rate-limit, etc.) is
                # scored as a hard 0.0 rather than silently dropped — a
                # model that can't even produce a classifier for a dataset
                # must be penalized in aggregate charts, not excused from
                # the average by being absent from it.
                row["accuracy"] = 0.0
                row["error"] = m["error"]
                row["fit_time_s"] = None
            else:
                row.update({k: v for k, v in m.items() if k not in ("fit_time_s",)})
                row["fit_time_s"] = m.get("fit_time_s")
            rows.append(row)

    return pd.DataFrame(rows)


def _envelope_line_xy(release_dates: pd.Series, best_so_far: pd.Series, line_end: pd.Timestamp):
    """Build (x, y) ready for a steps-post ``ax.plot`` of a cumulative-max
    "best so far" envelope, handling two things a plain
    ``pd.concat([dates, best_so_far])`` misses:

    1. Same-day ties: when two rows share a release_date (e.g. two models
       launched the same day), a steps-post line has zero x-width between
       them, so it silently jumps to the higher value with no rendered
       vertical connector -- the lower point looks like it was never part
       of the series at all. Insert a zero-width vertical segment (repeat
       the date, old y then new y) so the jump renders exactly like it
       would between any two distinct dates.
    2. Right-edge extension: the "best so far" value is still current even
       after the newest model, so append line_end at the final value —
       otherwise the line stops dead at the last release date instead of
       reaching the plot's right border.
    """
    xs: list = []
    ys: list = []
    prev_y = None
    for x, y in zip(release_dates, best_so_far):
        if xs and x == xs[-1] and prev_y is not None and y != prev_y:
            xs.append(x)
            ys.append(prev_y)
        xs.append(x)
        ys.append(y)
        prev_y = y
    xs.append(line_end)
    ys.append(prev_y)
    return xs, ys


def plot_progression(df: pd.DataFrame, output_dir: Path):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
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

    # Right edge for the envelope lines below -- "best so far" is still
    # current today even though no newer model has been released, so the
    # line should reach the right edge of the plot rather than stopping
    # dead at the last release date. Extending only to "today" (rather than
    # to the same right edge the axis itself gets pinned to further down)
    # left a visible gap between the line's end and the plot border, since
    # the axis right edge includes extra slack margin beyond today.
    _today = datetime.now().date()
    _today_num = mdates.date2num(_today)
    _release_min_num = mdates.date2num(pl_data["release_date"].min())
    _line_end = pd.Timestamp(
        mdates.num2date(_today_num + 0.035 * (_today_num - _release_min_num)).date()
    )

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
            zorder=1,
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
            zorder=1,
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
            zorder=1,
        )

    # skribe — one solid line per provider (base models) + one dashed line
    # per provider (+web models).  Cumulative-max envelope so weaker models
    # don't cause visual dips.
    provider_styles = {
        "openai": {"color": "#D65F5F", "marker": "o", "label": "skribe OpenAI"},
        "google": {
            "color": "#4285F4",
            "marker": "s",
            "label": "skribe Google",
        },
        "anthropic": {
            "color": "#5FA05F",
            "marker": "^",
            "label": "skribe Anthropic",
        },
        "ollama": {
            "color": "#8E44AD",
            "marker": "D",
            "label": "skribe Ollama",
        },
    }
    if "web_search" not in pl_data.columns:
        pl_data["web_search"] = False

    _scatter_objects: list = []
    _label_texts: list = []
    _label_colors: list = []
    _annotation_targets: list = []

    if not pl_data.empty:
        # One merged envelope line per provider, combining base + web +
        # reasoning-mode runs into a single best-so-far series -- otherwise
        # a genuinely-best +web result (e.g. GPT-5.6 Sol +web) never shows
        # up in the "best OpenAI has ever done" story, stranded on its own
        # permanently-separate dotted line instead.
        for provider, grp in pl_data.groupby("provider"):
            # Same-day ties broken ascending by accuracy, not row order (which
            # is otherwise arbitrary, e.g. alphabetical on model_id) -- so
            # cummax() visits the lower value before the higher one instead
            # of maxing out immediately and hiding the lower tie from the
            # envelope line entirely.
            grp = grp.sort_values(["release_date", "accuracy"]).reset_index(drop=True)
            style = provider_styles.get(
                provider,
                {"color": "#999", "marker": "o", "label": f"skribe / {provider}"},
            )
            color = style["color"]

            # Cumulative-max envelope line (the "best so far" trajectory).
            # steps-post: flat at the previous best until the next model's exact
            # release date, then a vertical jump — a straight ax.plot() would
            # instead draw a diagonal ramp between release dates, which looks
            # like a real upward trend even when no better model existed yet.
            grp["best_so_far"] = grp["accuracy"].cummax()
            final_acc = grp["best_so_far"].iloc[-1]
            # Extend the envelope flat from the newest model's release date out
            # to the plot's right edge, so the line reaches the border instead
            # of stopping dead at the last release -- the "best so far" value
            # is still current even though no newer model has appeared. Also
            # inserts a visible vertical connector for same-day releases
            # (e.g. two models launched the same day) that a plain steps-post
            # line would otherwise jump across with zero rendered width.
            line_x, line_y = _envelope_line_xy(
                grp["release_date"], grp["best_so_far"], _line_end
            )
            ax.plot(
                line_x,
                line_y,
                color=color,
                linewidth=2.5,
                linestyle="-",
                drawstyle="steps-post",
                label=f"{style['label']} best ({final_acc:.3f})",
                zorder=3,
            )

            # Individual model dots — weaker models shown slightly faded.
            # Labels stay fully opaque regardless so every model name reads
            # equally well; only the dot communicates "not the best so far".
            for _, row in grp.iterrows():
                is_best = abs(row["accuracy"] - row["best_so_far"]) < 1e-9
                dot_alpha = 1.0 if is_best else 0.55
                sc = ax.scatter(
                    row["release_date"],
                    row["accuracy"],
                    marker="o",
                    s=60,
                    color=color,
                    alpha=dot_alpha,
                    zorder=4,
                )
                _scatter_objects.append(sc)
                # The GPT-5.6 Sol/Terra release cluster all landed the same
                # week, so every variant's label piles up in the same few
                # pixels at the plot's right edge -- keep only the
                # highest-accuracy one ("+web") labeled and leave the rest as
                # unlabeled dots rather than let textalloc fight over space
                # it doesn't have.
                if row["llm_label"].startswith("GPT-5.6") and row["llm_label"] != "GPT-5.6 Sol +web":
                    continue
                _label_texts.append(row["llm_label"])
                _label_colors.append(color)
                _annotation_targets.append(
                    (mdates.date2num(row["release_date"]), row["accuracy"])
                )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    fig.autofmt_xdate(rotation=30)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0.45, 1.0)
    # Pin the right edge to the same _line_end the envelope lines were
    # extended to above, so the lines visually reach the border instead of
    # stopping short of it.
    x_min, _ = ax.get_xlim()
    ax.set_xlim(x_min, _line_end)

    if _label_texts:
        # textalloc places each label at the first free candidate slot near
        # its anchor point (bounded by max_distance), drawing a thin leader
        # line back to the point only when the label actually had to move --
        # this structurally can't "fling" a label across the chart the way
        # adjustText's force physics could for a dense same-day cluster.
        _target_x = [t[0] for t in _annotation_targets]
        _target_y = [t[1] for t in _annotation_targets]
        # Marker radius in display pixels (s=60 -> area in points^2 for every
        # dot drawn above), so textalloc treats each dot as an actual disc to
        # avoid rather than a zero-size point -- otherwise a label can be
        # placed with its text centered right on top of another series' dot.
        _marker_radius_px = np.sqrt(60) / 2 * fig.dpi / 72.0
        ta.allocate(
            ax,
            _target_x,
            _target_y,
            _label_texts,
            x_scatter=_target_x,
            y_scatter=_target_y,
            scatter_sizes=[_marker_radius_px] * len(_target_x),
            y_lines=[[_by, _by] for _by in _baseline_ys] if _baseline_ys else None,
            x_lines=(
                [[min(_target_x), max(_target_x)] for _ in _baseline_ys]
                if _baseline_ys
                else None
            ),
            textsize=9.5,
            textcolor=_label_colors,
            linecolor=_label_colors,
            linewidth=0.7,
            min_distance=0.015,
            max_distance=0.4,
            margin=0.008,
            nbr_candidates=1000,
            avoid_label_lines_overlap=True,
            avoid_crossing_label_lines=True,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.85),
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

    handles, labels = (
        zip(*sorted(zip(handles, labels), key=_legend_sort_key)) if handles else (handles, labels)
    )
    if not lr_data.empty:
        # Anchor the legend's top edge just below the Logistic Regression
        # line instead of the axes' top-left corner — x stays at the left
        # edge (axes fraction 0), only y moves, via a transform that mixes
        # axes-fraction x with data-coordinate y.
        ax.legend(
            handles,
            labels,
            fontsize=10,
            loc="upper left",
            bbox_to_anchor=(0, lr_mean - 0.01),
            bbox_transform=ax.get_yaxis_transform(),
        )
    else:
        ax.legend(handles, labels, fontsize=10, loc="upper left")
    fig.tight_layout()
    out = output_dir / "model_progression.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved timeline chart → %s", out)
    plt.close(fig)

    # ── 2. Per-dataset heatmap: datasets × LLM models, skribe accuracy ──
    # Order columns by release date.
    col_order = (
        pl_data.sort_values("release_date")["llm_label"].tolist() if not pl_data.empty else None
    )
    pl_pivot = pl_df.pivot_table(index="dataset", columns="llm_label", values="accuracy")
    if col_order:
        pl_pivot = pl_pivot.reindex(columns=[c for c in col_order if c in pl_pivot.columns])
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
    _provider_bar_color = {
        "openai": "#D65F5F",
        "google": "#4285F4",
        "anthropic": "#5FA05F",
        "ollama": "#8E44AD",
    }
    _provider_bar_name = {
        "openai": "OpenAI GPT",
        "google": "Google Gemini",
        "anthropic": "Anthropic Claude",
        "ollama": "Ollama",
    }
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
            2,
            1,
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
                prov_name = _provider_bar_name.get(prov, prov)
                label_str = f"skribe / {prov_name}"
                bar = ax.bar(
                    i,
                    row["accuracy"],
                    color=color,
                    label=label_str if label_str not in _legend_seen else "_nolegend_",
                    zorder=3,
                )
                _legend_seen.add(label_str)
                ax.text(
                    i,
                    row["accuracy"] + 0.005,
                    f"{row['accuracy']:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            for lbl, val in baseline_means_3.items():
                color_map = {
                    "Logistic Regression": "#4878CF",
                    "XGBoost": "#6ACC65",
                    "TabPFN": "#FF7F0E",
                }
                c = color_map.get(lbl, "#888")
                ax.axhline(
                    val,
                    color=c,
                    linewidth=1.8,
                    linestyle="--",
                    label=f"{lbl}  ({val:.3f})",
                    zorder=4,
                )
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
        _draw_bar_row(
            ax3_bot, web_by_base, "With web search  (+web variants only; gaps = no web support)"
        )

        ax3_bot.set_xticks(x)
        ax3_bot.set_xticklabels(base_order, rotation=30, ha="right", fontsize=9)
        ax3_bot.set_xlabel("LLM model (oldest → newest)", fontsize=12)

        fig3.suptitle(
            "skribe vs baselines: mean accuracy by LLM generation",
            fontsize=13,
            y=1.01,
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
                    i,
                    row["gap"],
                    color=color,
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
            ax4.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
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
        fig5, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 3.8), squeeze=False)

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
                ax.plot(
                    [x_min, _line_end],
                    [val, val],
                    color=color,
                    linewidth=lw,
                    linestyle=ls,
                    label=f"{learner} ({val:.3f})",
                    zorder=1,
                )
                _ds_baseline_ys.append(val)

            # skribe — one solid envelope line per provider, combining base
            # and +web runs into a single best-so-far series. web vs
            # non-web is not visually distinguished on the line/markers;
            # it's only visible via hover/legend.
            pl_ds = ds_df[ds_df["learner"].str.startswith("skribe[")].reset_index(drop=True).copy()
            pl_ds["release_date"] = pd.to_datetime(pl_ds["release_date"])
            ds_provider_styles = {
                "openai": {"color": "#D65F5F", "label": "OpenAI GPT"},
                "google": {"color": "#4285F4", "label": "Gemini"},
                "anthropic": {"color": "#5FA05F", "label": "Claude"},
                "ollama": {"color": "#8E44AD", "label": "Ollama"},
            }

            ds_acc = ds_df["accuracy"]
            # Extra top margin (vs. a plain +0.08) so labels for a tightly
            # clustered top row -- e.g. several near-100%-accuracy models on
            # an easy dataset -- have vertical room to stack instead of
            # fighting for space right at the axes' edge.
            ax.set_ylim(max(0.0, ds_acc.min() - 0.08), min(1.15, ds_acc.max() + 0.18))

            _ds_label_texts: list = []
            _ds_label_colors: list = []
            _ds_target_x: list = []
            _ds_target_y: list = []

            for provider, grp in pl_ds.groupby("provider"):
                # Same-day ties broken ascending by accuracy, not row order
                # (which is otherwise arbitrary, e.g. alphabetical on
                # model_id) -- so cummax() visits the lower value before the
                # higher one instead of maxing out immediately and hiding
                # the lower tie from the envelope line entirely.
                grp = grp.sort_values(["release_date", "accuracy"]).reset_index(drop=True)
                pstyle = ds_provider_styles.get(provider, {"color": "#999", "label": provider})
                grp["best_so_far"] = grp["accuracy"].cummax()
                final_acc = grp["best_so_far"].iloc[-1]
                # Extend flat to the same right edge as the aggregate chart
                # above, so the line reaches the border instead of stopping
                # dead at the last release, and insert a visible vertical
                # connector for same-day releases instead of a zero-width
                # jump that leaves the lower point looking disconnected.
                line_x, line_y = _envelope_line_xy(
                    grp["release_date"], grp["best_so_far"], _line_end
                )
                ax.plot(
                    line_x,
                    line_y,
                    color=pstyle["color"],
                    linewidth=2.2,
                    linestyle="-",
                    drawstyle="steps-post",
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
                    _ds_label_texts.append(row["llm_label"])
                    _ds_label_colors.append(pstyle["color"])
                    _ds_target_x.append(mdates.date2num(row["release_date"]))
                    _ds_target_y.append(row["accuracy"])

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=5))
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
            # Pin the right edge to the same _line_end the envelope lines
            # were extended to above, so the lines visually reach the
            # border instead of stopping short of it.
            _ds_x_min, _ = ax.get_xlim()
            ax.set_xlim(_ds_x_min, _line_end)

            if _ds_label_texts:
                # Same textalloc-based placement as the aggregate chart --
                # labels default to sitting near their dot and only move
                # when actually crowded, with a thin leader line drawn back
                # when they do.
                _ds_marker_radius_px = np.sqrt(40) / 2 * fig5.dpi / 72.0
                ta.allocate(
                    ax,
                    _ds_target_x,
                    _ds_target_y,
                    _ds_label_texts,
                    x_scatter=_ds_target_x,
                    y_scatter=_ds_target_y,
                    scatter_sizes=[_ds_marker_radius_px] * len(_ds_target_x),
                    y_lines=(
                        [[_by, _by] for _by in _ds_baseline_ys] if _ds_baseline_ys else None
                    ),
                    x_lines=(
                        [[min(_ds_target_x), max(_ds_target_x)] for _ in _ds_baseline_ys]
                        if _ds_baseline_ys
                        else None
                    ),
                    textsize=6.5,
                    textcolor=_ds_label_colors,
                    linecolor=_ds_label_colors,
                    linewidth=0.6,
                    min_distance=0.02,
                    max_distance=0.6,
                    margin=0.008,
                    nbr_candidates=2000,
                    avoid_label_lines_overlap=True,
                    avoid_crossing_label_lines=True,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                )

            ax.set_title(dataset, fontsize=11, fontweight="bold")
            _h, _lb = ax.get_legend_handles_labels()
            if _h:
                _h, _lb = zip(
                    *sorted(
                        zip(_h, _lb),
                        key=lambda hl: (
                            -float(m.group(1))
                            if (m := _re.search(r"\((\d+\.\d+)\)", hl[1]))
                            else 0.0
                        ),
                    )
                )
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

    # ── 6. Overview ranking: every baseline + every skribe variant, sorted ──
    _overview_rows = []
    for _name, _bdata, _color in [
        ("Logistic Regression", lr_data, "#4878CF"),
        ("XGBoost", xgb_data, "#6ACC65"),
        ("TabPFN", tabpfn_data, "#FF7F0E"),
    ]:
        if not _bdata.empty:
            _overview_rows.append(
                {"label": _name, "accuracy": _bdata["accuracy"].mean(), "color": _color, "kind": "baseline"}
            )
    for (_label, _provider), _grp in pl_df.groupby(["llm_label", "provider"]):
        _overview_rows.append(
            {
                "label": _label,
                "accuracy": _grp["accuracy"].mean(),
                "color": provider_styles.get(_provider, {"color": "#999"})["color"],
                "kind": "skribe",
            }
        )

    if _overview_rows:
        overview_df = pd.DataFrame(_overview_rows).sort_values("accuracy", ascending=False)
        n_bars = len(overview_df)
        fig6, ax6 = plt.subplots(figsize=(10, max(6, n_bars * 0.4)))
        y_pos = np.arange(n_bars)[::-1]
        # Baselines get a hatch pattern so they read as structurally different
        # from skribe/LLM variants at a glance, not just by color.
        for y, (_, row) in zip(y_pos, overview_df.iterrows()):
            ax6.barh(
                y,
                row["accuracy"],
                color=row["color"],
                hatch="//" if row["kind"] == "baseline" else "",
                edgecolor="white" if row["kind"] == "baseline" else "none",
                linewidth=0.5,
                zorder=3,
            )
            ax6.text(
                row["accuracy"] + 0.005,
                y,
                f"{row['accuracy']:.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(overview_df["label"].tolist(), fontsize=9.5)
        ax6.set_xlim(0, min(1.08, overview_df["accuracy"].max() + 0.08))
        ax6.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax6.set_xlabel(f"Mean accuracy ({n_datasets} datasets)", fontsize=12)
        ax6.set_title(
            "Overview: every baseline and skribe model/variant, best first\n"
            "// hatch = classical ML baseline  ·  solid = skribe (LLM) variant",
            fontsize=12,
        )
        ax6.grid(axis="x", alpha=0.18)
        fig6.tight_layout()
        out6 = output_dir / "overview_ranking.png"
        fig6.savefig(out6, dpi=150)
        logger.info("Saved overview ranking → %s", out6)
        plt.close(fig6)


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
        row_data = {ds: brows[brows["dataset"] == ds]["accuracy"].mean() for ds in all_datasets}
        row_data["MEAN"] = brows["accuracy"].mean()
        row_series = pd.Series(row_data, name=learner)
        print(f"\n{learner}")
        print(row_series.to_string(float_format="%.3f"))
        print()
