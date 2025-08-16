"""
Simple, code-first benchmark runner for scikit‑learn compatible estimators.
V1 goals: minimal deps, **no CLI**, strong **resumability**, and sensible defaults.

✅ Includes **promptlearn** by default (if installed) alongside sklearn baselines.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import logging
import sys
import time
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_string_dtype

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("promptlearn.benchmark")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_type: str  # "classification" | "regression"
    target: str
    metrics: List[str]
    train_csv: Path
    test_csv: Path
    knowledge_mode: str = "closed_book"  # or "kb_join" (v1 offline only)
    oracle_jsonl: Optional[Path] = None
    random_seed: int = 42

    @staticmethod
    def from_dir(task_dir: Union[str, Path]) -> "TaskSpec":
        task_dir = Path(task_dir)
        meta_path = task_dir / "meta.yaml"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        tid = meta.get("id", task_dir.name)
        ttype = meta["task_type"]
        target = meta["target"]
        metrics = list(
            meta.get("metrics", ["accuracy" if ttype == "classification" else "rmse"])
        )
        splits = meta.get("splits", {})
        train_csv = task_dir / splits.get("train", "core/train.csv")
        test_csv = task_dir / splits.get("test", "core/test.csv")
        knowledge_mode = meta.get("knowledge_mode", "closed_book")
        oracle_jsonl = (
            task_dir / "oracle.jsonl"
            if knowledge_mode == "kb_join" and (task_dir / "oracle.jsonl").exists()
            else None
        )
        random_seed = int(meta.get("random_seed", 42))
        return TaskSpec(
            task_id=tid,
            task_type=ttype,
            target=target,
            metrics=metrics,
            train_csv=train_csv,
            test_csv=test_csv,
            knowledge_mode=knowledge_mode,
            oracle_jsonl=oracle_jsonl,
            random_seed=random_seed,
        )

    def fingerprint(self) -> str:
        """Hash task identity from meta + core files for resumability."""
        h = hashlib.sha1()
        for p in [self.train_csv, self.test_csv]:
            with open(p, "rb") as f:
                h.update(hashlib.sha1(f.read()).digest())
        meta_path = self.train_csv.parent.parent / "meta.yaml"
        with open(meta_path, "rb") as f:
            h.update(hashlib.sha1(f.read()).digest())
        if self.oracle_jsonl and self.oracle_jsonl.exists():
            with open(self.oracle_jsonl, "rb") as f:
                h.update(hashlib.sha1(f.read()).digest())
        return h.hexdigest()[:16]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    import_path: Optional[str] = None  # e.g., "sklearn.linear_model.LogisticRegression"
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    instance: Optional[Any] = (
        None  # alternatively, pass an already-constructed estimator
    )

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        h.update(
            json.dumps(
                {
                    "name": self.name,
                    "import_path": self.import_path,
                    "params": self.params,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        return h.hexdigest()[:16]

    def build(self):
        if self.instance is not None:
            return self.instance
        assert self.import_path, "Either import_path or instance must be provided"
        module, cls = self.import_path.rsplit(".", 1)
        mod = importlib.import_module(module)
        Est = getattr(mod, cls)
        params = dict(self.params) if self.params else {}

        # Allow passing a specific LLM/backend hint for PromptLearn estimators
        model_hint = params.pop("__pl_model_hint__", None)
        if model_hint is not None and str(self.import_path).startswith("promptlearn"):
            try:
                est = Est(model=model_hint, **params)
                logger.info(
                    f"[PromptLearn] instantiated with model='{model_hint}' for {self.name}"
                )
                return est
            except TypeError as e:
                logger.warning(
                    f"[PromptLearn] Could not pass model hint via `model=` ({e}); falling back to default constructor"
                )
        est = Est(**params)
        return est


# -----------------------------------------------------------------------------
# Defaults & helpers (includes promptlearn by default if available)
# -----------------------------------------------------------------------------


def _discover_first(candidates: List[str]) -> Optional[str]:
    for path in candidates:
        try:
            module, cls = path.rsplit(".", 1)
            mod = importlib.import_module(module)
            getattr(mod, cls)
            return path
        except Exception:
            continue
    return None


def make_sklearn(name: str, import_path: str, **params) -> ModelSpec:
    return ModelSpec(name=name, import_path=import_path, params=params)


def make_promptlearn(
    kind: str = "classifier", name: Optional[str] = None, **params
) -> Optional[ModelSpec]:
    """Return a ModelSpec for a promptlearn estimator if present; else None.
    Tries several likely import paths and uses the first that imports.
    """
    if kind not in {"classifier", "regressor"}:
        raise ValueError("kind must be 'classifier' or 'regressor'")

    if kind == "classifier":
        candidates = [
            "promptlearn.PromptClassifier",
            "promptlearn.estimators.PromptClassifier",
            "promptlearn.models.PromptClassifier",
            "promptlearn.classifier.PromptClassifier",
            "promptlearn.PromptLearnClassifier",
        ]
        default_name = "PromptLearnClf"
    else:
        candidates = [
            "promptlearn.PromptRegressor",
            "promptlearn.estimators.PromptRegressor",
            "promptlearn.models.PromptRegressor",
            "promptlearn.regressor.PromptRegressor",
            "promptlearn.PromptLearnRegressor",
        ]
        default_name = "PromptLearnReg"

    import_path = _discover_first(candidates)
    if not import_path:
        return None
    return ModelSpec(name=name or default_name, import_path=import_path, params=params)


def make_promptlearn_variant(
    kind: str, llm_name: str, name: Optional[str] = None, **params
) -> Optional[ModelSpec]:
    """Construct a PromptLearn ModelSpec pinned to a specific underlying LLM/backend.
    We pass a special key `__pl_model_hint__` that ModelSpec.build consumes and tries
    several likely constructor kwarg names (model/model_name/llm/engine/...).
    """
    ms = make_promptlearn(kind=kind, name=name, **params)
    if ms is None:
        return None
    new_params = dict(ms.params)
    new_params["__pl_model_hint__"] = llm_name
    # Also allow custom display name if provided
    display_name = name or ms.name
    return dataclasses.replace(ms, name=display_name, params=new_params)


def default_models_for_task_type(task_type: str) -> List[ModelSpec]:
    models: List[ModelSpec] = []
    # Optional env var: PLBENCH_LLM_MODELS="gpt-4o,gpt-5" to get multiple PromptLearn variants
    llm_env = os.getenv("PLBENCH_LLM_MODELS", "").strip()
    llm_variants = [s.strip() for s in llm_env.split(",") if s.strip()]

    if task_type == "classification":
        models.append(
            make_sklearn(
                "LogReg", "sklearn.linear_model.LogisticRegression", max_iter=1000
            )
        )
        models.append(
            make_sklearn(
                "RF",
                "sklearn.ensemble.RandomForestClassifier",
                n_estimators=300,
                random_state=0,
            )
        )
        if llm_variants:
            for llm in llm_variants:
                pl = make_promptlearn_variant(
                    "classifier", llm_name=llm, name=f"PromptLearnClf[{llm}]"
                )
                if pl:
                    models.append(pl)
        else:
            pl = make_promptlearn("classifier")
            if pl:
                models.append(pl)
    elif task_type == "regression":
        models.append(make_sklearn("LinReg", "sklearn.linear_model.LinearRegression"))
        models.append(
            make_sklearn(
                "RFReg",
                "sklearn.ensemble.RandomForestRegressor",
                n_estimators=300,
                random_state=0,
            )
        )
        if llm_variants:
            for llm in llm_variants:
                pl = make_promptlearn_variant(
                    "regressor", llm_name=llm, name=f"PromptLearnReg[{llm}]"
                )
                if pl:
                    models.append(pl)
        else:
            pl = make_promptlearn("regressor")
            if pl:
                models.append(pl)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
    return models


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_row_id(df: pd.DataFrame) -> np.ndarray:
    if "row_id" in df.columns:
        return df["row_id"].astype("int64").values
    return df.index.astype("int64").values


def _split_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    assert target in df.columns, f"Target column '{target}' not in DataFrame"
    y = df[target]
    drop_cols = [target]
    # Never treat bookkeeping columns as features
    if "row_id" in df.columns:
        drop_cols.append("row_id")
    X = df.drop(columns=drop_cols)
    return X, y


def _maybe_wrap_preprocessor(
    est: Any, model_spec: ModelSpec, X_train: pd.DataFrame, task_type: str
):
    """If estimator is an sklearn model and X has categorical/text columns, wrap with
    a ColumnTransformer(OneHotEncoder) → Pipeline(est). PromptLearn models are
    left untouched to keep raw text available.
    """
    # Heuristic: only wrap sklearn models
    is_sklearn = isinstance(
        model_spec.import_path, str
    ) and model_spec.import_path.startswith("sklearn.")
    if not is_sklearn:
        return est, {"pre": None}

    # Identify categorical columns
    cat_cols = [
        c
        for c in X_train.columns
        if is_categorical_dtype(X_train[c])
        or is_string_dtype(X_train[c])
        or X_train[c].dtype == object
    ]
    if not cat_cols:
        return est, {"pre": None}

    num_cols = [c for c in X_train.columns if c not in cat_cols]

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    pipe = Pipeline(steps=[("pre", pre), ("est", est)])
    return pipe, {"pre_cat_cols": cat_cols, "pre_num_cols": num_cols}


def _is_sklearn_model(model_spec: ModelSpec) -> bool:
    return isinstance(
        model_spec.import_path, str
    ) and model_spec.import_path.startswith("sklearn.")


def _model_task_category(model_spec: ModelSpec) -> str:
    """Heuristics to infer whether a model is a classifier or regressor.
    Returns one of {"classification", "regression", "both", "unknown"}.
    - For sklearn classes, we import the class and check ClassifierMixin/RegressorMixin.
    - For others, we fall back to name/path cues.
    """
    try:
        if _is_sklearn_model(model_spec) and model_spec.import_path:
            from sklearn.base import ClassifierMixin, RegressorMixin

            module, cls = model_spec.import_path.rsplit(".", 1)
            Cls = getattr(importlib.import_module(module), cls)
            is_clf = issubclass(Cls, ClassifierMixin)
            is_reg = issubclass(Cls, RegressorMixin)
            if is_clf and is_reg:
                return "both"
            if is_clf:
                return "classification"
            if is_reg:
                return "regression"
    except Exception:
        pass

    s = (model_spec.name + " " + (model_spec.import_path or "")).lower()
    clf_tokens = ["clf", "classifier", "logreg", "logistic"]
    reg_tokens = ["reg", "regressor", "linreg", "ridge", "lasso"]
    is_clf = any(tok in s for tok in clf_tokens)
    is_reg = any(tok in s for tok in reg_tokens)
    if is_clf and is_reg:
        return "both"
    if is_clf:
        return "classification"
    if is_reg:
        return "regression"
    return "unknown"


def _is_model_compatible_with_task(model_spec: ModelSpec, task_type: str) -> bool:
    cat = _model_task_category(model_spec)
    if cat == "both" or cat == "unknown":
        return True  # be permissive if unknown
    return cat == task_type


# -----------------------------------------------------------------------------
# Core runner (CSV append for bulletproof resumability in v1)
# -----------------------------------------------------------------------------


def run_benchmark(
    task_dirs: Iterable[Union[str, Path]],
    models: Optional[Iterable[ModelSpec]] = None,
    out_dir: Union[str, Path] = "runs",
    seed: int = 42,
    resume: bool = True,
    chunk_size: int = 50_000,
    return_kind: str = "wide",  # "wide" | "split" | "all"
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Run a grid of (task × model) with resumability.
    If `models` is None, choose sensible defaults per task type, attempting to include
    **promptlearn** estimators by default when available.

    Returns tables according to `return_kind`:
      - "wide"  → a single Problem×(Model,Metric) mixed table (default)
      - "split" → (classification_table, regression_table)
      - "all"   → {"wide": df_wide, "classification": df_cls, "regression": df_reg}
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results_long: List[Dict[str, Any]] = []

    for task_path in task_dirs:
        task = TaskSpec.from_dir(task_path)
        task_hash = task.fingerprint()
        logger.info(
            f"Task: {task.task_id} [{task_hash}] :: {task.task_type} → target='{task.target}'"
        )

        # Load data once per task
        train_df = pd.read_csv(task.train_csv)
        test_df = pd.read_csv(task.test_csv)
        X_train, y_train = _split_X_y(train_df, task.target)
        X_test, y_test = _split_X_y(test_df, task.target)
        test_row_ids = _ensure_row_id(test_df)

        models_for_task = (
            list(models)
            if models is not None
            else default_models_for_task_type(task.task_type)
        )

        for m in models_for_task:
            m_hash = m.fingerprint()

            # Skip clearly incompatible (model, task) pairs (e.g., classifier on regression)
            if not _is_model_compatible_with_task(m, task.task_type):
                logger.info(
                    f"[SKIP] {m.name} on {task.task_id}: incompatible with task_type='{task.task_type}'"
                )
                continue

            # Empty TRAIN split handling
            if len(X_train) == 0:
                if _is_sklearn_model(m):
                    run_dir = out_root / task.task_id / m.name
                    (run_dir / "fit").mkdir(parents=True, exist_ok=True)
                    (run_dir / "predict").mkdir(parents=True, exist_ok=True)
                    metric_path = run_dir / "metrics.json"
                    json.dump(
                        {
                            "task_id": task.task_id,
                            "model": m.name,
                            "fit_seconds": 0.0,
                            "predict_seconds": 0.0,
                            "n_test": int(len(X_test)),
                            "metrics": {},
                            "skipped_reason": "empty_train_split",
                            "completed_at": _now_iso(),
                        },
                        open(metric_path, "w"),
                        indent=2,
                    )
                    logger.info(
                        f"[SKIP] {m.name} on {task.task_id}: empty train split — sklearn requires fit"
                    )
                    continue
                # For PromptLearn/non-sklearn: fall through to normal fit() which can zero-shot compile from schema.

                logger.info(
                    f"[FIT] {m.name} on {task.task_id}: empty-train; proceeding with schema-only fit"
                )

            run_dir = out_root / task.task_id / m.name
            fit_dir = run_dir / "fit"
            pred_dir = run_dir / "predict"
            metric_path = run_dir / "metrics.json"

            run_dir.mkdir(parents=True, exist_ok=True)
            fit_dir.mkdir(parents=True, exist_ok=True)
            pred_dir.mkdir(parents=True, exist_ok=True)

            # Fit caching
            fit_hash_path = fit_dir / "fit_hash.json"
            model_pkl_path = fit_dir / "model.pkl"
            fit_hash_obj = {
                "task_hash": task_hash,
                "model": {
                    "name": m.name,
                    "import_path": m.import_path,
                    "params": m.params,
                },
                "seed": seed,
            }
            fit_hash = hashlib.sha1(
                json.dumps(fit_hash_obj, sort_keys=True).encode("utf-8")
            ).hexdigest()

            need_fit = True
            if resume and model_pkl_path.exists() and fit_hash_path.exists():
                try:
                    old_hash = json.load(open(fit_hash_path, "r"))["fit_hash"]
                    need_fit = old_hash != fit_hash
                except Exception:
                    need_fit = True

            if need_fit:
                logger.info(f"[FIT] {m.name} on {task.task_id}…")
                np.random.seed(seed)
                est = m.build()
                # Coerce regression targets to numeric if needed
                if task.task_type == "regression" and not is_numeric_dtype(y_train):
                    y_train = pd.to_numeric(y_train, errors="raise")
                # Auto-wrap sklearn models with categorical/text preprocessing
                est, preinfo = _maybe_wrap_preprocessor(est, m, X_train, task.task_type)
                if hasattr(est, "set_params"):
                    try:
                        est.set_params(random_state=seed)
                    except Exception:
                        pass
                t0 = time.time()
                est.fit(X_train, y_train)
                fit_s = time.time() - t0
                joblib.dump(est, model_pkl_path)
                json.dump(
                    {"fit_hash": fit_hash, "fit_seconds": fit_s, "at": _now_iso()},
                    open(fit_hash_path, "w"),
                )
                logger.info(f"[FIT] done in {fit_s:.2f}s; cached → {model_pkl_path}")
            else:
                est = joblib.load(model_pkl_path)
                logger.info(f"[FIT] reused cache → {model_pkl_path}")

            # Capture estimator meta (useful for PromptLearn variants)
            est_meta = {}
            if not _is_sklearn_model(m):
                est_meta = {
                    "estimator_class": m.import_path,
                    "llm_model": getattr(est, "model", None),
                    "verbose": getattr(est, "verbose", None),
                    "max_train_rows": getattr(est, "max_train_rows", None),
                }

            # Predict with resumability using CSV append (most robust)
            preds_csv = pred_dir / "predictions.csv"
            progress_path = pred_dir / "progress.json"

            # If not resuming, clear any prior prediction artifacts to avoid double-appends
            if not resume:
                try:
                    if preds_csv.exists():
                        preds_csv.unlink()
                    if progress_path.exists():
                        progress_path.unlink()
                except Exception:
                    pass

            done_ids: set = set()
            if resume and preds_csv.exists():
                try:
                    existing = pd.read_csv(preds_csv)
                    done_ids = set(existing["row_id"].astype("int64").tolist())
                    logger.info(
                        f"[PREDICT] resuming; found {len(done_ids)} rows already predicted"
                    )
                except Exception as e:
                    logger.warning(f"Could not resume predictions: {e}")

            all_ids = test_row_ids
            if len(done_ids) > 0:
                mask = ~np.isin(all_ids, np.fromiter(done_ids, dtype=np.int64))
                remaining_idx = np.where(mask)[0]
            else:
                remaining_idx = np.arange(len(all_ids))

            n_total = len(all_ids)
            n_remaining = len(remaining_idx)
            logger.info(
                f"[PREDICT] {m.name} on {task.task_id}: {n_remaining}/{n_total} rows remaining"
            )

            # Short-circuit: empty test split → create empty predictions and skip scoring
            if n_total == 0:
                # Ensure empty predictions artifact exists for consistency
                pd.DataFrame(columns=["row_id", "y_pred"]).to_csv(
                    preds_csv, index=False
                )
                json.dump(
                    {
                        "last_chunk_rows": 0,
                        "done_rows": 0,
                        "total_rows": 0,
                        "at": _now_iso(),
                    },
                    open(progress_path, "w"),
                )
                metrics_obj = {
                    "task_id": task.task_id,
                    "model": m.name,
                    "fit_seconds": json.load(open(fit_hash_path, "r")).get(
                        "fit_seconds", None
                    ),
                    "predict_seconds": 0.0,
                    "n_test": 0,
                    "metrics": {},
                    "skipped_reason": "empty_test_split",
                    "completed_at": _now_iso(),
                    "estimator": est_meta,
                }
                json.dump(metrics_obj, open(metric_path, "w"), indent=2)
                logger.info(
                    f"[SKIP] {m.name} on {task.task_id}: empty test split — no metrics computed"
                )
                continue

            # Predict & append with resumability
            t_pred0 = time.time()
            for start in range(0, len(remaining_idx), chunk_size):
                end = min(start + chunk_size, len(remaining_idx))
                sel = remaining_idx[start:end]
                X_chunk = X_test.iloc[sel]
                rows_chunk = all_ids[sel]
                y_hat = est.predict(X_chunk)
                out_df = pd.DataFrame({"row_id": rows_chunk, "y_pred": y_hat})
                mode = "a" if preds_csv.exists() else "w"
                header = not preds_csv.exists()
                out_df.to_csv(preds_csv, mode=mode, header=header, index=False)
                json.dump(
                    {
                        "last_chunk_rows": int(len(out_df)),
                        "done_rows": int(end),
                        "total_rows": int(len(remaining_idx)),
                        "at": _now_iso(),
                    },
                    open(progress_path, "w"),
                )
                logger.info(
                    f"[PREDICT] rows {start}–{end} / {len(remaining_idx)} appended"
                )

            t_pred = time.time() - t_pred0

            # Score
            pred_df = (
                pd.read_csv(preds_csv)
                .drop_duplicates(subset=["row_id"], keep="last")
                .sort_values("row_id")
            )
            order = np.argsort(all_ids)
            y_true = y_test.values[order]
            id_to_pred = dict(
                zip(pred_df["row_id"].astype(np.int64), pred_df["y_pred"])
            )
            try:
                y_pred = np.array([id_to_pred[int(r)] for r in all_ids[order]])
            except KeyError:
                missing = set(all_ids) - set(pred_df["row_id"].astype(np.int64))
                extra = set(pred_df["row_id"].astype(np.int64)) - set(all_ids)
                raise AssertionError(
                    f"Prediction/ground-truth row_id mismatch. Missing={len(missing)} Extra={len(extra)}. "
                    f"Consider deleting {preds_csv} or run with resume=False"
                )

            metrics: Dict[str, float] = {}
            if task.task_type == "classification":
                from sklearn.metrics import accuracy_score, f1_score

                metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                if "macro_f1" in task.metrics:
                    metrics["macro_f1"] = float(
                        f1_score(y_true, y_pred, average="macro")
                    )
            elif task.task_type == "regression":
                y_true = y_true.astype(float)
                y_pred = y_pred.astype(float)
                metrics["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            else:
                raise ValueError(f"Unsupported task_type: {task.task_type}")

            metrics_obj = {
                "task_id": task.task_id,
                "model": m.name,
                "fit_seconds": json.load(open(fit_hash_path, "r")).get(
                    "fit_seconds", None
                ),
                "predict_seconds": t_pred,
                "n_test": int(len(y_true)),
                "metrics": metrics,
                "completed_at": _now_iso(),
                "estimator": est_meta,
            }
            json.dump(metrics_obj, open(metric_path, "w"), indent=2)
            for k, v in metrics.items():
                results_long.append(
                    {"task_id": task.task_id, "model": m.name, "metric": k, "value": v}
                )
            logger.info(f"[SCORE] {m.name} on {task.task_id}: {metrics}")

    # Build tables
    df_long = pd.DataFrame(results_long)
    if df_long.empty:
        logger.warning("No results produced. Check inputs.")
        return pd.DataFrame()

    # Unified (mixed) table keeps all models/metrics → will show NaN for not-applicable cells
    df_wide = df_long.pivot_table(
        index="task_id", columns=["model", "metric"], values="value"
    )
    df_wide = df_wide.sort_index(axis=1, level=0)

    # Also write clearer per-task-type tables to avoid NaNs from not-applicable models
    df_cls = (
        df_long[df_long["metric"].isin(["accuracy", "macro_f1"])]
        .pivot_table(index="task_id", columns=["model", "metric"], values="value")
        .sort_index(axis=1, level=0)
    )
    df_reg = (
        df_long[df_long["metric"].isin(["rmse"])]
        .pivot_table(index="task_id", columns=["model", "metric"], values="value")
        .sort_index(axis=1, level=0)
    )

    # Save aggregate artifacts at root
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    df_long.to_csv(out_root / f"benchmark_long_{stamp}.csv", index=False)
    df_wide.to_csv(out_root / f"benchmark_wide_{stamp}.csv")
    if not df_cls.empty:
        df_cls.to_csv(out_root / f"benchmark_cls_wide_{stamp}.csv")
    if not df_reg.empty:
        df_reg.to_csv(out_root / f"benchmark_reg_wide_{stamp}.csv")

    logger.info(
        "Wrote wide tables: benchmark_wide_*.csv, benchmark_cls_wide_*.csv, benchmark_reg_wide_*.csv"
    )

    # Also return for interactive use
    if return_kind == "split":
        return df_cls, df_reg
    if return_kind == "all":
        return {"wide": df_wide, "classification": df_cls, "regression": df_reg}
    return df_wide
