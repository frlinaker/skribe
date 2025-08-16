"""
Compact benchmark runner for scikit‑learn-compatible estimators with resumability.
V1: no CLI, few deps, sensible defaults, and PromptLearn support by default.
"""

from __future__ import annotations

import dataclasses, hashlib, importlib, json, logging, os, sys, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Literal, overload

import joblib, numpy as np, pandas as pd, yaml
import pandas.api.types as ptypes

# --- Logging -----------------------------------------------------------------
logger = logging.getLogger("promptlearn.benchmark")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# --- Data classes -------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_type: str
    target: str
    metrics: List[str]
    train_csv: Path
    test_csv: Path

    @staticmethod
    def from_dir(task_dir: Union[str, Path]) -> "TaskSpec":
        p = Path(task_dir)
        meta_path = p / "meta.yaml"
        meta = yaml.safe_load(open(meta_path, "r", encoding="utf-8"))
        tid = meta.get("id", p.name)
        ttype = meta["task_type"]
        tgt = meta["target"]
        metrics = list(
            meta.get("metrics", ["accuracy" if ttype == "classification" else "rmse"])
        )
        splits = meta.get("splits", {})
        tr = p / splits.get("train", "core/train.csv")
        te = p / splits.get("test", "core/test.csv")
        return TaskSpec(tid, ttype, tgt, metrics, tr, te)

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for fp in [
            self.train_csv,
            self.test_csv,
            self.train_csv.parent.parent / "meta.yaml",
        ]:
            with open(fp, "rb") as f:
                h.update(hashlib.sha1(f.read()).digest())
        return h.hexdigest()[:16]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    import_path: Optional[str] = None
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def build(self):
        assert self.import_path, "Either import_path or instance must be provided"
        module, cls = self.import_path.rsplit(".", 1)
        Est = getattr(importlib.import_module(module), cls)
        params = dict(self.params) if self.params else {}
        hint = params.pop("__pl_model_hint__", None)
        if hint is not None and str(self.import_path).startswith("promptlearn"):
            try:
                est = Est(model=hint, **params)
                logger.info(
                    f"[PromptLearn] instantiated with model='{hint}' for {self.name}"
                )
                return est
            except TypeError:
                logger.warning(
                    "[PromptLearn] Could not pass model hint via model=; falling back to default constructor"
                )
        return Est(**params)


# --- PromptLearn factory (lean) -----------------------------------------------


def make_sklearn(name: str, import_path: str, **params) -> ModelSpec:
    return ModelSpec(name=name, import_path=import_path, params=params)


def make_promptlearn(
    kind: str = "classifier", name: Optional[str] = None, **params
) -> Optional[ModelSpec]:
    """Return a ModelSpec for the canonical PromptLearn estimators, if installed.
    We rely on the top-level symbols exported by `promptlearn.__init__`:
      - promptlearn.PromptClassifier
      - promptlearn.PromptRegressor
    If the package is not importable, return None gracefully.
    """
    if kind not in {"classifier", "regressor"}:
        raise ValueError("kind must be 'classifier' or 'regressor'")
    try:
        mod = importlib.import_module("promptlearn")
        if kind == "classifier":
            getattr(mod, "PromptClassifier")  # raises AttributeError if missing
            ip = "promptlearn.PromptClassifier"
            default_name = "PromptLearnClf"
        else:
            getattr(mod, "PromptRegressor")
            ip = "promptlearn.PromptRegressor"
            default_name = "PromptLearnReg"
        return ModelSpec(name or default_name, ip, params)
    except Exception:
        return None


def make_promptlearn_variant(
    kind: str, llm_name: str, name: Optional[str] = None, **params
) -> Optional[ModelSpec]:
    """Create a PromptLearn ModelSpec variant pinned to a specific LLM/backend.
    We attach a private param `__pl_model_hint__` that `ModelSpec.build()` uses to
    pass `model=llm_name` to the PromptLearn estimator constructor when possible.
    """
    base = make_promptlearn(kind=kind, name=None, **params)
    if base is None:
        return None
    new_params = dict(base.params)
    new_params["__pl_model_hint__"] = llm_name
    display = name or (
        "PromptLearnClf[" + llm_name + "]"
        if kind == "classifier"
        else "PromptLearnReg[" + llm_name + "]"
    )
    return dataclasses.replace(base, name=display, params=new_params)


# --- Model selection ----------------------------------------------------------


def default_models_for_task_type(task_type: str) -> List[ModelSpec]:
    models: List[ModelSpec] = []
    disable_llm = os.getenv("PLBENCH_DISABLE_LLM", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    llm_env = os.getenv("PLBENCH_LLM_MODELS", "").strip()
    llm_variants = (
        [s.strip() for s in llm_env.split(",") if s.strip()]
        if llm_env
        else ([] if disable_llm else ["gpt-4o", "gpt-5"])
    )

    def _try_add(name: str, import_path: str, **params):
        try:
            importlib.import_module(import_path.rsplit(".", 1)[0])
        except Exception:
            return
        models.append(make_sklearn(name, import_path, **params))

    if task_type == "classification":
        _try_add("LogReg", "sklearn.linear_model.LogisticRegression", max_iter=1000)
        _try_add("DT", "sklearn.tree.DecisionTreeClassifier", random_state=0)
        _try_add(
            "RF",
            "sklearn.ensemble.RandomForestClassifier",
            n_estimators=300,
            random_state=0,
        )
        _try_add(
            "SVM", "sklearn.svm.SVC", kernel="rbf", probability=True, random_state=0
        )
        _try_add(
            "XGB",
            "xgboost.XGBClassifier",
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
        )
        if not disable_llm:
            for llm in llm_variants:
                pl = make_promptlearn_variant(
                    "classifier", llm, name=f"PromptLearnClf[{llm}]"
                )
                if pl:
                    models.append(pl)
    elif task_type == "regression":
        _try_add("LinReg", "sklearn.linear_model.LinearRegression")
        _try_add("DTReg", "sklearn.tree.DecisionTreeRegressor", random_state=0)
        _try_add(
            "RFReg",
            "sklearn.ensemble.RandomForestRegressor",
            n_estimators=300,
            random_state=0,
        )
        _try_add("SVR", "sklearn.svm.SVR", kernel="rbf")
        _try_add(
            "XGBReg",
            "xgboost.XGBRegressor",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
        )
        if not disable_llm:
            for llm in llm_variants:
                pl = make_promptlearn_variant(
                    "regressor", llm, name=f"PromptLearnReg[{llm}]"
                )
                if pl:
                    models.append(pl)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
    return models


# --- Utilities ----------------------------------------------------------------


# Local dtype helpers to avoid Pylance/stubs churn across pandas versions
def _is_categorical_dtype(dtype) -> bool:
    f = getattr(ptypes, "is_categorical_dtype", None)
    if f is not None:
        try:
            return bool(f(dtype))
        except Exception:
            pass
    try:
        from pandas import CategoricalDtype

        return isinstance(dtype, CategoricalDtype) or str(dtype) == "category"
    except Exception:
        return str(dtype) == "category"


def _is_string_dtype(dtype) -> bool:
    f = getattr(ptypes, "is_string_dtype", None)
    if f is not None:
        try:
            return bool(f(dtype))
        except Exception:
            pass
    kind = getattr(dtype, "kind", "")
    return kind in ("O", "U", "S") or str(dtype).startswith("string")


def _is_object_dtype(dtype) -> bool:
    f = getattr(ptypes, "is_object_dtype", None)
    if f is not None:
        try:
            return bool(f(dtype))
        except Exception:
            pass
    return str(dtype) == "object" or getattr(dtype, "kind", "") == "O"


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_row_id(df: pd.DataFrame) -> np.ndarray:
    """Return a NumPy int64 array of row ids (never a pandas ExtensionArray)."""
    if "row_id" in df.columns:
        return df["row_id"].to_numpy(dtype=np.int64, copy=False)
    # Index → ndarray
    return df.index.to_numpy(dtype=np.int64, copy=False)


def _split_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    assert target in df.columns, f"Target column '{target}' not in DataFrame"
    y = df[target]
    drop = [target] + (["row_id"] if "row_id" in df.columns else [])
    return df.drop(columns=drop), y


def _maybe_wrap_preprocessor(est: Any, spec: ModelSpec, X: pd.DataFrame):
    ip = spec.import_path or ""
    is_numeric_lib = ip.startswith("sklearn.") or ip.startswith("xgboost.")
    if not is_numeric_lib:
        return est
    cat = [
        c
        for c in X.columns
        if _is_categorical_dtype(X[c].dtype)
        or _is_string_dtype(X[c].dtype)
        or _is_object_dtype(X[c].dtype)
    ]
    num = [c for c in X.columns if c not in cat]
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline

    needs_scale = (
        any(tok in ip.lower() for tok in [".svm.", "svc", "svr"]) and len(num) > 0
    )
    num_tr = StandardScaler(with_mean=False) if needs_scale else "passthrough"
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    pre = ColumnTransformer(
        [
            ("cat", ohe, cat) if cat else ("cat_drop", "drop", []),
            ("num", num_tr, num) if num else ("num_drop", "drop", []),
        ]
    )
    return Pipeline([("pre", pre), ("est", est)])


def _is_sklearn_model(spec: ModelSpec) -> bool:
    return isinstance(spec.import_path, str) and spec.import_path.startswith("sklearn.")


def _model_task_category(spec: ModelSpec) -> str:
    try:
        if _is_sklearn_model(spec) and spec.import_path:
            from sklearn.base import ClassifierMixin, RegressorMixin

            m, c = spec.import_path.rsplit(".", 1)
            Cls = getattr(importlib.import_module(m), c)
            is_c = issubclass(Cls, ClassifierMixin)
            is_r = issubclass(Cls, RegressorMixin)
            return (
                "both"
                if (is_c and is_r)
                else (
                    "classification" if is_c else ("regression" if is_r else "unknown")
                )
            )
    except Exception:
        pass
    s = (spec.name + " " + (spec.import_path or "")).lower()
    if any(t in s for t in ["clf", "classifier", "logreg", "logistic"]):
        return "classification"
    if any(t in s for t in ["reg", "regressor", "linreg", "ridge", "lasso"]):
        return "regression"
    return "unknown"


def _is_model_compatible_with_task(spec: ModelSpec, task_type: str) -> bool:
    cat = _model_task_category(spec)
    return True if cat in {"both", "unknown"} else (cat == task_type)


# --- Winner marking -----------------------------------------------------------


def _winner_mask(df: pd.DataFrame, primary_metric: str, higher_is_better: bool):
    if df is None or df.empty:
        return None
    if getattr(df.columns, "nlevels", 1) >= 2 and (
        df.columns.names and "metric" in df.columns.names
    ):
        try:
            vals = df.xs(primary_metric, axis=1, level="metric")
        except KeyError:
            return None
    else:
        vals = df
    try:
        vals_np = vals.to_numpy(dtype=float, copy=False)
    except Exception:
        vals_np = vals.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    best_arr = (
        np.nanmax(vals_np, axis=1) if higher_is_better else np.nanmin(vals_np, axis=1)
    )
    best = pd.Series(best_arr, index=vals.index)
    m = vals.eq(best, axis="index")
    if isinstance(m, pd.Series):
        m = pd.DataFrame({vals.columns[0]: m}, index=vals.index)
    return m


def mark_winners_for_display(
    df: pd.DataFrame,
    primary_metric: str,
    higher_is_better: bool,
    marker: str = ">> ",
    decimals: int = 4,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    mask = _winner_mask(df, primary_metric, higher_is_better)
    if mask is None:
        return df
    out = df.copy()
    if getattr(df.columns, "nlevels", 1) >= 2 and (
        df.columns.names and "metric" in df.columns.names
    ):
        mlev = df.columns.names.index("metric")
        mdl = (
            df.columns.names.index("model")
            if "model" in (df.columns.names or [])
            else 0
        )
        tcols = [c for c in df.columns if c[mlev] == primary_metric]
        for col in tcols:
            mdl_name = col[mdl]
            winners = mask[mdl_name].to_numpy()
            vals_col = out[col].values

            def _fmt(v):
                if isinstance(v, (int, float, np.floating)) and pd.notna(v):
                    return f"{float(v):.{decimals}f}"
                return v

            out[col] = [
                (marker + _fmt(v)) if (bool(w) and pd.notna(v)) else _fmt(v)
                for v, w in zip(vals_col, winners)
            ]
    else:
        try:
            vals_np = out.to_numpy(dtype=float, copy=False)
        except Exception:
            vals_np = out.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        best = pd.Series(
            np.nanmax(vals_np, 1) if higher_is_better else np.nanmin(vals_np, 1),
            index=out.index,
        )
        for c in out.columns:
            winners = out[c].eq(best).values
            vals_col = out[c].values

            def _fmt(v):
                if isinstance(v, (int, float, np.floating)) and pd.notna(v):
                    return f"{float(v):.{decimals}f}"
                return v

            out[c] = [
                (marker + _fmt(v)) if (bool(w) and pd.notna(v)) else _fmt(v)
                for v, w in zip(vals_col, winners)
            ]
    return out


# --- Core runner --------------------------------------------------------------
@overload
def run_benchmark(
    task_dirs: Iterable[Union[str, Path]],
    models: Optional[Iterable[ModelSpec]] = ...,
    out_dir: Union[str, Path] = ...,
    seed: int = ...,
    resume: bool = ...,
    chunk_size: int = ...,
    *,
    return_kind: Literal["split"],
) -> Tuple[pd.DataFrame, pd.DataFrame]: ...
@overload
def run_benchmark(
    task_dirs: Iterable[Union[str, Path]],
    models: Optional[Iterable[ModelSpec]] = ...,
    out_dir: Union[str, Path] = ...,
    seed: int = ...,
    resume: bool = ...,
    chunk_size: int = ...,
    *,
    return_kind: Literal["wide"],
) -> pd.DataFrame: ...
@overload
def run_benchmark(
    task_dirs: Iterable[Union[str, Path]],
    models: Optional[Iterable[ModelSpec]] = ...,
    out_dir: Union[str, Path] = ...,
    seed: int = ...,
    resume: bool = ...,
    chunk_size: int = ...,
    *,
    return_kind: Literal["all"],
) -> Dict[str, pd.DataFrame]: ...


def run_benchmark(
    task_dirs: Iterable[Union[str, Path]],
    models: Optional[Iterable[ModelSpec]] = None,
    out_dir: Union[str, Path] = "runs",
    seed: int = 42,
    resume: bool = True,
    chunk_size: int = 50_000,
    *,
    return_kind: str = "wide",
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], Dict[str, pd.DataFrame]]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results_long: List[Dict[str, Any]] = []
    for task_path in task_dirs:
        task = TaskSpec.from_dir(task_path)
        thash = task.fingerprint()
        logger.info(
            f"Task: {task.task_id} [{thash}] :: {task.task_type} → target='{task.target}'"
        )
        train_df = pd.read_csv(task.train_csv)
        test_df = pd.read_csv(task.test_csv)
        X_train, y_train = _split_X_y(train_df, task.target)
        X_test, y_test = _split_X_y(test_df, task.target)
        test_ids = _ensure_row_id(test_df)
        model_list = (
            list(models)
            if models is not None
            else default_models_for_task_type(task.task_type)
        )
        for m in model_list:
            if not _is_model_compatible_with_task(m, task.task_type):
                logger.info(
                    f"[SKIP] {m.name} on {task.task_id}: incompatible with task_type='{task.task_type}'"
                )
                continue
            # Empty train split
            if len(X_train) == 0:
                ip = m.import_path or ""
                needs_fit = ip.startswith("sklearn.") or ip.startswith("xgboost.")
                if needs_fit:
                    run_dir = out_root / task.task_id / m.name
                    (run_dir / "fit").mkdir(parents=True, exist_ok=True)
                    (run_dir / "predict").mkdir(parents=True, exist_ok=True)
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
                        open(run_dir / "metrics.json", "w"),
                        indent=2,
                    )
                    logger.info(
                        f"[SKIP] {m.name} on {task.task_id}: empty train split — estimator requires fit data"
                    )
                    continue
                logger.info(
                    f"[FIT] {m.name} on {task.task_id}: empty-train; proceeding with schema-only fit"
                )
            run_dir = out_root / task.task_id / m.name
            fit_dir = run_dir / "fit"
            pred_dir = run_dir / "predict"
            metric_path = run_dir / "metrics.json"
            for d in (fit_dir, pred_dir):
                d.mkdir(parents=True, exist_ok=True)
            fit_hash_path = fit_dir / "fit_hash.json"
            model_pkl = fit_dir / "model.pkl"
            fit_hash_obj = {
                "task_hash": thash,
                "model": {
                    "name": m.name,
                    "import_path": m.import_path,
                    "params": m.params,
                },
                "seed": seed,
            }
            fit_hash = hashlib.sha1(
                json.dumps(fit_hash_obj, sort_keys=True).encode()
            ).hexdigest()
            need_fit = True
            if resume and model_pkl.exists() and fit_hash_path.exists():
                try:
                    need_fit = json.load(open(fit_hash_path))["fit_hash"] != fit_hash
                except Exception:
                    need_fit = True
            fit_s = 0.0
            try:
                if need_fit:
                    logger.info(f"[FIT] {m.name} on {task.task_id}…")
                    np.random.seed(seed)
                    est = m.build()
                    if task.task_type == "regression" and not ptypes.is_numeric_dtype(
                        y_train
                    ):
                        y_train = pd.to_numeric(y_train, errors="raise")
                    est = _maybe_wrap_preprocessor(est, m, X_train)
                    if hasattr(est, "set_params"):
                        try:
                            est.set_params(random_state=seed)
                        except Exception:
                            pass
                    t0 = time.time()
                    est.fit(X_train, y_train)
                    fit_s = time.time() - t0
                    joblib.dump(est, model_pkl)
                    json.dump(
                        {"fit_hash": fit_hash, "fit_seconds": fit_s, "at": _now_iso()},
                        open(fit_hash_path, "w"),
                    )
                    logger.info(f"[FIT] done in {fit_s:.2f}s; cached → {model_pkl}")
                else:
                    est = joblib.load(model_pkl)
                    logger.info(f"[FIT] reused cache → {model_pkl}")
            except Exception as e:
                metric_keys = (
                    ["accuracy"] if task.task_type == "classification" else ["rmse"]
                ) + (
                    ["macro_f1"]
                    if task.task_type == "classification" and "macro_f1" in task.metrics
                    else []
                )
                metrics_nan = {k: float("nan") for k in metric_keys}
                json.dump(
                    {
                        "task_id": task.task_id,
                        "model": m.name,
                        "fit_seconds": fit_s,
                        "predict_seconds": 0.0,
                        "n_test": int(len(X_test)),
                        "metrics": metrics_nan,
                        "error": {"fit_error": repr(e)},
                        "completed_at": _now_iso(),
                    },
                    open(metric_path, "w"),
                    indent=2,
                )
                for k in metric_keys:
                    results_long.append(
                        {
                            "task_id": task.task_id,
                            "model": m.name,
                            "metric": k,
                            "value": np.nan,
                        }
                    )
                logger.exception(f"[FIT][ERROR] {m.name} on {task.task_id}: {e}")
                continue
            est_meta = (
                {}
                if _is_sklearn_model(m)
                else {
                    "estimator_class": m.import_path,
                    "llm_model": getattr(est, "model", None),
                    "verbose": getattr(est, "verbose", None),
                    "max_train_rows": getattr(est, "max_train_rows", None),
                }
            )
            preds_csv = pred_dir / "predictions.csv"
            prog = pred_dir / "progress.json"
            if not resume:
                try:
                    if preds_csv.exists():
                        preds_csv.unlink()
                    if prog.exists():
                        prog.unlink()
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
            all_ids = test_ids
            remaining_idx = (
                np.where(~np.isin(all_ids, np.fromiter(done_ids, dtype=np.int64)))[0]
                if len(done_ids) > 0
                else np.arange(len(all_ids))
            )
            n_total = len(all_ids)
            n_remaining = len(remaining_idx)
            logger.info(
                f"[PREDICT] {m.name} on {task.task_id}: {n_remaining}/{n_total} rows remaining"
            )
            t0 = time.time()
            if n_total == 0:
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
                    open(prog, "w"),
                )
                json.dump(
                    {
                        "task_id": task.task_id,
                        "model": m.name,
                        "fit_seconds": json.load(open(fit_hash_path)).get(
                            "fit_seconds"
                        ),
                        "predict_seconds": 0.0,
                        "n_test": 0,
                        "metrics": {},
                        "skipped_reason": "empty_test_split",
                        "completed_at": _now_iso(),
                        "estimator": est_meta,
                    },
                    open(metric_path, "w"),
                    indent=2,
                )
                logger.info(
                    f"[SKIP] {m.name} on {task.task_id}: empty test split — no metrics computed"
                )
                continue
            try:
                t0 = time.time()
                for s in range(0, len(remaining_idx), chunk_size):
                    e = min(s + chunk_size, len(remaining_idx))
                    sel = remaining_idx[s:e]
                    Xc = X_test.iloc[sel]
                    rows = all_ids[sel]
                    yhat = est.predict(Xc)
                    pd.DataFrame({"row_id": rows, "y_pred": yhat}).to_csv(
                        preds_csv,
                        mode=("a" if preds_csv.exists() else "w"),
                        header=not preds_csv.exists(),
                        index=False,
                    )
                    json.dump(
                        {
                            "last_chunk_rows": int(len(rows)),
                            "done_rows": int(e),
                            "total_rows": int(len(remaining_idx)),
                            "at": _now_iso(),
                        },
                        open(prog, "w"),
                    )
                    logger.info(
                        f"[PREDICT] rows {s}–{e} / {len(remaining_idx)} appended"
                    )
                t_pred = time.time() - t0
                pred_df = (
                    pd.read_csv(preds_csv)
                    .drop_duplicates(subset=["row_id"], keep="last")
                    .sort_values("row_id")
                )
                order = np.argsort(all_ids)
                y_true = y_test.to_numpy()[order]
                id2p = dict(zip(pred_df["row_id"].astype(np.int64), pred_df["y_pred"]))
                try:
                    y_pred = np.array([id2p[int(r)] for r in all_ids[order]])
                except KeyError as e:
                    raise RuntimeError(
                        "row_id mismatch between predictions and ground truth"
                    ) from e
                metrics: Dict[str, float] = {}
                if task.task_type == "classification":
                    from sklearn.metrics import accuracy_score, f1_score

                    y_true_np = np.asarray(y_true)
                    y_pred_np = np.asarray(y_pred)
                    metrics["accuracy"] = float(accuracy_score(y_true_np, y_pred_np))
                    if "macro_f1" in task.metrics:
                        metrics["macro_f1"] = float(
                            f1_score(y_true_np, y_pred_np, average="macro")
                        )
                else:
                    y_true = y_true.astype(float)
                    y_pred = y_pred.astype(float)
                    metrics["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                json.dump(
                    {
                        "task_id": task.task_id,
                        "model": m.name,
                        "fit_seconds": json.load(open(fit_hash_path)).get(
                            "fit_seconds"
                        ),
                        "predict_seconds": t_pred,
                        "n_test": int(len(y_true)),
                        "metrics": metrics,
                        "completed_at": _now_iso(),
                        "estimator": est_meta,
                    },
                    open(metric_path, "w"),
                    indent=2,
                )
                for k, v in metrics.items():
                    results_long.append(
                        {
                            "task_id": task.task_id,
                            "model": m.name,
                            "metric": k,
                            "value": v,
                        }
                    )
                logger.info(f"[SCORE] {m.name} on {task.task_id}: {metrics}")
            except Exception as e:
                t_pred = time.time() - t0
                metric_keys = (
                    ["accuracy"] if task.task_type == "classification" else ["rmse"]
                ) + (
                    ["macro_f1"]
                    if task.task_type == "classification" and "macro_f1" in task.metrics
                    else []
                )
                metrics_nan = {k: float("nan") for k in metric_keys}
                json.dump(
                    {
                        "task_id": task.task_id,
                        "model": m.name,
                        "fit_seconds": json.load(open(fit_hash_path)).get(
                            "fit_seconds"
                        ),
                        "predict_seconds": t_pred,
                        "n_test": int(len(X_test)),
                        "metrics": metrics_nan,
                        "error": {"predict_or_score_error": repr(e)},
                        "completed_at": _now_iso(),
                        "estimator": est_meta,
                    },
                    open(metric_path, "w"),
                    indent=2,
                )
                for k in metric_keys:
                    results_long.append(
                        {
                            "task_id": task.task_id,
                            "model": m.name,
                            "metric": k,
                            "value": np.nan,
                        }
                    )
                logger.exception(
                    f"[PREDICT/SCORE][ERROR] {m.name} on {task.task_id}: {e}"
                )
                continue
    df_long = pd.DataFrame(results_long)
    if df_long.empty:
        logger.warning("No results produced. Check inputs.")
        return pd.DataFrame()
    df_wide = df_long.pivot_table(
        index="task_id", columns=["model", "metric"], values="value"
    ).sort_index(axis=1, level=0)
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
    if return_kind == "split":
        return df_cls, df_reg
    if return_kind == "all":
        return {"wide": df_wide, "classification": df_cls, "regression": df_reg}
    return df_wide
