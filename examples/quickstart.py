#!/usr/bin/env python
"""A guided tour of promptlearn, as a menu of self-contained demos.

Every demo makes live LLM calls, so they cost real API usage. Running this
script with no arguments just prints this help and the demo list — it does not
run anything. Pick how to run:

    python quickstart.py                  # show this help + the demo list (runs nothing)
    python quickstart.py --list           # print just the demo names + summaries
    python quickstart.py --demo zero_row  # run ONE demo (cheap, recommended)
    python quickstart.py --all            # run EVERY demo (slow; many live LLM calls)

More examples:
    python quickstart.py --demo titanic --dump artifacts/   # save code + explanation + model
    python quickstart.py --all --dump artifacts/             # run all, persist everything
    python quickstart.py --demo compare --dataset mammal --rows 8
    python quickstart.py --demo world_knowledge --model claude-sonnet-4-6

--dump DIR persists, per demo, into DIR/<demo>/: the console output
(output.txt), each fitted model's generated code (*.raw.py, *.extended.py),
its plain-English explanation (*.explanation.txt), and the joblib model.

Note: each fit makes two LLM calls. The default gpt-5.5 gives the best
heuristics but is slow/expensive; for a fast, cheap tour use --model gpt-5.4-mini.
"""

import argparse
import contextlib
import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from promptlearn import (
    PromptClassifier,
    PromptRegressor,
    PromptFeatureEngineer,
    compare_models,
)


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --------------------------------------------------------------------------- #
# Artifact persistence (--dump): save each demo's output, generated code,
# explanation, and model. _DUMP_DIR is set by the runner while a demo runs.
# --------------------------------------------------------------------------- #
_DUMP_DIR = None


class _Tee:
    """Duplicate writes to several streams (console + a file) at once."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


@contextlib.contextmanager
def _tee_stdout(fileobj):
    original = sys.stdout
    sys.stdout = _Tee(original, fileobj)
    try:
        yield
    finally:
        sys.stdout = original


def dump_model(est, label="model"):
    """When --dump is active, persist a fitted promptlearn estimator into the
    current demo's artifact dir: its raw and extended generated code, a
    plain-English explanation, and the joblib model. A no-op otherwise."""
    if not _DUMP_DIR:
        return est
    import joblib

    base = os.path.join(_DUMP_DIR, label)
    if getattr(est, "raw_python_code_", None):
        Path(f"{base}.raw.py").write_text(est.raw_python_code_, encoding="utf-8")
    if getattr(est, "python_code_", None):
        Path(f"{base}.extended.py").write_text(est.python_code_, encoding="utf-8")
    try:
        Path(f"{base}.explanation.txt").write_text(
            est.explain().summary, encoding="utf-8"
        )
    except Exception as e:  # explanation is best-effort; never fail the dump
        Path(f"{base}.explanation.txt").write_text(
            f"(explain failed: {e})", encoding="utf-8"
        )
    joblib.dump(est, f"{base}.joblib")
    return est


# --------------------------------------------------------------------------- #
# Bite-size feature demos
# --------------------------------------------------------------------------- #
def demo_zero_row(args):
    """Fit with column names only — no rows. The LLM infers the rule from the
    schema and its world knowledge."""
    X = pd.DataFrame(columns=["country_name"])
    y = pd.Series(name="has_blue_in_flag", dtype=int)

    clf = PromptClassifier(model=args.model, verbose=False)
    clf.fit(X, y)  # only headers — nothing to learn from but the names
    dump_model(clf)

    for country, expected in [("Japan", "no"), ("France", "yes")]:
        pred = int(clf.predict(pd.DataFrame([{"country_name": country}]))[0])
        print(f"  {country}: has_blue_in_flag={pred}  (expected ~{expected})")


def demo_sample(args):
    """Generate synthetic rows from a fitted model with .sample(n)."""
    X = np.array([[-1], [0], [1], [2], [3]])
    y = np.array([1, 3, 5, 7, 9])  # y = 2x + 3

    reg = PromptRegressor(model=args.model, verbose=False)
    reg.fit(X, y)
    dump_model(reg)
    print(reg.sample(10).to_string(index=False))


def demo_joblib(args):
    """The fitted model is just code, so it serializes tiny and reloads without
    an LLM client."""
    import joblib

    X = np.array([[-1], [0], [1], [2], [3]])
    y = np.array([1, 3, 5, 7, 9])  # y = 2x + 3

    reg = PromptRegressor(model=args.model, verbose=False)
    reg.fit(X, y)
    dump_model(reg)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        joblib.dump(reg, path)
        size = path.stat().st_size
        reloaded = joblib.load(path)
        preds = reloaded.predict(np.array([[4], [5]]))
    print(f"  serialized to {size} bytes, reloaded and predicted: {np.round(preds, 2)}")
    print("  (no LLM client is stored; the heuristic is recompiled on load)")


def demo_linear(args):
    """Recover the simplest relationship, y = 2x + 3, from a handful of points."""
    X = np.linspace(-1, 3, 5).reshape(-1, 1)
    y = 2 * X.flatten() + 3

    reg = PromptRegressor(model=args.model, verbose=False)
    reg.fit(X, y)
    dump_model(reg)

    X_test = np.array([[4], [5], [6]])
    preds = reg.predict(X_test)
    for x_val, pred in zip(X_test.flatten(), preds):
        print(f"  x={x_val} → y={pred:.2f}  (expected {2 * x_val + 3})")


def demo_nonlinear(args):
    """Recover a nonlinear, multi-variable relationship:
    y = 3 * length^2 + 2 * volume + 5."""
    rng = np.random.default_rng(42)
    length = rng.uniform(-2, 2, size=40)
    volume = rng.uniform(0, 5, size=40)
    X = pd.DataFrame({"length": length, "volume": volume})
    y = pd.Series(3 * length**2 + 2 * volume + 5, name="output")

    reg = PromptRegressor(model=args.model, verbose=False)
    reg.fit(X, y)
    dump_model(reg)

    tests = pd.DataFrame(
        [
            {"length": 1.0, "volume": 2.0},
            {"length": -1.5, "volume": 3.0},
            {"length": 0.0, "volume": 4.0},
        ]
    )
    preds = reg.predict(tests)
    truth = 3 * tests["length"] ** 2 + 2 * tests["volume"] + 5
    for (_, row), pred, true in zip(tests.iterrows(), preds, truth):
        print(
            f"  length={row['length']:+.1f} volume={row['volume']:.1f}"
            f" → predicted={pred:6.2f}  expected={true:6.2f}"
        )


def demo_xor(args):
    """Learn XOR — the textbook nonlinearly-separable problem."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    clf = PromptClassifier(model=args.model, verbose=False)
    clf.fit(X, y)
    dump_model(clf)
    preds = clf.predict(X)
    for inputs, pred, true in zip(X.tolist(), preds, y):
        mark = "ok" if int(pred) == int(true) else "WRONG"
        print(f"  {inputs} → {int(pred)}  (expected {int(true)}) [{mark}]")


def demo_world_knowledge(args):
    """promptlearn can fold in real-world knowledge that the raw features alone
    don't contain — both for classification and regression."""
    banner("world knowledge — classification (does the flag contain blue?)")
    data = pd.DataFrame(
        {
            "country_name": ["Sweden", "Japan", "Italy", "United States", "Germany"],
            "has_blue_in_flag": [1, 0, 0, 1, 0],
        }
    )
    clf = PromptClassifier(model=args.model, verbose=False)
    clf.fit(data[["country_name"]], data["has_blue_in_flag"])
    dump_model(clf, "flag_classifier")
    for country in ["France", "Brazil", "Spain"]:
        pred = int(clf.predict(pd.DataFrame([{"country_name": country}]))[0])
        print(f"  {country}: {'blue' if pred else 'no blue'}")

    # The classic "how much money was the dog given?" riddle: payout scales with
    # the number of legs, which the model knows per animal.
    banner("world knowledge — regression (money-per-animal riddle)")
    train = pd.DataFrame({"animal": ["chicken", "ant", "spider"], "money": [7, 21, 28]})
    reg = PromptRegressor(model=args.model, verbose=False)
    reg.fit(train[["animal"]], train["money"])
    dump_model(reg, "money_regressor")
    for animal in ["dog", "bee", "crab"]:
        pred = float(reg.predict(pd.DataFrame([{"animal": animal}]))[0])
        print(f"  {animal}: {pred:.1f}")


def demo_feature_engineer(args):
    """PromptFeatureEngineer writes feature code from world knowledge, so a
    plain LogisticRegression can use signal the raw column doesn't expose."""
    from sklearn.compose import ColumnTransformer, make_column_selector as selector
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    # Predict whether a country is tropical — purely from its name.
    df = pd.DataFrame(
        {
            "country": [
                "Brazil",
                "Indonesia",
                "Kenya",
                "Norway",
                "Canada",
                "Russia",
                "Colombia",
                "Sweden",
                "Thailand",
                "Finland",
            ],
            "tropical": [1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
        }
    )
    X, y = df[["country"]], df["tropical"]

    fe = PromptFeatureEngineer(model=args.model, verbose=False)
    encode = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                selector(dtype_exclude="number"),
            )
        ],
        remainder="passthrough",
    )
    pipe = Pipeline(
        [
            ("features", fe),
            ("encode", encode),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(X, y)
    dump_model(fe, "feature_engineer")
    print(f"  engineered features: {fe.new_feature_names_}")
    print(f"  train accuracy: {pipe.score(X, y):.3f}")


def demo_multioutput(args):
    """promptlearn estimators are sklearn-compatible, so meta-estimators like
    MultiOutputRegressor wrap them directly (Linnerud: 3 targets)."""
    from sklearn.datasets import load_linnerud
    from sklearn.multioutput import MultiOutputRegressor

    data = load_linnerud()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=data.target_names)

    reg = MultiOutputRegressor(PromptRegressor(model=args.model, verbose=False))
    reg.fit(X, y)
    for target, inner in zip(y.columns, reg.estimators_):
        dump_model(inner, f"target_{target}")
    preds = pd.DataFrame(reg.predict(X.head()), columns=y.columns)
    print(preds.round(1).to_string(index=False))


def demo_gridsearch(args):
    """Because the estimators follow the sklearn API, GridSearchCV tunes their
    hyper-parameters (here: how many training rows to send the LLM)."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import GridSearchCV

    data = load_iris(as_frame=True)
    X, y = data.data.head(60), data.target.head(60)  # keep it small/cheap

    search = GridSearchCV(
        PromptClassifier(model=args.model, verbose=False),
        param_grid={"max_train_rows": [20, 40]},
        cv=2,
    )
    search.fit(X, y)
    dump_model(search.best_estimator_, "best_estimator")
    print(
        f"  best params: {search.best_params_}  best CV score: {search.best_score_:.3f}"
    )


def demo_large_dataset(args):
    """A real, larger dataset (OpenML 'adult', ~48k rows). max_train_rows caps
    how many rows are sent to the LLM, so cost stays bounded regardless of size."""
    from sklearn.datasets import fetch_openml
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    print("  fetching the 'adult' census dataset from OpenML ...")
    adult = fetch_openml("adult", version=2, as_frame=True)
    X = adult.data.astype(str)  # keep categoricals readable for the LLM
    y = (adult.target == ">50K").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    n_rows = min(args.rows * 5, 80)
    print(
        f"  full train set: {len(X_train)} rows; sending only max_train_rows={n_rows}"
    )

    clf = PromptClassifier(model=args.model, verbose=False, max_train_rows=n_rows)
    clf.fit(X_train, y_train)
    dump_model(clf)

    X_eval, y_eval = X_test.head(40), y_test.head(40)
    acc = accuracy_score(y_eval, clf.predict(X_eval))
    print(f"  accuracy on 40 held-out rows: {acc:.3f}")


# --------------------------------------------------------------------------- #
# compare: benchmark promptlearn vs classical models (the model zoo)
# --------------------------------------------------------------------------- #
def _load_csv_pair(stem, target, task):
    here = os.path.dirname(os.path.abspath(__file__))
    frames = [
        pd.read_csv(os.path.join(here, "data", f"{stem}_{split}.csv"))
        for split in ("train", "val")
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.drop(columns=[target]), df[target], task


def _load_compare_dataset(name):
    """Return (X, y, task) for one of: iris, titanic, diabetes, mammal, fall."""
    if name == "iris":
        from sklearn.datasets import load_iris

        data = load_iris(as_frame=True)
        return data.data, data.target, "classification"
    if name == "diabetes":
        from sklearn.datasets import load_diabetes

        data = load_diabetes(as_frame=True)
        return data.data, data.target, "regression"
    if name == "mammal":  # world-knowledge: only promptlearn can use the name
        return _load_csv_pair("mammal", "is_mammal", "classification")
    if name == "fall":  # physics: promptlearn can recover fall_time = sqrt(2h/g)
        return _load_csv_pair("fall", "fall_time_s", "regression")
    if name == "titanic":
        df, _ = _load_titanic()
        keep = {"pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"}
        df = df.dropna(subset=["survived"])
        X = df[[c for c in df.columns if c in keep]]
        return X, df["survived"].astype(int), "classification"
    raise ValueError(f"unknown dataset {name!r}")


def _build_compare_models(task, model):
    """Assemble {name: estimator} for the task, plus XGBoost if available."""
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression

    if task == "classification":
        models = {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "logreg": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=50, random_state=0),
            f"promptlearn[{model}]": PromptClassifier(model=model, verbose=False),
        }
        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(eval_metric="logloss")
        except ImportError:
            pass
    else:
        models = {
            "dummy": DummyRegressor(),
            "linreg": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=50, random_state=0),
            f"promptlearn[{model}]": PromptRegressor(model=model, verbose=False),
        }
        try:
            from xgboost import XGBRegressor

            models["xgboost"] = XGBRegressor()
        except ImportError:
            pass
    return models


def demo_compare(args):
    """Fit promptlearn alongside sklearn/XGBoost baselines on one dataset and
    print a side-by-side metrics table and a row-by-row predictions table."""
    from sklearn.model_selection import train_test_split

    X, y, task = _load_compare_dataset(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    models = _build_compare_models(task, args.model)
    print(
        f"  dataset: {args.dataset} ({task}, {len(X)} rows, {X.shape[1]} features)\n"
        f"  models:  {', '.join(models)}"
    )

    metrics, predictions = compare_models(
        models, X_train, y_train, X_test, y_test, task
    )
    # The promptlearn estimators are fitted in place; persist their heuristics.
    for model_name, est in models.items():
        if getattr(est, "python_code_", None):
            dump_model(est, model_name.replace("/", "_"))
    primary, ascending = ("rmse", True) if task == "regression" else ("accuracy", False)
    metrics = metrics.sort_values(primary, ascending=ascending)

    banner(f"METRICS (side by side, sorted by {primary})")
    print(metrics.to_string(float_format=lambda v: f"{v:.4f}"))
    banner(f"PREDICTIONS (row by row, first {args.rows} test rows)")
    print(predictions.head(args.rows).to_string(float_format=lambda v: f"{v:.3f}"))


# --------------------------------------------------------------------------- #
# titanic: the deep tour — generated code, explain(), joblib
# --------------------------------------------------------------------------- #
def _load_titanic():
    """Return (dataframe, target_column). Uses seaborn if available, else a CSV.

    The two sources name the target differently ('survived' vs 'Survived'), so
    we normalise to lowercase 'survived'.
    """
    try:
        import seaborn as sns

        return sns.load_dataset("titanic"), "survived"
    except Exception:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        df.columns = [c.lower() for c in df.columns]
        return df, "survived"


def _preprocess_titanic(df, target):
    """Drop leaky/sparse columns, fill gaps, keep categoricals as readable strings."""
    df = df.drop(
        columns=["deck", "embark_town", "alive", "who", "adult_male", "class", "alone"],
        errors="ignore",
    )
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target]).fillna("unknown")
    for col in [c for c in X.columns if X[c].dtype == object]:
        X[col] = X[col].astype(str)
    return X, df[target]


def demo_titanic(args):
    """The deep tour: fit on Titanic, then show the generated predict() function
    (the model *is* code), explain the rule globally and per-row. With --dump it
    also persists the code/explanation/model (like every demo)."""
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    df, target = _load_titanic()
    X, y = _preprocess_titanic(df, target)
    print(f"  loaded {len(X)} rows, {X.shape[1]} features: {list(X.columns)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    clf = PromptClassifier(model=args.model, verbose=args.verbose)
    clf.fit(X_train, y_train)
    dump_model(clf)

    y_pred = clf.predict(X_val)
    banner(f"VALIDATION ACCURACY: {accuracy_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred))

    banner("GENERATED PYTHON HEURISTIC  (clf.python_code_)")
    print(
        "This function *is* the model — predict() runs it directly, with no LLM\n"
        "call per row. The LLM wrote it from the training data, then it was\n"
        "validated to make sure it executes.\n"
    )
    print(clf.python_code_)

    banner("GLOBAL EXPLANATION  (clf.explain())")
    explanation = clf.explain()
    print(explanation.summary)
    print("\nFeatures the rule actually uses:", explanation.features_used)

    n = min(args.rows, len(X_val))
    banner(f"LOCAL EXPLANATIONS  (clf.explain(row)) — first {n} validation rows")
    for i in range(n):
        row = X_val.iloc[[i]]
        pred, actual = int(clf.predict(row)[0]), int(y_val.iloc[i])
        mark = "correct" if pred == actual else "WRONG"
        print(f"\n--- row {i}: predicted={pred}, actual={actual} ({mark}) ---")
        print(clf.explain(row).summary)


DEMOS = {
    "zero_row": demo_zero_row,
    "sample": demo_sample,
    "joblib": demo_joblib,
    "linear": demo_linear,
    "nonlinear": demo_nonlinear,
    "xor": demo_xor,
    "world_knowledge": demo_world_knowledge,
    "feature_engineer": demo_feature_engineer,
    "multioutput": demo_multioutput,
    "gridsearch": demo_gridsearch,
    "large_dataset": demo_large_dataset,
    "compare": demo_compare,
    "titanic": demo_titanic,
}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--demo", choices=list(DEMOS), help="Run a single demo.")
    p.add_argument(
        "--all",
        action="store_true",
        help="Run every demo in one go (slow; many live LLM calls).",
    )
    p.add_argument("--list", action="store_true", help="Print the demo list and exit.")
    p.add_argument(
        "--model",
        default="gpt-5.5",
        help="LLM model string (e.g. gpt-5.5, gpt-5.4-mini, claude-sonnet-4-6, ollama:llama3.1).",
    )
    p.add_argument(
        "--dataset",
        default="mammal",
        choices=["iris", "titanic", "diabetes", "mammal", "fall"],
        help="Dataset for the 'compare' demo.",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Rows of per-instance output to show (compare/titanic/large_dataset).",
    )
    p.add_argument(
        "--dump",
        nargs="?",
        const="artifacts",
        default=None,
        metavar="DIR",
        help="Persist each demo's output, generated code, explanation, and model "
        "into DIR/<demo>/ (default DIR: artifacts).",
    )
    p.add_argument(
        "--verbose", action="store_true", help="Show LLM prompts during fit."
    )
    return p


def _print_demo_list():
    print("\nDemos (run one with --demo NAME, or all with --all):\n")
    for name, fn in DEMOS.items():
        lines = (fn.__doc__ or "").strip().splitlines()
        summary = lines[0] if lines else ""
        print(f"  {name:16} {summary}")


def _run_demo(name, args):
    """Run one demo and return (status, seconds). Never raises, so running the
    whole suite isn't aborted by a single failing demo (e.g. a flaky LLM call)."""
    banner(f"SCENARIO: {name}  (model={args.model})")
    print("  running… (live LLM calls; a reasoning model can take a while per fit)\n")

    global _DUMP_DIR
    out_dir = None
    if args.dump is not None:
        out_dir = os.path.join(args.dump, name)
        os.makedirs(out_dir, exist_ok=True)
    _DUMP_DIR = out_dir

    start = time.time()
    try:
        if out_dir:
            # Tee the demo's console output to a file alongside its artifacts.
            with open(os.path.join(out_dir, "output.txt"), "w", encoding="utf-8") as f:
                with _tee_stdout(f):
                    DEMOS[name](args)
            print(f"  ↳ artifacts saved to {os.path.abspath(out_dir)}")
        else:
            DEMOS[name](args)
        return "ok", time.time() - start
    except Exception:
        traceback.print_exc()
        return "FAILED", time.time() - start
    finally:
        _DUMP_DIR = None


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        _print_demo_list()
        return

    if args.demo:
        selected = [args.demo]
    elif args.all:
        selected = list(DEMOS)
    else:
        # No demo requested: show how to run things and execute nothing.
        parser.print_help()
        _print_demo_list()
        return

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    results = [(name, *_run_demo(name, args)) for name in selected]

    # Summarise whenever more than one demo ran (i.e. the --all run).
    if len(results) > 1:
        banner("SUMMARY")
        for name, status, secs in results:
            print(f"  {name:16} {status:7} {secs:6.1f}s")
        failed = [name for name, status, _ in results if status != "ok"]
        if failed:
            print(
                f"\n{len(failed)} of {len(results)} demos failed: {', '.join(failed)}"
            )
            sys.exit(1)
        print(f"\nAll {len(results)} demos passed.")


if __name__ == "__main__":
    main()
