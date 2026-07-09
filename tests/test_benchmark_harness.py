"""Regression tests for benchmarks/run_openml_fit.py's cache-writing behavior.

benchmarks/ isn't part of the skribe package (no __init__.py, standalone
scripts invoked by run_all_models.sh), so these tests reach into it via
sys.path the same way the scripts reach into benchmark_utils.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import run_openml_fit  # noqa: E402

from skribe.classifier import SkribeClassifier


@pytest.fixture
def tiny_csv_spec(tmp_path):
    """A (openml_name, version, csv_path, target_col, description) spec pointing
    at a small local CSV, so run_one_skribe never touches the network."""
    df = pd.DataFrame(
        {
            "a": list(range(20)),
            "target": ([0] * 10 + [1] * 10),
        }
    )
    csv_path = tmp_path / "tiny.csv"
    df.to_csv(csv_path, index=False)
    return ("unused", 1, str(csv_path), "target", "toy dataset for tests")


def test_cached_generated_code_matches_scored_code(monkeypatch, tiny_csv_spec):
    """result['skribe']['generated_code'] must be the code predict_fn was actually
    built from (clf.python_code_, post-extend-pass) -- not the pre-extension draft
    (clf.raw_python_code_) -- since that's what produced the reported accuracy.

    Regression test for the pass-9 cache-audit finding: the extend pass can
    change behavior (it's supposed to only expand categorical mappings, but
    empirically doesn't always honor that), so caching the wrong code silently
    decouples the stored source from the score it produced.
    """
    raw_code = "def predict(**features): return 0"
    extended_code = "def predict(**features): return 1"  # deliberately different behavior

    monkeypatch.setattr(
        SkribeClassifier, "_call_llm", lambda self, prompt, web_search=False: raw_code
    )
    monkeypatch.setattr(
        SkribeClassifier, "_extend_code", lambda self, code, web_search=False: extended_code
    )

    result = run_openml_fit.run_one_skribe(
        dataset="tiny",
        spec=tiny_csv_spec,
        model_id="gpt-5.4-mini",
        max_rows=None,
        cache_dir=None,
        skip_context=True,
    )

    assert "error" not in result["skribe"], result["skribe"].get("error")
    assert result["skribe"]["generated_code"] == extended_code
    assert result["skribe"]["generated_code"] != raw_code


def test_cached_result_has_explicit_status_on_success(monkeypatch, tiny_csv_spec):
    """result['skribe']['status'] must be an explicit 'ok'/'error' marker, not
    something callers have to derive by checking for the absence of an
    'error' key or an accuracy of 0 -- those checks are already duplicated
    (and subtly inconsistent) across benchmark_utils.build_summary_df,
    run_openml_fit's own cache-read retry check, and
    build_skribe_inspector.py, each re-deriving the same state slightly
    differently."""
    monkeypatch.setattr(
        SkribeClassifier,
        "_call_llm",
        lambda self, prompt, web_search=False: "def predict(**features): return 0",
    )
    monkeypatch.setattr(SkribeClassifier, "_extend_code", lambda self, code, web_search=False: code)

    result = run_openml_fit.run_one_skribe(
        dataset="tiny",
        spec=tiny_csv_spec,
        model_id="gpt-5.4-mini",
        max_rows=None,
        cache_dir=None,
        skip_context=True,
    )

    assert result["skribe"]["status"] == "ok"


def test_cached_result_has_explicit_status_on_failure(monkeypatch, tiny_csv_spec):
    """Same explicit marker on the failure path -- status must be 'error'
    whenever result['skribe']['error'] is set, derived from the same place
    that sets the error, not left for each downstream reader to infer."""
    monkeypatch.setattr(
        SkribeClassifier,
        "_call_llm",
        lambda self, prompt, web_search=False: "def predict(**features): raise ValueError('nope')",
    )
    monkeypatch.setattr(SkribeClassifier, "_extend_code", lambda self, code, web_search=False: code)

    result = run_openml_fit.run_one_skribe(
        dataset="tiny",
        spec=tiny_csv_spec,
        model_id="gpt-5.4-mini",
        max_rows=None,
        cache_dir=None,
        skip_context=True,
    )

    assert result["skribe"]["status"] == "error"


def test_cached_baseline_result_has_explicit_status(tiny_csv_spec):
    """Baselines (logreg/xgboost/tabpfn) go through a separate code path
    (run_one_baseline) than skribe results -- it must carry the same
    explicit status marker on success."""
    result = run_openml_fit.run_one_baseline(
        dataset="tiny",
        spec=tiny_csv_spec,
        model="logreg",
        max_rows=None,
        cache_dir=None,
    )
    assert result["logreg"]["status"] == "ok"


def test_cached_result_includes_fit_log(monkeypatch, tiny_csv_spec):
    """result['skribe']['fit_log'] must capture every validation failure/retry
    that happened during fit(), not just the final accuracy -- so a cache
    file is enough to see what happened during a run (rate limits, retries,
    validation errors) without needing the ephemeral run log.

    Regression for the request that followed the cache audit: the only
    signal in a cached result used to be a single terminal accuracy or error
    string, discarding everything that happened along the way (e.g. a
    validation error that was fixed on retry #2)."""
    bad_code = "def predict(**features): raise ValueError('boom')"
    good_code = "def predict(**features): return 0"
    outputs = iter([bad_code, good_code])

    monkeypatch.setattr(
        SkribeClassifier, "_call_llm", lambda self, prompt, web_search=False: next(outputs)
    )
    monkeypatch.setattr(SkribeClassifier, "_extend_code", lambda self, code, web_search=False: code)

    result = run_openml_fit.run_one_skribe(
        dataset="tiny",
        spec=tiny_csv_spec,
        model_id="gpt-5.4-mini",
        max_rows=None,
        cache_dir=None,
        skip_context=True,
    )

    assert "error" not in result["skribe"], result["skribe"].get("error")
    fit_log = result["skribe"]["fit_log"]
    assert len(fit_log) == 1
    assert fit_log[0]["stage"] == "validation"
    assert "boom" in fit_log[0]["error"]


def test_cached_failure_includes_fit_log(monkeypatch, tiny_csv_spec):
    """Even when every retry attempt fails and run_one_skribe reports a
    top-level error, the retry history leading up to that failure must
    still be attached -- otherwise a cached failure only ever shows the
    final error, hiding whether earlier attempts hit a different problem."""
    monkeypatch.setattr(
        SkribeClassifier,
        "_call_llm",
        lambda self, prompt, web_search=False: "def predict(**features): raise ValueError('nope')",
    )
    monkeypatch.setattr(SkribeClassifier, "_extend_code", lambda self, code, web_search=False: code)

    result = run_openml_fit.run_one_skribe(
        dataset="tiny",
        spec=tiny_csv_spec,
        model_id="gpt-5.4-mini",
        max_rows=None,
        cache_dir=None,
        skip_context=True,
    )

    assert "error" in result["skribe"]
    fit_log = result["skribe"]["fit_log"]
    # default max_retries=2 -> 3 total attempts, all failing.
    assert len(fit_log) == 3
    assert all("nope" in entry["error"] for entry in fit_log)
