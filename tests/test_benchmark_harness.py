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

from skribe.classifier import SkribeClassifier

import run_openml_fit  # noqa: E402


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
    monkeypatch.setattr(SkribeClassifier, "_extend_code", lambda self, code: extended_code)

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
