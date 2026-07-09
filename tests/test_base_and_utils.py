import warnings

import pandas as pd
import pytest
from skribe.base import BaseSkribeEstimator, _CONTEXT_HEADROOM
from skribe.utils import sanitize_dataset_description


def test_get_set_params():
    est = BaseSkribeEstimator(model="gpt-4", verbose=True, max_train_rows=10)
    params = est.get_params()
    assert params["model"] == "gpt-4"
    est.set_params(model="gpt-3.5-turbo")
    assert est.model == "gpt-3.5-turbo"


def test_call_llm_raises(monkeypatch):
    import litellm

    est = BaseSkribeEstimator(model="gpt-4", verbose=False, max_train_rows=1)

    def boom(*args, **kwargs):
        raise RuntimeError("provider error")

    monkeypatch.setattr(litellm, "completion", boom)
    with pytest.raises(RuntimeError):
        est._call_llm("this should fail")


def test_call_llm_normalizes_ollama_model(monkeypatch):
    """The documented ``ollama:model`` syntax is mapped to litellm's ``ollama/model``."""
    import litellm

    captured = {}

    def fake_completion(model, messages, **kwargs):
        captured["model"] = model
        message = type("Message", (), {"content": "hello"})
        choice = type("Choice", (), {"message": message})
        return type("Response", (), {"choices": [choice]})

    monkeypatch.setattr(litellm, "completion", fake_completion)
    est = BaseSkribeEstimator(model="ollama:llama3.1", verbose=False, max_train_rows=1)
    out = est._call_llm("hi")
    assert captured["model"] == "ollama/llama3.1"
    assert out == "hello"


def test_call_llm_passes_timeout(monkeypatch):
    """litellm.completion() must be called with an explicit timeout -- without
    one, a stalled provider call (observed: a Vertex AI gemini-2.5-flash-lite
    request that hung for 600s+ with no local bound) blocks that call
    indefinitely, and under --workers=1 that means the entire sequential
    benchmark queue stalls behind a single flaky request instead of failing
    fast and moving on."""
    import litellm

    captured = {}

    def fake_completion(model, messages, **kwargs):
        captured.update(kwargs)
        message = type("Message", (), {"content": "hello"})
        choice = type("Choice", (), {"message": message})
        return type("Response", (), {"choices": [choice]})

    monkeypatch.setattr(litellm, "completion", fake_completion)
    est = BaseSkribeEstimator(model="gpt-4", verbose=False, max_train_rows=1)
    est._call_llm("hi")
    assert "timeout" in captured
    assert isinstance(captured["timeout"], (int, float))
    assert captured["timeout"] > 0


def test_sanitize_description_strips_and_cleans():
    assert sanitize_dataset_description("  hello  ") == "hello"


def test_sanitize_description_removes_braces():
    result = sanitize_dataset_description("context {data} here")
    assert "{" not in result and "}" not in result
    assert "context" in result and "data" in result and "here" in result


def test_sanitize_description_no_length_cap():
    long = "x" * 600
    result = sanitize_dataset_description(long)
    assert len(result) == 600  # no truncation — context window handles sizing


def test_sanitize_description_collapses_whitespace():
    assert sanitize_dataset_description("a  b\t\tc") == "a b c"


def test_truncate_no_op_when_fits(monkeypatch):
    """When the prompt fits in the context window, df is returned unchanged."""
    import litellm

    est = BaseSkribeEstimator(model="gpt-4o", verbose=False, max_train_rows=None)
    monkeypatch.setattr(
        litellm, "get_model_info", lambda m: {"max_input_tokens": 128_000}
    )
    monkeypatch.setattr(litellm, "token_counter", lambda **kw: 100)

    df = pd.DataFrame({"x": range(50)})
    result = est._truncate_to_context_window(df, "prompt {data} end")
    assert len(result) == 50


def test_truncate_warns_and_reduces(monkeypatch):
    """When the prompt exceeds the budget, a UserWarning is raised and rows are dropped."""
    import litellm

    est = BaseSkribeEstimator(model="gpt-4o", verbose=False, max_train_rows=None)
    monkeypatch.setattr(
        litellm, "get_model_info", lambda m: {"max_input_tokens": 1000}
    )
    # First call (full df) exceeds budget (>920 = 1000*0.92); ≤25 rows fits.
    call_count = [0]

    def fake_counter(**kw):
        call_count[0] += 1
        content = kw["messages"][0]["content"]
        # Proxy for row count: count newlines in the content after the header
        rows_in_csv = content.count("\n") - 1  # subtract header row
        return 950 if rows_in_csv > 25 else 500

    monkeypatch.setattr(litellm, "token_counter", fake_counter)

    df = pd.DataFrame({"x": range(100)})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = est._truncate_to_context_window(df, "{data}")

    assert len(result) < 100
    assert any("Truncating" in str(warning.message) for warning in w)


def test_truncate_skips_check_when_model_unknown(monkeypatch):
    """If get_model_info raises, truncation is skipped with a warning (no crash)."""
    import litellm

    est = BaseSkribeEstimator(model="unknown-model", verbose=False, max_train_rows=None)
    monkeypatch.setattr(litellm, "get_model_info", lambda m: (_ for _ in ()).throw(Exception("unknown")))

    df = pd.DataFrame({"x": range(20)})
    result = est._truncate_to_context_window(df, "{data}")
    assert len(result) == 20


def test_max_train_rows_none_default():
    """max_train_rows defaults to None (no hard cap)."""
    from skribe import SkribeClassifier, SkribeRegressor
    assert SkribeClassifier().max_train_rows is None
    assert SkribeRegressor().max_train_rows is None


def test_max_train_rows_explicit_still_caps(monkeypatch):
    """When max_train_rows is set, data is still capped before the context check."""
    import litellm
    from skribe import SkribeClassifier

    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, max_train_rows=5)
    monkeypatch.setattr(litellm, "get_model_info", lambda m: {"max_input_tokens": 1_000_000})
    monkeypatch.setattr(litellm, "token_counter", lambda **kw: 10)

    prompts_seen = []

    def fake_llm(prompt, web_search=False):
        prompts_seen.append(prompt)
        return "def predict(**f): return 0"

    monkeypatch.setattr(clf, "_call_llm", fake_llm)

    X = pd.DataFrame({"a": range(50)})
    y = pd.Series([0] * 25 + [1] * 25)
    clf.fit(X, y)

    # The fit prompt is stored on the estimator; check the CSV section has ≤5 data rows.
    fit_prompt = clf.fit_prompt_
    # CSV portion: everything after "Data:\n"
    csv_section = fit_prompt.split("Data:\n", 1)[-1].strip()
    data_rows = [l for l in csv_section.splitlines() if l.strip()][1:]  # skip header
    assert len(data_rows) <= 5


def test_extend_code_handles_llm_failure(monkeypatch):
    class DummyEstimator(BaseSkribeEstimator):
        def __init__(self):
            super().__init__(model="dummy-model", verbose=False, max_train_rows=10)

        def _call_llm(self, prompt: str):
            raise RuntimeError("Mocked LLM failure")

    estimator = DummyEstimator()
    # Should log a warning and return original code unchanged
    result = estimator._extend_code("def predict(**features): return 42")
    assert result.strip() == "def predict(**features): return 42"
