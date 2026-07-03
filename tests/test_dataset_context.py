"""Tests for dataset_description and web_search features."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from promptlearn import PromptClassifier, PromptRegressor

SIMPLE_CODE = "def predict(**f): return 0"


def _mock_llm(code=SIMPLE_CODE):
    """Return a mock that makes _call_llm return valid Python code."""
    m = MagicMock(return_value=code)
    return m


# ---------------------------------------------------------------------------
# dataset_description
# ---------------------------------------------------------------------------


def test_classifier_description_in_prompt(monkeypatch):
    # context_prepass=False so _call_llm is only called for fit + extend, not pre-pass.
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=False)
    calls = []

    def fake_call_llm(prompt, web_search=False):
        calls.append({"prompt": prompt, "web_search": web_search})
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"age": [25, 40], "income": [30000, 80000]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="UCI Adult: predict income >50k.")

    first = calls[0]
    assert "Dataset context" in first["prompt"]
    assert "UCI Adult" in first["prompt"]
    assert first["web_search"] is False


def test_regressor_description_in_prompt(monkeypatch):
    # context_prepass=False so the raw description flows straight into the prompt.
    reg = PromptRegressor(model="gpt-5.4-mini", verbose=False, context_prepass=False)
    calls = []

    def fake_call_llm(prompt, web_search=False):
        calls.append({"prompt": prompt})
        return "def predict(**f): return 1.0"

    monkeypatch.setattr(reg, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"height_m": [1.0, 5.0]})
    y = pd.Series([0.45, 1.01])
    reg.fit(X, y, dataset_description="Falling body: predict fall time in seconds.")

    assert "Dataset context" in calls[0]["prompt"]
    assert "Falling body" in calls[0]["prompt"]


def test_description_precedes_instructions(monkeypatch):
    """Dataset context block must appear after task instructions but before Data: in fit_prompt_."""
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=False)

    def fake_call_llm(prompt, web_search=False):
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"age": [25, 40], "income": [30000, 80000]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="UCI Adult: predict income >50k.")

    fit_prompt = clf.fit_prompt_
    instructions_pos = fit_prompt.index("Output a single valid Python")
    context_pos = fit_prompt.index("Dataset context")
    data_pos = fit_prompt.index("Data:")
    assert instructions_pos < context_pos < data_pos


def test_fit_prompt_includes_web_search_prefix(monkeypatch):
    """fit_prompt_ must reflect the actual prompt sent, including the web search prefix."""
    clf = PromptClassifier(model="gpt-5.5", verbose=False, web_search=True)

    def fake_call_llm(prompt, web_search=False):
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert "search the web" in clf.fit_prompt_.lower()


def test_no_description_prompt_unchanged(monkeypatch):
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False)
    captured = {}

    def fake_call_llm(prompt, web_search=False):
        captured["prompt"] = prompt
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert "Dataset context:" not in captured["prompt"]


# ---------------------------------------------------------------------------
# context_prepass
# ---------------------------------------------------------------------------


def test_context_prepass_fires_before_fit(monkeypatch):
    """When context_prepass=True and dataset_description is given, an extra LLM
    call is made before the main fit call, and context_summary_ is set."""
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    calls = []

    def fake_call_llm(prompt, web_search=False):
        calls.append(prompt)
        if len(calls) == 1:
            return "This is a clean summary."  # pre-pass response
        return SIMPLE_CODE  # fit + extend responses

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"age": [25, 40], "income": [30000, 80000]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="UCI Adult income dataset.")

    # At least 2 calls: pre-pass + fit (+ possibly extend)
    assert len(calls) >= 2
    # Pre-pass prompt contains the raw description
    assert "UCI Adult" in calls[0]
    # context_summary_ was set
    assert clf.context_summary_ is not None


def test_context_prepass_summary_appears_in_fit_prompt(monkeypatch):
    """The pre-pass output replaces the raw description in the main prompt."""
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    SUMMARY = "CLEAN SUMMARY: predicts income class."
    call_count = [0]

    def fake_call_llm(prompt, web_search=False):
        call_count[0] += 1
        if call_count[0] == 1:
            return SUMMARY  # pre-pass
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"age": [25, 40], "income": [30000, 80000]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="UCI Adult income dataset.")

    assert SUMMARY in clf.fit_prompt_
    # Raw description should not appear verbatim — only the summary does
    assert "UCI Adult income dataset." not in clf.fit_prompt_


def test_context_prepass_disabled_uses_raw_description(monkeypatch):
    """With context_prepass=False, the raw description (sanitized) goes straight into
    the prompt and context_summary_ stays None."""
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=False)

    def fake_call_llm(prompt, web_search=False):
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"age": [25, 40]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="UCI Adult income dataset.")

    assert clf.context_summary_ is None
    assert "UCI Adult" in clf.fit_prompt_


def test_context_prepass_no_description_skipped(monkeypatch):
    """Pre-pass is skipped entirely when no dataset_description is given."""
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    calls = []

    def fake_call_llm(prompt, web_search=False):
        calls.append(prompt)
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)  # no dataset_description

    assert clf.context_summary_ is None
    # Only fit + extend calls, no pre-pass
    assert not any("preparing a structured dataset summary" in c for c in calls)


def test_context_prepass_web_search_forwarded(monkeypatch):
    """When web_search=True, the pre-pass call also gets web_search=True."""
    clf = PromptClassifier(
        model="gpt-5.5", verbose=False, web_search=True, context_prepass=True
    )
    prepass_web_search = []

    def fake_call_llm(prompt, web_search=False):
        if "preparing a structured dataset summary" in prompt:
            prepass_web_search.append(web_search)
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="Some dataset.")

    assert prepass_web_search == [True]


def test_context_prepass_fallback_on_llm_failure(monkeypatch):
    """If the pre-pass LLM call fails, fit() continues with the sanitized raw description."""
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    call_count = [0]

    def fake_call_llm(prompt, web_search=False):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("network error")
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"age": [25, 40]})
    y = pd.Series([0, 1])
    # Should not raise — falls back to raw description
    clf.fit(X, y, dataset_description="UCI Adult income dataset.")

    assert "UCI Adult" in clf.fit_prompt_


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


def test_web_search_passed_to_call_llm(monkeypatch):
    clf = PromptClassifier(model="gpt-5.5", verbose=False, web_search=True)
    captured = []

    def fake_call_llm(prompt, web_search=False):
        captured.append(web_search)
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    # First call (fit) should have web_search=True; extend call should not.
    assert captured[0] is True
    assert all(not v for v in captured[1:])


def test_web_search_false_by_default(monkeypatch):
    clf = PromptClassifier(model="gpt-5.5", verbose=False)
    captured = []

    def fake_call_llm(prompt, web_search=False):
        captured.append(web_search)
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert all(not v for v in captured)


def test_web_search_prompt_prefix(monkeypatch):
    clf = PromptClassifier(model="gpt-5.5", verbose=False, web_search=True)
    captured = {}

    def fake_call_llm(prompt, web_search=False):
        if not captured:
            captured["prompt"] = prompt
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert "search the web" in captured["prompt"].lower()


def test_web_search_unsupported_model_warns(monkeypatch, caplog):
    import logging

    clf = PromptClassifier(model="claude-sonnet-4-6", verbose=False, web_search=True)

    calls = []

    def fake_completion(model, messages, **kwargs):
        calls.append(kwargs)
        resp = MagicMock()
        resp.choices[0].message.content = SIMPLE_CODE
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])

    with caplog.at_level(logging.WARNING, logger="promptlearn"):
        clf.fit(X, y)

    assert "not in the known supported list" in caplog.text
    # web_search_options should NOT have been passed to litellm
    assert all("web_search_options" not in c for c in calls)


def test_web_search_supported_model_passes_options(monkeypatch):
    """Chat Completions path: gpt-4o-search-preview gets web_search_options."""
    clf = PromptClassifier(
        model="gpt-4o-search-preview", verbose=False, web_search=True
    )

    calls = []

    def fake_completion(model, messages, **kwargs):
        calls.append(kwargs)
        resp = MagicMock()
        resp.choices[0].message.content = SIMPLE_CODE
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert "web_search_options" in calls[0]


def test_web_search_responses_api_model(monkeypatch):
    """Responses API path: gpt-5.5 uses litellm.responses with web_search tool."""
    clf = PromptClassifier(model="gpt-5.5", verbose=False, web_search=True)

    responses_calls = []

    def fake_responses(prompt, model, **kwargs):
        responses_calls.append({"model": model, "tools": kwargs.get("tools")})
        # Build a minimal mock response matching ResponsesAPIResponse structure.
        msg = MagicMock()
        msg.type = "message"
        content_part = MagicMock()
        content_part.type = "output_text"
        content_part.text = SIMPLE_CODE
        msg.content = [content_part]
        resp = MagicMock()
        resp.output = [msg]
        return resp

    monkeypatch.setattr("litellm.responses", fake_responses)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert len(responses_calls) >= 1
    assert responses_calls[0]["model"] == "gpt-5.5"
    assert {"type": "web_search"} in responses_calls[0]["tools"]
