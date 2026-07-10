"""Tests for _call_llm's handling of litellm.RateLimitError (HTTP 429).

A 429 means transient quota exhaustion (requests/tokens per minute), not
that the request itself was bad -- the right response is to wait and resend
the *same* prompt, not give up after one sleep.
"""

import httpx
import litellm
import pytest

from skribe.classifier import SkribeClassifier


def _rate_limit_error(model="gemini-2.5-flash", provider="vertex_ai"):
    resp = httpx.Response(status_code=429, request=httpx.Request("POST", "https://example.com"))
    return litellm.RateLimitError(
        message="Resource exhausted. Please try again later.",
        llm_provider=provider,
        model=model,
        response=resp,
    )


def test_rate_limit_retries_and_succeeds(monkeypatch):
    """A RateLimitError on the first attempt must be retried with the same
    prompt -- not immediately surfaced as a fatal error -- and succeed once
    the underlying call stops failing."""
    clf = SkribeClassifier(model="vertex_ai/gemini-2.5-flash", verbose=False)
    monkeypatch.setattr("time.sleep", lambda _: None)

    calls = {"n": 0}

    def fake_completion(model, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _rate_limit_error()
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.choices[0].message.content = "def predict(**f): return 0"
        resp.choices[0].finish_reason = "stop"
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)

    result = clf._call_llm("some prompt")

    assert result == "def predict(**f): return 0"
    assert calls["n"] == 2


def test_rate_limit_gives_up_after_max_retries(monkeypatch):
    """A RateLimitError that never clears must eventually raise -- the retry
    loop is bounded, not infinite -- and the final error must be surfaced as
    a RuntimeError like other LLM call failures."""
    clf = SkribeClassifier(model="vertex_ai/gemini-2.5-flash", verbose=False)
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    def fake_completion(model, messages, **kwargs):
        calls["n"] += 1
        raise _rate_limit_error()

    monkeypatch.setattr("litellm.completion", fake_completion)

    with pytest.raises(RuntimeError, match="LLM call failed"):
        clf._call_llm("some prompt")

    # 1 initial attempt + _MAX_RATE_LIMIT_RETRIES retries, each retry preceded
    # by a sleep (the final failed attempt does not sleep again).
    from skribe.base import _MAX_RATE_LIMIT_RETRIES

    assert calls["n"] == _MAX_RATE_LIMIT_RETRIES + 1
    assert len(sleeps) == _MAX_RATE_LIMIT_RETRIES


def test_rate_limit_retry_logged_in_fit_log(monkeypatch):
    """Every rate-limit attempt (including ones that get retried) must show
    up in fit_log_ -- so a cached result reveals that a run hit quota
    exhaustion and needed retries, not just the final accuracy."""
    clf = SkribeClassifier(model="vertex_ai/gemini-2.5-flash", verbose=False)
    monkeypatch.setattr("time.sleep", lambda _: None)

    calls = {"n": 0}

    def fake_completion(model, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _rate_limit_error()
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.choices[0].message.content = "def predict(**f): return 0"
        resp.choices[0].finish_reason = "stop"
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)

    clf._call_llm("some prompt")

    rate_limit_entries = [e for e in clf.fit_log_ if "RateLimitError" in e.get("error", "")]
    assert len(rate_limit_entries) == 2
    assert all(e["retrying"] for e in rate_limit_entries)
