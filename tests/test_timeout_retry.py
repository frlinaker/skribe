"""Tests for _call_llm's handling of litellm.Timeout (connection/read timeout).

A timeout is a transient network stall, not evidence the request itself was
bad -- the right response is to resend the *same* prompt, not give up after
one slow round-trip.
"""

import litellm
import pytest

from skribe.classifier import SkribeClassifier


def _timeout_error(model="gemini-3.6-flash", provider="vertex_ai"):
    return litellm.Timeout(
        message="Connection timed out after None seconds.",
        model=model,
        llm_provider=provider,
    )


def test_timeout_retries_and_succeeds(monkeypatch):
    """A Timeout on the first attempt must be retried with the same prompt --
    not immediately surfaced as a fatal error -- and succeed once the
    underlying call stops failing."""
    clf = SkribeClassifier(model="vertex_ai/gemini-3.6-flash", verbose=False)
    monkeypatch.setattr("time.sleep", lambda _: None)

    calls = {"n": 0}

    def fake_completion(model, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _timeout_error()
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.choices[0].message.content = "def predict(**f): return 0"
        resp.choices[0].finish_reason = "stop"
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)

    result = clf._call_llm("some prompt")

    assert result == "def predict(**f): return 0"
    assert calls["n"] == 2


def test_timeout_gives_up_after_max_retries(monkeypatch):
    """A Timeout that never clears must eventually raise -- the retry loop
    is bounded, not infinite -- and the final error must be surfaced as a
    RuntimeError like other LLM call failures."""
    clf = SkribeClassifier(model="vertex_ai/gemini-3.6-flash", verbose=False)
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    def fake_completion(model, messages, **kwargs):
        calls["n"] += 1
        raise _timeout_error()

    monkeypatch.setattr("litellm.completion", fake_completion)

    with pytest.raises(RuntimeError, match="LLM call failed"):
        clf._call_llm("some prompt")

    # 1 initial attempt + _MAX_TIMEOUT_RETRIES retries, each retry preceded
    # by a sleep (the final failed attempt does not sleep again).
    from skribe.base import _MAX_TIMEOUT_RETRIES

    assert calls["n"] == _MAX_TIMEOUT_RETRIES + 1
    assert len(sleeps) == _MAX_TIMEOUT_RETRIES


def test_timeout_retry_logged_in_fit_log(monkeypatch):
    """Every timeout attempt (including ones that get retried) must show up
    in fit_log_ -- so a cached result reveals that a run hit transient
    timeouts and needed retries, not just the final accuracy."""
    clf = SkribeClassifier(model="vertex_ai/gemini-3.6-flash", verbose=False)
    monkeypatch.setattr("time.sleep", lambda _: None)

    calls = {"n": 0}

    def fake_completion(model, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _timeout_error()
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.choices[0].message.content = "def predict(**f): return 0"
        resp.choices[0].finish_reason = "stop"
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)

    clf._call_llm("some prompt")

    timeout_entries = [e for e in clf.fit_log_ if "Timeout" in e.get("error", "")]
    assert len(timeout_entries) == 1
    assert all(e["retrying"] for e in timeout_entries)


def test_rate_limit_and_timeout_retries_are_tracked_independently(monkeypatch):
    """Rate limits and timeouts each get their own retry budget -- a call
    that hits one of each shouldn't have the second one count against the
    first's attempt limit (or vice versa)."""
    import httpx

    clf = SkribeClassifier(model="vertex_ai/gemini-3.6-flash", verbose=False)
    monkeypatch.setattr("time.sleep", lambda _: None)

    calls = {"n": 0}

    def fake_completion(model, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            resp = httpx.Response(
                status_code=429, request=httpx.Request("POST", "https://example.com")
            )
            raise litellm.RateLimitError(
                message="Resource exhausted.",
                llm_provider="vertex_ai",
                model="gemini-3.6-flash",
                response=resp,
            )
        if calls["n"] == 2:
            raise _timeout_error()
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.choices[0].message.content = "def predict(**f): return 0"
        resp.choices[0].finish_reason = "stop"
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)

    result = clf._call_llm("some prompt")

    assert result == "def predict(**f): return 0"
    assert calls["n"] == 3
