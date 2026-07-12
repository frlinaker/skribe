"""Tests for dataset_description and web_search features."""

from unittest.mock import MagicMock

import pandas as pd

from skribe import SkribeClassifier, SkribeRegressor

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
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=False)
    calls = []

    def fake_call_llm(prompt, web_search=False, **kwargs):
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
    reg = SkribeRegressor(model="gpt-5.4-mini", verbose=False, context_prepass=False)
    calls = []

    def fake_call_llm(prompt, web_search=False, **kwargs):
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
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=False)

    def fake_call_llm(prompt, web_search=False, **kwargs):
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
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, web_search=True)

    def fake_call_llm(prompt, web_search=False, **kwargs):
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert "search the web" in clf.fit_prompt_.lower()


def test_no_description_prompt_unchanged(monkeypatch):
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False)
    captured = {}

    def fake_call_llm(prompt, web_search=False, **kwargs):
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


def test_context_prepass_caps_high_cardinality_column_preview(monkeypatch):
    """A column with many unique values must not blow up the pre-pass prompt.

    Reproduces a real failure: fitting on the spotify-genre dataset (track_name
    has 23,449 unique values across 26,229 rows) sent a 391,893-token pre-pass
    prompt to gpt-4o, which only has a 128,000-token window — the pre-pass
    call failed every time. _build_dataset_context() listed every unique value
    per column with no cap, so a single high-cardinality (or near-continuous)
    column dominates prompt size regardless of how many training rows are used.
    """
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    calls = []

    def fake_call_llm(prompt, web_search=False, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return "This is a clean summary."  # pre-pass response
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    # 500 distinct "track names" — far more than any reasonable cap.
    track_names = [f"track_{i}" for i in range(500)]
    X = pd.DataFrame({"track_name": track_names})
    y = pd.Series([0, 1] * 250)
    clf.fit(X, y, dataset_description="Track genre classification.")

    prepass_prompt = calls[0]
    # Only a bounded preview of the 500 values should appear, not all of them.
    assert "track_499" not in prepass_prompt
    assert "more unique values" in prepass_prompt


def test_context_prepass_fires_before_fit(monkeypatch):
    """When context_prepass=True and dataset_description is given, an extra LLM
    call is made before the main fit call, and context_summary_ is set."""
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    calls = []

    def fake_call_llm(prompt, web_search=False, **kwargs):
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
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    SUMMARY = "CLEAN SUMMARY: predicts income class."
    call_count = [0]

    def fake_call_llm(prompt, web_search=False, **kwargs):
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
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=False)

    def fake_call_llm(prompt, web_search=False, **kwargs):
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"age": [25, 40]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="UCI Adult income dataset.")

    assert clf.context_summary_ is None
    assert "UCI Adult" in clf.fit_prompt_


def test_context_prepass_no_description_skipped(monkeypatch):
    """Pre-pass is skipped entirely when no dataset_description is given."""
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    calls = []

    def fake_call_llm(prompt, web_search=False, **kwargs):
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
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, web_search=True, context_prepass=True)
    prepass_web_search = []

    def fake_call_llm(prompt, web_search=False, **kwargs):
        if "preparing a structured dataset summary" in prompt:
            prepass_web_search.append(web_search)
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="Some dataset.")

    assert prepass_web_search == [True]


def test_context_prepass_states_true_label_mapping(monkeypatch):
    """fit() must accept raw string class labels directly and have the
    pre-pass prompt state the REAL label for each class — not a bare
    integer code the LLM has to guess the meaning of.

    Before this fix, skribe's public contract required y to already be
    int-coded by the caller (predict() must return an int; see
    DEFAULT_CLASSIFICATION_PROMPT_TEMPLATE and _validate_predict_fn), so any
    caller with string labels — e.g. benchmarks/benchmark_utils.py's
    ``classes = {c: i for i, c in enumerate(sorted(y.unique()))}`` — had to
    encode away the original names before fit() ever saw them. By the time
    _build_dataset_context() ran, the target column held nothing but bare
    ints (0, 1, 2), so the pre-pass LLM had to guess what each meant and
    models converged on "intuitive" but wrong guesses (e.g. common-sense
    ordering instead of the true sorted-alphabetical one).
    skribe should do this encoding itself, internally, so it always knows
    (and can state) the true label for every class code — callers should be
    able to just pass their original labels straight to fit().
    """
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    calls = []

    def fake_call_llm(prompt, web_search=False, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return "This is a clean summary."  # pre-pass response
        return SIMPLE_CODE  # fit + extend responses

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"legs": [4, 2, 0], "feathers": [False, True, False]})
    # Raw string labels, passed straight in — no pre-encoding by the caller.
    y = pd.Series(["mammal", "bird", "fish"])

    clf.fit(X, y, dataset_description="Zoo animal classification.")

    prepass_prompt = calls[0]
    for label in ("mammal", "bird", "fish"):
        assert label in prepass_prompt, (
            f"Pre-pass prompt never mentions the true class label {label!r} — "
            "fit() must thread the true label mapping into the context "
            "pre-pass instead of leaving only bare integer codes to guess from."
        )

    # predict() must still return integers (unchanged public contract), and
    # classes_ must expose the true sorted label-to-code mapping so callers
    # can recover the original label from a prediction.
    assert list(clf.classes_) == ["bird", "fish", "mammal"]


def test_fit_prompt_states_label_mapping_even_if_prepass_omits_it(monkeypatch):
    """The code-generation prompt (fit_prompt_) must state the literal
    code->label mapping for the target column, unconditionally — not just
    rely on the context pre-pass to have mentioned it.

    In a real run (gpt-4.1-web on the zoo dataset), the pre-pass summary
    correctly listed the 7 true label names but never restated which
    integer code corresponds to which label — it's a second LLM call
    producing free text, and it's not guaranteed to preserve that explicit
    correspondence even when the labels are right. The training data CSV
    itself only ever contains bare ints, so without an explicit, deterministic
    "code=label" line in the main prompt, the code-generation LLM still has
    to guess the mapping — reproducing the exact bug this fix targets.
    """
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    calls = []

    def fake_call_llm(prompt, web_search=False, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            # Pre-pass summary lists the labels but NOT the code mapping —
            # exactly what happened in the real gpt-4.1-web zoo run.
            return "This dataset predicts the animal class: amphibian, bird, fish, mammal."
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"legs": [4, 2, 0, 8]})
    y = pd.Series(["mammal", "bird", "fish", "amphibian"])
    clf.fit(X, y, dataset_description="Zoo animal classification.")

    fit_prompt = clf.fit_prompt_
    for code, label in enumerate(["amphibian", "bird", "fish", "mammal"]):
        assert f'{code}="{label}"' in fit_prompt, (
            f"fit_prompt_ never states the explicit mapping {code}={label!r} — "
            "the code-generation LLM only sees bare integers in the training "
            "data and has nothing authoritative to tie them back to real labels."
        )


def test_fit_prompt_mapping_line_unambiguous_when_labels_look_numeric(monkeypatch):
    """When the original dataset labels are themselves numeric-looking
    strings (e.g. OpenML datasets that store category codes as '1', '2'),
    the mapping line must not read as if the LABEL is the value to return.

    Reproduces a real regression: bank-marketing's true labels are the
    strings '1' (no subscription) and '2' (subscription), encoded to codes
    0 and 1. An earlier, bare "0=1, 1=2" phrasing was indistinguishable from
    a code->code mapping, and gpt-5.5 responded by writing `return 1` /
    `return 2` (the label) instead of `return 0` / `return 1` (the actual
    training code) — accuracy on bank-marketing collapsed from 0.90 to 0.08.
    """
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=False)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"age": [40, 50, 60, 30]})
    y = pd.Series(["1", "2", "1", "2"])
    clf.fit(X, y)

    fit_prompt = clf.fit_prompt_
    # The original labels must be visually distinguished (quoted) from the
    # bare training codes, and the prompt must explicitly say the function
    # returns the code, not the label.
    assert '0="1"' in fit_prompt
    assert '1="2"' in fit_prompt
    assert (
        "never return the original label" in fit_prompt or "never the original label" in fit_prompt
    )


def test_context_prepass_fallback_on_llm_failure(monkeypatch):
    """If the pre-pass LLM call fails, fit() continues with the sanitized raw description."""
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    call_count = [0]

    def fake_call_llm(prompt, web_search=False, **kwargs):
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
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, web_search=True)
    captured = []

    def fake_call_llm(prompt, web_search=False, reasoning_effort=None):
        captured.append(web_search)
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    # Code-gen's first attempt and the extend pass's first attempt both get
    # web_search=True (expanding categorical mappings is a lookup task web
    # search directly helps with) -- everything else stays False.
    assert captured[0] is True  # code-gen, attempt 0
    assert captured[1] is True  # extend, attempt 0
    assert all(not v for v in captured[2:])


def test_web_search_false_by_default(monkeypatch):
    clf = SkribeClassifier(model="gpt-5.5", verbose=False)
    captured = []

    def fake_call_llm(prompt, web_search=False, **kwargs):
        captured.append(web_search)
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert all(not v for v in captured)


def test_web_search_prompt_prefix(monkeypatch):
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, web_search=True)
    captured = {}

    def fake_call_llm(prompt, web_search=False, **kwargs):
        if not captured:
            captured["prompt"] = prompt
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert "search the web" in captured["prompt"].lower()


def test_web_search_prompt_states_precedence_over_stated_mapping(monkeypatch):
    """The web-search instruction must tell the model the explicit code->label
    mapping stated in the prompt always wins over anything it finds/recalls
    while searching -- otherwise a search-reinforced wrong guess can override
    a correct, already-stated mapping (the same failure mode the removed
    anti-fabrication guard used to target, but scoped to search specifically)."""
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, web_search=True)
    captured = {}

    def fake_call_llm(prompt, web_search=False, **kwargs):
        if not captured:
            captured["prompt"] = prompt
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    prompt_lower = captured["prompt"].lower()
    assert "always win" in prompt_lower or "never" in prompt_lower


def test_web_search_unsupported_model_warns(monkeypatch, caplog):
    import logging

    clf = SkribeClassifier(model="claude-sonnet-4-6", verbose=False, web_search=True)

    calls = []

    def fake_completion(model, messages, **kwargs):
        calls.append(kwargs)
        resp = MagicMock()
        resp.choices[0].message.content = SIMPLE_CODE
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])

    with caplog.at_level(logging.WARNING, logger="skribe"):
        clf.fit(X, y)

    assert "not in the known supported list" in caplog.text
    # web_search_options should NOT have been passed to litellm
    assert all("web_search_options" not in c for c in calls)


def test_web_search_supported_model_passes_options(monkeypatch):
    """Chat Completions path: gpt-4o-search-preview gets web_search_options."""
    clf = SkribeClassifier(model="gpt-4o-search-preview", verbose=False, web_search=True)

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
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, web_search=True)

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


def test_web_search_bare_model_name_matches_prefixed_registry_entry(monkeypatch):
    """gpt-5.6-sol is registered as "openai/gpt-5.6-sol" (to match how the
    benchmark harness invokes it), but a user typing the bare model name --
    e.g. SkribeClassifier(model="gpt-5.6-sol", web_search=True), as shown in
    the docs -- must still get web_search wired up. Regression test for a bug
    where the exact-string membership check silently dropped web_search for
    any spelling that didn't match the registry's exact prefix."""
    clf = SkribeClassifier(model="gpt-5.6-sol", verbose=False, web_search=True)

    responses_calls = []

    def fake_responses(prompt, model, **kwargs):
        responses_calls.append({"model": model, "tools": kwargs.get("tools")})
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
    assert {"type": "web_search"} in responses_calls[0]["tools"]


def test_context_prepass_requests_high_search_context_size(monkeypatch):
    """The context pre-pass call (most likely to benefit from finding real
    dataset documentation) should ask for the Responses API's highest
    search_context_size tier -- code-gen/extend calls don't need this."""
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, web_search=True, context_prepass=True)

    responses_calls = []

    def fake_responses(prompt, model, **kwargs):
        responses_calls.append(kwargs.get("tools"))
        msg = MagicMock()
        msg.type = "message"
        content_part = MagicMock()
        content_part.type = "output_text"
        content_part.text = SIMPLE_CODE
        content_part.annotations = []
        msg.content = [content_part]
        resp = MagicMock()
        resp.output = [msg]
        return resp

    monkeypatch.setattr("litellm.responses", fake_responses)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="Some dataset about widgets.")

    # First Responses call is the context pre-pass.
    prepass_tools = responses_calls[0]
    assert prepass_tools[0]["search_context_size"] == "high"

    # A later call (code-gen) should NOT carry the pre-pass-specific config.
    assert not any(tools[0].get("search_context_size") for tools in responses_calls[1:] if tools)


def test_web_search_responses_api_records_evidence_in_fit_log(monkeypatch):
    """Responses API path: web_search_call events and url citations land in fit_log_."""
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, web_search=True)

    def fake_responses(prompt, model, **kwargs):
        search_call = MagicMock()
        search_call.type = "web_search_call"

        msg = MagicMock()
        msg.type = "message"
        citation = MagicMock()
        citation.url = "https://archive.ics.uci.edu/dataset/example"
        content_part = MagicMock()
        content_part.type = "output_text"
        content_part.text = SIMPLE_CODE
        content_part.annotations = [citation]
        msg.content = [content_part]

        resp = MagicMock()
        resp.output = [search_call, msg]
        return resp

    monkeypatch.setattr("litellm.responses", fake_responses)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    web_search_entries = [e for e in clf.fit_log_ if e.get("stage") == "web_search"]
    assert web_search_entries, "expected at least one web_search entry in fit_log_"
    assert web_search_entries[0]["search_call_count"] == 1
    assert web_search_entries[0]["citations"] == ["https://archive.ics.uci.edu/dataset/example"]


def test_web_search_chat_completions_records_citations_in_fit_log(monkeypatch):
    """Chat Completions path (e.g. Gemini grounding): citations land in fit_log_."""
    clf = SkribeClassifier(model="vertex_ai/gemini-3.5-flash", verbose=False, web_search=True)

    def fake_completion(model, messages, **kwargs):
        resp = MagicMock()
        resp.choices[0].message.content = SIMPLE_CODE
        resp.choices[0].finish_reason = "stop"
        resp.choices[0].message.annotations = [
            {"url_citation": {"url": "https://en.wikipedia.org/wiki/Example"}}
        ]
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    web_search_entries = [e for e in clf.fit_log_ if e.get("stage") == "web_search"]
    assert web_search_entries, "expected at least one web_search entry in fit_log_"
    assert web_search_entries[0]["citations"] == ["https://en.wikipedia.org/wiki/Example"]


def test_no_web_search_evidence_logged_when_web_search_disabled(monkeypatch):
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, web_search=False)

    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert not [e for e in clf.fit_log_ if e.get("stage") == "web_search"]


# ---------------------------------------------------------------------------
# reasoning_effort
# ---------------------------------------------------------------------------


def test_reasoning_effort_none_by_default(monkeypatch):
    """No reasoning_effort kwarg reaches litellm.completion when unset."""
    clf = SkribeClassifier(model="gpt-5.5", verbose=False)

    calls = []

    def fake_completion(model, messages, **kwargs):
        calls.append(kwargs)
        resp = MagicMock()
        resp.choices[0].message.content = SIMPLE_CODE
        resp.choices[0].finish_reason = "stop"
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert all("reasoning_effort" not in c for c in calls)


def test_reasoning_effort_passed_to_chat_completions(monkeypatch):
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, reasoning_effort="high")

    calls = []

    def fake_completion(model, messages, **kwargs):
        calls.append(kwargs)
        resp = MagicMock()
        resp.choices[0].message.content = SIMPLE_CODE
        resp.choices[0].finish_reason = "stop"
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert all(c.get("reasoning_effort") == "high" for c in calls)


def test_reasoning_effort_passed_to_responses_api(monkeypatch):
    clf = SkribeClassifier(
        model="gpt-5.5", verbose=False, web_search=True, reasoning_effort="xhigh"
    )

    responses_calls = []

    def fake_responses(prompt, model, **kwargs):
        responses_calls.append(kwargs)
        msg = MagicMock()
        msg.type = "message"
        content_part = MagicMock()
        content_part.type = "output_text"
        content_part.text = SIMPLE_CODE
        content_part.annotations = []
        msg.content = [content_part]
        resp = MagicMock()
        resp.output = [msg]
        return resp

    monkeypatch.setattr("litellm.responses", fake_responses)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert len(responses_calls) >= 1
    assert responses_calls[0]["reasoning_effort"] == "xhigh"


def test_reasoning_effort_get_params_roundtrip():
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, reasoning_effort="medium")
    assert clf.get_params()["reasoning_effort"] == "medium"


def test_reasoning_effort_max_routes_to_responses_api(monkeypatch):
    """ "max" effort is Responses-API-only (Chat Completions rejects it), so it
    must route there automatically -- without requiring web_search=True, which
    is an unrelated feature."""
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, reasoning_effort="max")

    completion_calls = []

    def fake_completion(model, messages, **kwargs):
        completion_calls.append(kwargs)
        resp = MagicMock()
        resp.choices[0].message.content = SIMPLE_CODE
        resp.choices[0].finish_reason = "stop"
        return resp

    responses_calls = []

    def fake_responses(prompt, model, **kwargs):
        responses_calls.append(kwargs)
        msg = MagicMock()
        msg.type = "message"
        content_part = MagicMock()
        content_part.type = "output_text"
        content_part.text = SIMPLE_CODE
        content_part.annotations = []
        msg.content = [content_part]
        resp = MagicMock()
        resp.output = [msg]
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)
    monkeypatch.setattr("litellm.responses", fake_responses)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert not completion_calls
    assert len(responses_calls) >= 1
    assert responses_calls[0]["reasoning_effort"] == "max"


def test_reasoning_mode_pro_sent_as_nested_dict_with_effort(monkeypatch):
    """reasoning_mode has no Chat Completions equivalent -- setting it must
    force Responses routing and be sent alongside reasoning_effort as the
    nested {"effort": ..., "mode": ...} dict the real API expects."""
    clf = SkribeClassifier(
        model="gpt-5.5", verbose=False, reasoning_effort="max", reasoning_mode="pro"
    )

    completion_calls = []

    def fake_completion(model, messages, **kwargs):
        completion_calls.append(kwargs)
        resp = MagicMock()
        resp.choices[0].message.content = SIMPLE_CODE
        resp.choices[0].finish_reason = "stop"
        return resp

    responses_calls = []

    def fake_responses(prompt, model, **kwargs):
        responses_calls.append(kwargs)
        msg = MagicMock()
        msg.type = "message"
        content_part = MagicMock()
        content_part.type = "output_text"
        content_part.text = SIMPLE_CODE
        content_part.annotations = []
        msg.content = [content_part]
        resp = MagicMock()
        resp.output = [msg]
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)
    monkeypatch.setattr("litellm.responses", fake_responses)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert not completion_calls
    assert len(responses_calls) >= 1
    assert responses_calls[0]["reasoning_effort"] == {"effort": "max", "mode": "pro"}


def test_reasoning_mode_without_explicit_effort_defaults_to_medium(monkeypatch):
    """reasoning_mode="pro" with no reasoning_effort set should still route to
    Responses and default the effort sub-field to "medium", matching the real
    API's own default when effort is omitted."""
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, reasoning_mode="pro")

    responses_calls = []

    def fake_responses(prompt, model, **kwargs):
        responses_calls.append(kwargs)
        msg = MagicMock()
        msg.type = "message"
        content_part = MagicMock()
        content_part.type = "output_text"
        content_part.text = SIMPLE_CODE
        content_part.annotations = []
        msg.content = [content_part]
        resp = MagicMock()
        resp.output = [msg]
        return resp

    monkeypatch.setattr("litellm.responses", fake_responses)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert len(responses_calls) >= 1
    assert responses_calls[0]["reasoning_effort"] == {"effort": "medium", "mode": "pro"}


def test_reasoning_mode_get_params_roundtrip():
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, reasoning_mode="pro")
    assert clf.get_params()["reasoning_mode"] == "pro"


def test_responses_api_incomplete_status_triggers_retry(monkeypatch):
    """status='incomplete' (Responses-API truncation signal) must raise
    _OutputTruncated and trigger the same shrink-and-retry path Chat
    Completions gets via finish_reason='length' -- otherwise a max-effort
    call that gets cut off would silently return partial code."""
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, reasoning_effort="max")

    responses_calls = []

    def fake_responses(prompt, model, **kwargs):
        responses_calls.append(kwargs)
        msg = MagicMock()
        msg.type = "message"
        content_part = MagicMock()
        content_part.type = "output_text"
        content_part.annotations = []
        msg.content = [content_part]
        resp = MagicMock()
        resp.output = [msg]
        if len(responses_calls) == 1:
            content_part.text = "def predict(**f):\n    return 0  # cut off mid"
            resp.status = "incomplete"
            resp.incomplete_details = MagicMock(reason="max_output_tokens")
        else:
            content_part.text = SIMPLE_CODE
            resp.status = "completed"
            resp.incomplete_details = None
        return resp

    monkeypatch.setattr("litellm.responses", fake_responses)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert len(responses_calls) >= 2


def test_responses_api_context_window_exceeded_triggers_shrink_and_retry(monkeypatch):
    """The Responses API branch had no try/except at all around
    litellm.responses() -- any BadRequestError (including a context-window
    rejection) propagated straight past _call_llm's retry loop and failed the
    whole fit, instead of getting the same shrink-and-retry treatment the
    Chat Completions branch already has via litellm.ContextWindowExceededError.
    This reproduces a real failure seen with openai/gpt-5.6-sol-web on larger
    datasets (adult, bank-marketing, spotify-genre), where OpenAI's error used
    a phrasing ("Your input exceeds the context window of this model.") that
    litellm's own classifier doesn't recognize either, so it surfaced as a
    bare BadRequestError rather than litellm.ContextWindowExceededError."""
    import litellm

    clf = SkribeClassifier(model="gpt-5.5", verbose=False, reasoning_effort="max")

    responses_calls = []

    def fake_responses(prompt, model, **kwargs):
        responses_calls.append(kwargs)
        if len(responses_calls) == 1:
            raise litellm.BadRequestError(
                message=(
                    "Your input exceeds the context window of this model. "
                    "Please adjust your input and try again."
                ),
                model=model,
                llm_provider="openai",
            )
        msg = MagicMock()
        msg.type = "message"
        content_part = MagicMock()
        content_part.type = "output_text"
        content_part.text = SIMPLE_CODE
        content_part.annotations = []
        msg.content = [content_part]
        resp = MagicMock()
        resp.output = [msg]
        resp.status = "completed"
        resp.incomplete_details = None
        return resp

    monkeypatch.setattr("litellm.responses", fake_responses)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert len(responses_calls) >= 2


def test_web_search_reenabled_on_final_retry_attempt(monkeypatch):
    """A retry caused by a knowledge gap (e.g. bad value mapping) benefits from
    a lookup exactly as much as the first attempt -- so the last retry should
    get web_search=True again, even though middle retries don't. Tests
    _generate_code directly to avoid the extend pass's own _call_llm calls
    (also web_search-aware) muddying which call is which."""
    clf = SkribeClassifier(model="gpt-5.5", verbose=False, max_retries=2)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)

    captured = []
    responses = [
        "def predict(**f): return undefined_name",  # attempt 0: NameError on validate
        "def predict(**f): return also_undefined",  # attempt 1: NameError on validate
        SIMPLE_CODE,  # attempt 2 (final): valid
    ]

    def fake_call_llm(prompt, web_search=False, reasoning_effort=None):
        captured.append(web_search)
        return responses[len(captured) - 1]

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    clf._generate_code("base prompt", [], [], web_search=True)

    # attempt 0 (index 0) and the final attempt (index 2) get web_search=True;
    # the middle retry (index 1) does not.
    assert captured == [True, False, True]
