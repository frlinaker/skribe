import warnings

import pandas as pd
import pytest

from skribe.base import BaseSkribeEstimator, _check_unresolved_names
from skribe.postprocess import ConstantPostProcessor, find_unverified_thresholds
from skribe.utils import sanitize_dataset_description


def test_get_set_params():
    est = BaseSkribeEstimator(model="gpt-4", verbose=True, max_train_rows=10)
    params = est.get_params()
    assert params["model"] == "gpt-4"
    est.set_params(model="gpt-3.5-turbo")
    assert est.model == "gpt-3.5-turbo"


def test_default_postprocessor_is_noop():
    """With no postprocessor passed, the estimator defaults to a no-op --
    constant tuning is opt-in, not applied unless a caller explicitly
    injects ConstantPostProcessor()."""
    from skribe.postprocess import NoOpPostProcessor

    est = BaseSkribeEstimator(model="gpt-4", verbose=False, max_train_rows=10)
    assert isinstance(est.postprocessor, NoOpPostProcessor)


def test_explicit_constant_post_processor_is_used():
    """A caller can still opt into constant tuning by passing
    ConstantPostProcessor() explicitly."""
    est = BaseSkribeEstimator(
        model="gpt-4",
        verbose=False,
        max_train_rows=10,
        postprocessor=ConstantPostProcessor(),
    )
    assert isinstance(est.postprocessor, ConstantPostProcessor)


def test_custom_postprocessor_is_injected_and_used(monkeypatch):
    """A caller can swap in a different postprocessing strategy (or a no-op)
    without subclassing BaseSkribeEstimator -- _generate_code must call
    whatever object was injected, not a hardcoded ConstantPostProcessor."""
    from skribe.classifier import SkribeClassifier

    calls = []

    class RecordingPostProcessor:
        def process(self, code, rows, labels, is_classification):
            calls.append(is_classification)
            return code

    clf = SkribeClassifier(model="gpt-5.4-mini", postprocessor=RecordingPostProcessor())
    assert isinstance(clf.postprocessor, RecordingPostProcessor)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)
    monkeypatch.setattr(
        clf,
        "_call_llm",
        lambda prompt, web_search=False: "def predict(**features): return 0",
    )
    clf.fit(pd.DataFrame({"a": [1, 2, 3, 4]}), pd.Series([0, 0, 1, 1], name="target"))
    assert calls == [True]  # classifier -> is_classification=True, called once


def test_postprocessor_receives_full_training_data_not_prompt_sample(monkeypatch):
    """The postprocessor does pure local fitting (no LLM, no token budget),
    so it must be handed the full max_train_rows-capped training set, not
    whatever smaller slice got truncated into the LLM's prompt to fit the
    context window -- otherwise it would be needlessly limited by a
    constraint that doesn't apply to it."""
    from skribe.classifier import SkribeClassifier

    seen_row_counts = []

    class RecordingPostProcessor:
        def process(self, code, rows, labels, is_classification):
            seen_row_counts.append(len(rows))
            return code

    clf = SkribeClassifier(model="gpt-5.4-mini", postprocessor=RecordingPostProcessor())
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)
    monkeypatch.setattr(
        clf,
        "_call_llm",
        lambda prompt, web_search=False: "def predict(**features): return 0",
    )
    # Force the prompt sample to be truncated well below the full dataset by
    # capping the simulated context window to a handful of rows.
    monkeypatch.setattr(
        clf,
        "_truncate_to_context_window",
        lambda data, prompt_template, headroom=None, max_input_override=None: data.head(3),
    )
    n_rows = 20
    clf.fit(
        pd.DataFrame({"a": list(range(n_rows))}),
        pd.Series([i % 2 for i in range(n_rows)], name="target"),
    )
    assert seen_row_counts == [n_rows]  # full training data, not the 3-row prompt sample


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
    monkeypatch.setattr(litellm, "get_model_info", lambda m: {"max_input_tokens": 128_000})
    monkeypatch.setattr(litellm, "token_counter", lambda **kw: 100)

    df = pd.DataFrame({"x": range(50)})
    result = est._truncate_to_context_window(df, "prompt {data} end")
    assert len(result) == 50


def test_truncate_warns_and_reduces(monkeypatch):
    """When the prompt exceeds the budget, a UserWarning is raised and rows are dropped."""
    import litellm

    est = BaseSkribeEstimator(model="gpt-4o", verbose=False, max_train_rows=None)
    monkeypatch.setattr(litellm, "get_model_info", lambda m: {"max_input_tokens": 1000})
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


def test_fit_retries_context_window_error_with_api_reported_limit_even_when_repeated(
    monkeypatch,
):
    """The _fit retry loop must always trust a real, API-reported token limit
    over blind headroom-percentage guessing -- including on a *repeat* of the
    same limit, not just the first time it differs from our prior guess.
    Repeating the same override should still shrink headroom (our margin
    against that confirmed number wasn't enough) but must record it as a
    confirmation of the API's number, not an unrelated guess."""
    from skribe.base import _ContextWindowExceeded
    from skribe.classifier import SkribeClassifier

    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False)
    calls = []

    def fake_generate_code(base_prompt, validation_rows, validation_labels=[], **kwargs):
        calls.append(1)
        if len(calls) <= 3:
            # Same real limit reported every time -- must still be adopted
            # (and logged as "confirmed") rather than falling through to a
            # headroom-percentage guess just because it isn't new anymore.
            raise _ContextWindowExceeded("rejected", real_max_input_tokens=50_000)
        return "code", "code", lambda **features: 0

    monkeypatch.setattr(clf, "_generate_code", fake_generate_code)

    clf.fit(pd.DataFrame({"a": [1, 2, 3, 4]}), pd.Series([0, 0, 1, 1], name="target"))

    context_window_entries = [e for e in clf.fit_log_ if e.get("stage") == "context_window"]
    assert len(context_window_entries) == 3
    assert context_window_entries[0]["action"] == "corrected max_input_tokens to 50000"
    # Every subsequent hit of the *same* reported limit is a confirmation,
    # never a fallback to an unrelated headroom-shrink guess.
    for entry in context_window_entries[1:]:
        assert entry["action"].startswith("confirmed max_input_tokens at 50000")
    assert not any("shrunk headroom" in e["action"] for e in context_window_entries)


def test_default_known_context_windows_is_empty():
    """skribe has no built-in list of verified context windows and no config
    file of its own -- known_context_windows defaults to an empty dict, not
    any hardcoded entries. Callers that maintain their own verified limits
    (e.g. the benchmarks harness) inject them via the constructor."""
    est = BaseSkribeEstimator(model="gpt-4", verbose=False, max_train_rows=10)
    assert est.known_context_windows == {}


def test_truncate_uses_injected_known_context_window_before_litellm_registry(monkeypatch):
    """A model_id present in the caller-injected known_context_windows dict
    (a hand-verified limit for a model litellm's own registry doesn't know
    about) is used instead of ever calling litellm.get_model_info() -- so a
    brand-new model the caller has already confirmed the real limit for
    doesn't pay the discover-it-at-runtime cost (an oversized first prompt, a
    guaranteed rejection, a learned correction) on every subsequent run."""
    import litellm

    est = BaseSkribeEstimator(
        model="vertex_ai/gemini-3.6-flash",
        verbose=False,
        max_train_rows=None,
        known_context_windows={"vertex_ai/gemini-3.6-flash": 1_048_576},
    )
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda m: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    df = pd.DataFrame({"x": range(20)})
    result = est._truncate_to_context_window(df, "{data}")
    assert len(result) == 20


def test_known_context_windows_injected_via_classifier_constructor(monkeypatch):
    """known_context_windows passed to SkribeClassifier() reaches
    _truncate_to_context_window through the full DI path (not just directly
    on BaseSkribeEstimator), and is consulted before litellm.get_model_info()
    -- skribe/ has no config file of its own, so this is the only way a
    caller's verified limits reach the fit-time truncation logic."""
    import litellm

    from skribe.classifier import SkribeClassifier

    clf = SkribeClassifier(
        model="vertex_ai/gemini-3.6-flash",
        known_context_windows={"vertex_ai/gemini-3.6-flash": 1_048_576},
    )
    assert clf.known_context_windows == {"vertex_ai/gemini-3.6-flash": 1_048_576}
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda m: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    df = pd.DataFrame({"x": range(20)})
    result = clf._truncate_to_context_window(df, "{data}")
    assert len(result) == 20


def test_truncate_uses_fallback_budget_when_model_unknown(monkeypatch):
    """If get_model_info raises (e.g. a brand-new, not-yet-registered model),
    a conservative fallback budget is used instead of skipping the check --
    a small dataset that fits comfortably within the fallback is returned
    unchanged, with a warning logged rather than a crash."""
    import litellm

    est = BaseSkribeEstimator(model="unknown-model", verbose=False, max_train_rows=None)
    monkeypatch.setattr(
        litellm, "get_model_info", lambda m: (_ for _ in ()).throw(Exception("unknown"))
    )

    df = pd.DataFrame({"x": range(20)})
    result = est._truncate_to_context_window(df, "{data}")
    assert len(result) == 20


def test_truncate_actually_truncates_when_model_unknown_and_data_too_large(monkeypatch):
    """The fallback budget is a real ceiling, not a bypass -- data that would
    exceed it still gets truncated even though get_model_info() has no
    answer for this model."""
    import litellm

    from skribe.base import _UNKNOWN_MODEL_CONTEXT_FALLBACK

    est = BaseSkribeEstimator(model="unknown-model", verbose=False, max_train_rows=None)
    monkeypatch.setattr(
        litellm, "get_model_info", lambda m: (_ for _ in ()).throw(Exception("unknown"))
    )
    # Force _count_tokens to report a fixed, large per-call token count so
    # this test doesn't depend on a real tokenizer or on how large a
    # DataFrame would need to be to naturally exceed the fallback budget.
    monkeypatch.setattr(est, "_count_tokens", lambda prompt: len(prompt.splitlines()) * 1000)

    prompt_template = "{data}"
    df = pd.DataFrame({"x": range(500)})
    result = est._truncate_to_context_window(df, prompt_template)
    assert len(result) < 500
    # Confirm the kept row count is consistent with _UNKNOWN_MODEL_CONTEXT_FALLBACK
    # specifically (not some other budget): the largest row count whose fake
    # token count still fits within 92% of the fallback.
    budget = int(_UNKNOWN_MODEL_CONTEXT_FALLBACK * 0.92)

    def tokens_for(n_rows):
        csv = df.iloc[:n_rows].to_csv(index=False)
        return est._count_tokens(prompt_template.replace("{data}", csv))

    assert tokens_for(len(result)) <= budget
    assert tokens_for(len(result) + 1) > budget


@pytest.mark.parametrize(
    "text,expected",
    [
        # OpenAI Responses API.
        ("Input tokens exceed the configured limit of 272000 tokens.", 272000),
        # OpenAI Chat Completions.
        ("This model's maximum context length is 128000 tokens.", 128000),
        # Vertex AI (gemini-3.6-flash, unmapped in litellm as of this
        # writing) -- no unit word after the number at all.
        ("The input token count exceeds the maximum number of tokens allowed 1048576.", 1048576),
        # A thousands-separator variant.
        ("This model has a configured limit of 128,000 tokens.", 128000),
        # Anthropic Chat Completions: two numbers before ">", the real
        # ceiling is the one after it, not the input length before it.
        (
            "input length and `max_tokens` exceed context limit: 178902 + 32000 > 200000",
            200000,
        ),
        # Not a context-window error at all -- must not false-positive just
        # because a number and the word "limit" both appear.
        ("Rate limit exceeded, please try again in 20 seconds", None),
        ("Error code: 404, model not found", None),
        ("Connection timed out after None seconds.", None),
    ],
)
def test_extract_context_limit(text, expected):
    from skribe.base import _extract_context_limit

    assert _extract_context_limit(text) == expected


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


def test_check_unresolved_names_allows_module_level_lookup_dict():
    # A module-level constant (e.g. a categorical lookup table the extend
    # pass commonly generates) referenced from inside predict() is valid
    # Python and must not be flagged as unresolved.
    code = (
        "CHEST_PAIN_MAP = {'typical': 0, 'atypical': 1}\n\n"
        "def predict(**features):\n"
        "    return CHEST_PAIN_MAP.get(features.get('chest_pain'), 0)\n"
    )
    _check_unresolved_names(code)  # must not raise


def test_check_unresolved_names_allows_module_level_helper_and_import():
    code = (
        "import math\n\n"
        "def _normalize(x):\n"
        "    return x.lower()\n\n"
        "def predict(**features):\n"
        "    y = _normalize(features.get('a', ''))\n"
        "    return math.floor(1.0) if y == 'yes' else 0\n"
    )
    _check_unresolved_names(code)  # must not raise


def test_check_unresolved_names_still_catches_genuine_typo_with_suggestion():
    # A name that doesn't match anything bound in the function, module
    # scope, or builtins is still a real bug -- and now that module-level
    # names are visible, the suggestion should point at the correct one.
    code = (
        "CHEST_PAIN_MAP = {'typical': 0, 'atypical': 1}\n\n"
        "def predict(**features):\n"
        "    return CHEST_PAIN_MA.get(features.get('chest_pain'), 0)\n"
    )
    with pytest.raises(NameError, match="CHEST_PAIN_MA.*Did you mean: 'CHEST_PAIN_MAP'"):
        _check_unresolved_names(code)


def test_postprocess_constants_fits_threshold():
    """A bare numeric threshold gets replaced with the real best split point
    for the feature when doing so improves training accuracy, not left at
    the LLM's original guess."""
    pp = ConstantPostProcessor()
    code = (
        "def predict(**features):\n"
        "    age = float(features['age'])\n"
        "    return 1 if age > 999.0 else 0\n"
    )
    rows = [{"age": a} for a in [10, 20, 30, 40, 50, 60, 70, 80]]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    result = pp.process(code, rows, labels, is_classification=True)
    assert "45.0" in result  # true split is the 40/50 midpoint
    assert "999.0" not in result


def test_postprocess_constants_fits_joint_coefficients():
    """Multiple bare numeric coefficients combined linearly are fit jointly
    via LinearRegression, recovering the real relationship between two
    features and the target rather than keeping arbitrary guessed weights,
    since the joint fit strictly improves on the guessed constants here."""
    pp = ConstantPostProcessor()
    code = (
        "def predict(**features):\n"
        "    age = float(features['age'])\n"
        "    income = float(features['income'])\n"
        "    return 999.0 * age + 999.0 * income\n"
    )
    rows = []
    labels = []
    # age and income vary independently here (not collinear) so the joint
    # fit has a unique solution to recover.
    for age, income in [
        (20, 50000),
        (30, 15000),
        (40, 80000),
        (50, 25000),
        (60, 60000),
        (25, 90000),
        (45, 10000),
    ]:
        rows.append({"age": age, "income": income})
        labels.append(2.0 * age + 0.001 * income)

    result = pp.process(code, rows, labels, is_classification=False)
    # Extract the two fitted literals and check they recovered ~2.0 / ~0.001.
    import ast as _ast

    tree = _ast.parse(result)
    consts = [
        n.value
        for n in _ast.walk(tree)
        if isinstance(n, _ast.Constant) and isinstance(n.value, float)
    ]
    assert any(abs(c - 2.0) < 0.01 for c in consts)
    assert any(abs(c - 0.001) < 0.0001 for c in consts)


def test_postprocess_constants_uses_holdout_split_above_min_rows():
    """Above the minimum-rows threshold, process() must internally split
    rows/labels into a fit slice and a holdout slice rather than fitting and
    scoring accept/reject on the exact same rows -- otherwise every fit would
    look like an improvement even when it's just memorizing the sample."""
    from skribe.postprocess import _MIN_ROWS_FOR_SPLIT, _fit_holdout_split

    rows = [{"age": a} for a in range(_MIN_ROWS_FOR_SPLIT + 10)]
    labels = [0] * len(rows)
    fit_rows, fit_labels, holdout_rows, holdout_labels = _fit_holdout_split(rows, labels)

    assert holdout_rows, "expected a non-empty holdout slice above the minimum threshold"
    assert fit_rows, "expected a non-empty fit slice above the minimum threshold"
    assert len(fit_rows) + len(holdout_rows) == len(rows)
    # No row should appear in both slices.
    fit_ages = {r["age"] for r in fit_rows}
    holdout_ages = {r["age"] for r in holdout_rows}
    assert fit_ages.isdisjoint(holdout_ages)


def test_postprocess_constants_falls_back_to_full_reuse_below_min_rows():
    """Below the minimum-rows threshold there isn't enough data to split
    meaningfully, so process() should fall back to reusing every row for
    both fitting and scoring rather than being left with too little data on
    either side."""
    from skribe.postprocess import _fit_holdout_split

    rows = [{"age": a} for a in range(5)]
    labels = [0, 0, 1, 1, 1]
    fit_rows, fit_labels, holdout_rows, holdout_labels = _fit_holdout_split(rows, labels)

    assert fit_rows == rows
    assert fit_labels == labels
    assert holdout_rows == []
    assert holdout_labels == []


def test_postprocess_constants_keeps_original_when_no_rows():
    """When there's no usable training data (e.g. no rows), the code is
    returned unchanged rather than the fit failing or guessing blindly."""
    pp = ConstantPostProcessor()
    code = (
        "def predict(**features):\n"
        "    age = float(features['age'])\n"
        "    return 1 if age > 45.0 else 0\n"
    )
    result = pp.process(code, [], [], is_classification=True)
    assert result == code


def test_postprocess_constants_no_sites_returns_code_unchanged():
    """Code with no qualifying numeric literals at all (e.g.
    SkribeFeatureEngineer's transform() output) passes through untouched."""
    pp = ConstantPostProcessor()
    code = "def transform(**features):\n    return {}\n"
    assert pp.process(code, [{"a": 1}], [0], is_classification=True) == code


def test_postprocess_constants_rejects_fit_that_regresses_accuracy():
    """A literal that is already correct (scores at least as well as any
    data-fit alternative) must be left completely unchanged -- the accuracy
    safety rail is what makes this safe as a pure postprocessor with no
    declared LLM intent to lean on. Here the threshold is already a perfect
    separator, so a naive fit that moved it would only hurt."""
    pp = ConstantPostProcessor()
    code = (
        "def predict(**features):\n"
        "    age = float(features['age'])\n"
        "    return 1 if age > 45.0 else 0\n"
    )
    # 45.0 already perfectly separates the two classes; feeding in a single
    # extreme outlier row would pull a naive best-split scan away from it,
    # but the rail must reject that since it can't improve on a perfect fit.
    rows = [{"age": a} for a in [10, 20, 30, 40, 50, 60, 70, 5000]]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    result = pp.process(code, rows, labels, is_classification=True)
    assert "45.0" in result


def test_postprocess_constants_handles_negative_literal():
    """A bare negative threshold (age > -45.0) parses as UnaryOp(USub,
    Constant(45.0)), not a plain Constant -- must still be recognized and
    fit, not silently skipped. Regression for a real bug found via a live
    credit-g benchmark run where every negative-default calibrate() call
    (the former marker convention, since removed) was invisible to the site
    finder."""
    pp = ConstantPostProcessor()
    code = (
        "def predict(**features):\n"
        "    age = float(features['age'])\n"
        "    return 1 if age > -999.0 else 0\n"
    )
    rows = [{"age": a} for a in [10, 20, 30, 40, 50, 60, 70, 80]]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    result = pp.process(code, rows, labels, is_classification=True)
    assert "45.0" in result  # true split (40/50 midpoint), not the negative guess


def test_find_unverified_thresholds_flags_negative_bare_literal():
    """A bare negative threshold (age < -45.0) parses as UnaryOp(USub,
    Constant), not a plain Constant -- must still be flagged."""
    code = (
        "def predict(**features):\n"
        "    age = float(features['age'])\n"
        "    return 1 if age < -45.0 else 0\n"
    )
    findings = find_unverified_thresholds(code)
    assert len(findings) == 1
    assert findings[0]["literal"] == -45.0
    assert findings[0]["feature"] == "age"


def test_postprocess_constants_fits_dict_literal_weight_table():
    """Numeric dict-literal lookup-table values (a common LLM pattern for
    categorical risk scoring, e.g. credit_history -> risk weight) get fit
    jointly via one-hot indicator columns against the label, recovering
    each category's relative sign/ordering -- this is what fixes a
    reversed-polarity category, not just a wrong scalar. Regression for the
    real credit-g benchmark bug: the LLM guessed 'critical/other existing
    credit' as a mildly-good weight (-0.85, on a 'lower is better' scale)
    when the data says it's strongly bad."""
    pp = ConstantPostProcessor()
    code = (
        "def predict(**features):\n"
        "    credit_history = features['credit_history']\n"
        "    risk = 0.0\n"
        "    history_risk = {\n"
        "        'critical/other existing credit': -0.85,\n"
        "        'all paid': 0.45,\n"
        "        'existing paid': 4.0,\n"
        "    }\n"
        "    risk += history_risk.get(credit_history, 0.0)\n"
        "    return 1 if risk > 0 else 0\n"
    )
    rows = (
        [{"credit_history": "critical/other existing credit"}] * 20
        + [{"credit_history": "all paid"}] * 20
        + [{"credit_history": "existing paid"}] * 20
    )
    labels = [0] * 20 + [1] * 20 + [1] * 20

    result = pp.process(code, rows, labels, is_classification=True)

    import ast as _ast

    tree = _ast.parse(result)
    dict_node = next(n for n in _ast.walk(tree) if isinstance(n, _ast.Dict))
    fitted = {
        k.value: (v.value if isinstance(v, _ast.Constant) else -v.operand.value)
        for k, v in zip(dict_node.keys, dict_node.values)
    }
    # The fit must correct the polarity: critical/other existing credit
    # should end up negative (bad), not the LLM's mildly-positive-leaning guess.
    assert fitted["critical/other existing credit"] < 0
    assert fitted["all paid"] > 0


def test_postprocess_constants_dict_keeps_defaults_when_no_feature_matches():
    """If no feature's raw values match the dict's keys well enough, the
    fit is degenerate and the accuracy rail rejects it, leaving the LLM's
    own literals in place rather than fitting noise."""
    pp = ConstantPostProcessor()
    code = (
        "def predict(**features):\n"
        "    x = features['x']\n"
        "    risk_map = {\n"
        "        'alpha': 0.5,\n"
        "        'beta': -0.5,\n"
        "    }\n"
        "    return 1 if risk_map.get(x, 0.0) > 0 else 0\n"
    )
    rows = [{"x": "unrelated_value_1"}, {"x": "unrelated_value_2"}]
    labels = [0, 1]
    result = pp.process(code, rows, labels, is_classification=True)
    assert "0.5" in result and "-0.5" in result


def test_postprocess_constants_splices_multiline_literal():
    """A numeric literal that happens to sit on its own line inside a
    multi-line expression (a real, reproduced pattern from a live benchmark
    run on a large dataset, where the LLM line-wraps a long generated
    function) must still be found and spliced correctly using absolute
    source offsets, not corrupted by naive per-line splicing."""
    pp = ConstantPostProcessor()
    code = (
        "def predict(**features):\n"
        "    age = float(features['age'])\n"
        "    return 1 if age > (\n"
        "        999.0\n"
        "    ) else 0\n"
    )
    rows = [{"age": a} for a in [10, 20, 30, 40, 50, 60, 70, 80]]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    result = pp.process(code, rows, labels, is_classification=True)
    assert "45.0" in result
    import ast as _ast

    _ast.parse(result)  # must still be valid Python after the multi-line splice


def test_postprocess_constants_splices_multiline_and_singleline_on_shared_lines():
    """Multiple numeric literals, some multi-line and some single-line,
    combined in one expression -- the absolute-offset splice must handle
    both without corrupting each other's spans."""
    pp = ConstantPostProcessor()
    code = (
        "def predict(**features):\n"
        "    age = float(features['age'])\n"
        "    income = float(features['income'])\n"
        "    return (\n"
        "        (\n"
        "            999.0\n"
        "        ) * age\n"
        "        + 999.0 * income\n"
        "    )\n"
    )
    rows = []
    labels = []
    for age, income in [
        (20, 50000),
        (30, 15000),
        (40, 80000),
        (50, 25000),
        (60, 60000),
        (25, 90000),
        (45, 10000),
    ]:
        rows.append({"age": age, "income": income})
        labels.append(2.0 * age + 0.001 * income)

    result = pp.process(code, rows, labels, is_classification=False)
    import ast as _ast

    tree = _ast.parse(result)
    consts = [
        n.value
        for n in _ast.walk(tree)
        if isinstance(n, _ast.Constant) and isinstance(n.value, float)
    ]
    assert any(abs(c - 2.0) < 0.01 for c in consts)
    assert any(abs(c - 0.001) < 0.0001 for c in consts)


def test_postprocess_constants_skips_small_integers():
    """Small integers (e.g. a count of 2) in threshold/coefficient position
    are excluded from calibration entirely -- they're almost always
    structural (a count, an index, a boolean-like flag), not a
    dataset-dependent guess worth refitting."""
    pp = ConstantPostProcessor()
    code = (
        "def predict(**features):\n" "    n = int(features['n'])\n" "    return 1 if n > 2 else 0\n"
    )
    rows = [{"n": v} for v in [0, 1, 2, 3, 4, 5]]
    labels = [0, 0, 0, 1, 1, 1]
    result = pp.process(code, rows, labels, is_classification=True)
    assert result == code  # untouched: "2" never became a candidate site


def test_find_unverified_thresholds_flags_bare_comparison():
    code = (
        "def predict(**features):\n"
        "    age = float(features['age'])\n"
        "    return 1 if age > 45.0 else 0\n"
    )
    findings = find_unverified_thresholds(code)
    assert len(findings) == 1
    assert findings[0]["feature"] == "age"
    assert findings[0]["literal"] == 45.0
    assert findings[0]["kind"] == "threshold"


def test_find_unverified_thresholds_flags_bare_coefficient():
    code = (
        "def predict(**features):\n"
        "    age = float(features['age'])\n"
        "    income = float(features['income'])\n"
        "    return 0.3 * age + 0.7 * income\n"
    )
    findings = find_unverified_thresholds(code)
    kinds_and_features = {(f["kind"], f["feature"]) for f in findings}
    assert ("coefficient", "age") in kinds_and_features
    assert ("coefficient", "income") in kinds_and_features


def test_find_unverified_thresholds_ignores_small_integers():
    """A small integer in threshold position (e.g. a count check) is
    structural, not a calibration candidate -- the gate must not flag it."""
    code = (
        "def predict(**features):\n" "    n = int(features['n'])\n" "    return 1 if n > 2 else 0\n"
    )
    assert find_unverified_thresholds(code) == []


def test_find_unverified_thresholds_no_findings_on_clean_code():
    code = "def predict(**features):\n    return 0\n"
    assert find_unverified_thresholds(code) == []


def test_fit_forces_search_retry_when_threshold_unverified(monkeypatch):
    """Generated code with a bare numeric threshold and no web search logged
    this attempt must trigger a corrective retry that forces web_search=True
    on the next attempt, per Task 2's deterministic post-generation gate.

    Attempt 0 already has web_search=True from the pre-existing "first or
    last attempt" rule -- the point being tested here is not that flag, but
    that the LLM having *access* to search doesn't mean it searched: since
    no fit_log_ web_search entry was recorded, the unverified threshold in
    attempt 0's code must still be rejected and retried, with the
    unverified_threshold stage logged and web_search forced on again for
    attempt 1 regardless of where it falls in the retry sequence.
    """
    from skribe.classifier import SkribeClassifier

    clf = SkribeClassifier(model="gpt-5.4-mini", web_search=True, max_retries=2)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)

    calls = []

    def fake_llm(prompt, web_search=False):
        calls.append(web_search)
        if len(calls) == 1:
            # Attempt 0: web_search=True was offered, but the model didn't
            # actually search (no fit_log_ entry) and hardcoded a threshold.
            return "def predict(**features): return 1 if float(features['a']) > 10.0 else 0"
        # Attempt 1: pretend a search happened this time -- the gate checks
        # whether a search was logged this attempt, not whether the literal
        # itself changed, so the code can keep the same bare threshold.
        clf.fit_log_.append(
            {"stage": "web_search", "search_call_count": 1, "citations": ["http://example.com"]}
        )
        return "def predict(**features): return 1 if float(features['a']) > 10.0 else 0"

    monkeypatch.setattr(clf, "_call_llm", fake_llm)
    clf.fit(pd.DataFrame({"a": [1, 2, 3, 4]}), pd.Series([0, 0, 1, 1], name="target"))

    assert calls == [True, True]  # both requested search access; attempt 1 forced by the gate
    unverified_entries = [e for e in clf.fit_log_ if e.get("stage") == "unverified_threshold"]
    assert len(unverified_entries) == 1
    assert unverified_entries[0]["feature"] == "a"
    assert clf.predict_fn is not None


def test_fit_forces_search_on_middle_attempt_that_would_otherwise_skip_it(monkeypatch):
    """Distinguishes the gate's forcing behavior from the pre-existing
    "first or last attempt" web-search rule: a middle attempt (neither
    first nor last) normally has web_search=False, but must be forced to
    True if the previous attempt's code was rejected by the gate."""
    from skribe.classifier import SkribeClassifier

    clf = SkribeClassifier(model="gpt-5.4-mini", web_search=True, max_retries=3)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)

    calls = []

    def fake_llm(prompt, web_search=False):
        calls.append(web_search)
        n = len(calls)
        if n == 1:
            # Attempt 0: an ordinary validation failure, unrelated to the gate.
            return "def predict(**features): raise ValueError('boom')"
        if n == 2:
            # Attempt 1: middle attempt, would naturally be web_search=False.
            # Bare unverified threshold triggers the gate.
            return "def predict(**features): return 1 if float(features['a']) > 10.0 else 0"
        # Attempt 2: must have been forced to web_search=True by the gate.
        clf.fit_log_.append({"stage": "web_search", "search_call_count": 1, "citations": ["x"]})
        return "def predict(**features): return 1 if float(features['a']) > 10.0 else 0"

    monkeypatch.setattr(clf, "_call_llm", fake_llm)
    clf.fit(pd.DataFrame({"a": [1, 2, 3, 4]}), pd.Series([0, 0, 1, 1], name="target"))

    assert calls == [True, False, True]
    assert clf.predict_fn is not None


def test_fit_allows_unverified_threshold_on_last_attempt(monkeypatch):
    """The gate must not fire on the final attempt -- forcing yet another
    retry there would just exhaust max_retries and fail the whole fit,
    which is worse than shipping a working-but-unverified threshold."""
    from skribe.classifier import SkribeClassifier

    clf = SkribeClassifier(model="gpt-5.4-mini", web_search=True, max_retries=0)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)
    monkeypatch.setattr(
        clf,
        "_call_llm",
        lambda prompt, web_search=False: "def predict(**features): return 1 if float(features['a']) > 10.0 else 0",
    )
    clf.fit(pd.DataFrame({"a": [1, 2, 3, 4]}), pd.Series([0, 0, 1, 1], name="target"))
    assert clf.predict_fn is not None
    assert not any(entry.get("stage") == "unverified_threshold" for entry in clf.fit_log_)


def test_fit_succeeds_when_forced_search_retry_llm_call_fails(monkeypatch):
    """An unverified threshold is a policy warning, not a code-correctness
    bug -- the code that trips it already compiles and validates fine. The
    gate forces a corrective retry with web search on; if that retry's LLM
    call itself fails outright (e.g. a network timeout), the fit must still
    succeed using the prior, unverified-but-working code, rather than
    failing the whole fit over a transient network error."""
    from skribe.classifier import SkribeClassifier

    clf = SkribeClassifier(model="gpt-5.4-mini", web_search=True, max_retries=2)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)

    calls = []

    def fake_llm(prompt, web_search=False):
        calls.append(web_search)
        if len(calls) == 1:
            # Attempt 0: bare unverified threshold, no search logged -> gate
            # rejects it and forces web_search=True on the next attempt.
            return "def predict(**features): return 1 if float(features['a']) > 10.0 else 0"
        # Attempt 1 (the forced-search retry): the call itself fails.
        raise RuntimeError("Connection timed out")

    monkeypatch.setattr(clf, "_call_llm", fake_llm)
    clf.fit(pd.DataFrame({"a": [1, 2, 3, 4]}), pd.Series([0, 0, 1, 1], name="target"))

    assert calls == [True, True]
    assert clf.predict_fn is not None
    unverified_entries = [e for e in clf.fit_log_ if e.get("stage") == "unverified_threshold"]
    assert len(unverified_entries) == 2  # the original gate rejection + the recovery entry
    assert unverified_entries[-1]["action"].startswith("LLM call failed")
    assert clf.predict(pd.DataFrame({"a": [20]}))[0] == 1
