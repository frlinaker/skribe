"""Regression tests for prompt section markers and their usage.

Covers:
- prompt_markers.py constants have the exact expected values
- classifier.py and regressor.py templates use the constants (not hard-coded copies)
- fit_prompt_ contains the exact marker strings in the correct order
- Pre-pass prompt explicitly forbids markdown in its output instruction
- Pre-pass prompt explains output is embedded in a downstream prompt
- Value summary caps unique values at 20 and appends ", ..." suffix
"""

import pandas as pd

from skribe import SkribeClassifier, SkribeRegressor
from skribe.prompt_markers import CONTEXT_END, CONTEXT_START, DATA_MARKER

SIMPLE_CODE = "def predict(**f): return 0"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_marker_values():
    assert CONTEXT_START == "--- Dataset context ---"
    assert CONTEXT_END == "--- End context ---"
    assert DATA_MARKER == "Data:"


def test_classifier_template_uses_data_marker():
    from skribe.classifier import DEFAULT_CLASSIFICATION_PROMPT_TEMPLATE

    assert DATA_MARKER in DEFAULT_CLASSIFICATION_PROMPT_TEMPLATE
    # Must not hard-code a different string
    assert "Data:\n" in DEFAULT_CLASSIFICATION_PROMPT_TEMPLATE


def test_regressor_template_uses_data_marker():
    from skribe.regressor import DEFAULT_REGRESSION_PROMPT_TEMPLATE

    assert DATA_MARKER in DEFAULT_REGRESSION_PROMPT_TEMPLATE
    assert "Data:\n" in DEFAULT_REGRESSION_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Markers appear verbatim in fit_prompt_ and in the correct order
# ---------------------------------------------------------------------------


def test_fit_prompt_marker_order_with_context(monkeypatch):
    """Instructions < CONTEXT_START < CONTEXT_END < DATA_MARKER."""
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=False)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"age": [25, 40], "income": [30000, 80000]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="UCI Adult.")

    p = clf.fit_prompt_
    assert CONTEXT_START in p
    assert CONTEXT_END in p
    assert DATA_MARKER in p

    assert p.index("Output a single valid Python") < p.index(CONTEXT_START)
    assert p.index(CONTEXT_START) < p.index(CONTEXT_END)
    assert p.index(CONTEXT_END) < p.index(DATA_MARKER)


def test_fit_prompt_no_description_still_has_data_marker(monkeypatch):
    """Without a dataset_description, DATA_MARKER is still present.

    CONTEXT_START/END are NOT necessarily absent here: SkribeClassifier.fit()
    always computes majority_class_ from the training data itself (not from
    dataset_description) and _build_prompt_without_data() always states it,
    since the alternative — the prompt template's old generic "fallback such
    as 0" wording — steered the LLM toward defaulting to whichever class
    sorts first regardless of true frequency (see majority_class_ docstring
    in classifier.py). So a context section can appear even with zero
    dataset_description; this test only asserts DATA_MARKER survives.
    """
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    p = clf.fit_prompt_
    assert DATA_MARKER in p


def test_fit_prompt_majority_class_stated_even_without_description(monkeypatch):
    """The majority-class fallback line is derived from training data, not
    dataset_description, so it must appear even when no description is given.
    """
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    # 3 rows of class 1, 1 row of class 0 -> majority class is 1.
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([1, 1, 1, 0])
    clf.fit(X, y)

    assert clf.majority_class_ == 1
    p = clf.fit_prompt_
    assert CONTEXT_START in p
    assert CONTEXT_END in p
    assert "training code 1" in p
    assert "not necessarily code 0" in p


def test_fit_prompt_contains_no_unresolved_placeholders(monkeypatch):
    """No __PLACEHOLDER__ tokens should survive into the final prompt."""
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=False)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"age": [25, 40]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="Some description.")

    assert "__CONTEXT_START__" not in clf.fit_prompt_
    assert "__CONTEXT_END__" not in clf.fit_prompt_
    assert "__DATA_MARKER__" not in clf.fit_prompt_


def test_regressor_fit_prompt_marker_order(monkeypatch):
    reg = SkribeRegressor(model="gpt-5.4-mini", verbose=False, context_prepass=False)
    monkeypatch.setattr(
        reg, "_call_llm", lambda p, web_search=False: "def predict(**f): return 1.0"
    )

    X = pd.DataFrame({"height": [1.0, 2.0]})
    y = pd.Series([0.5, 1.0])
    reg.fit(X, y, dataset_description="Physics dataset.")

    p = reg.fit_prompt_
    assert p.index(CONTEXT_START) < p.index(CONTEXT_END) < p.index(DATA_MARKER)


def test_regressor_fallback_states_median_even_without_description(monkeypatch):
    """SkribeRegressor.fit() always computes median_target_ from the
    training data and states it as the fallback/default value — regardless
    of dataset_description — since the old generic "such as 0.0" wording
    in the template is often a nonsensical default for a real target scale
    (e.g. a price or age of 0.0).
    """
    reg = SkribeRegressor(model="gpt-5.4-mini", verbose=False, context_prepass=False)
    monkeypatch.setattr(
        reg, "_call_llm", lambda p, web_search=False: "def predict(**f): return 1.0"
    )

    X = pd.DataFrame({"height": [1.0, 2.0, 3.0]})
    y = pd.Series([10.0, 20.0, 1000.0])  # median=20.0, mean~343.3
    reg.fit(X, y)

    assert reg.median_target_ == 20.0
    p = reg.fit_prompt_
    assert CONTEXT_START in p
    assert "20.0" in p
    assert "not necessarily 0.0" in p


def test_classifier_majority_class_uses_true_frequency_not_sort_order(monkeypatch):
    """majority_class_ must reflect actual training-data frequency, not
    whichever class happens to sort first (which is what code 0 means).
    """
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=False)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    # "bird" sorts before "mammal" alphabetically (so would be code 0), but
    # mammal is the actual majority class in this training data.
    X = pd.DataFrame({"legs": [4, 4, 4, 2]})
    y = pd.Series(["mammal", "mammal", "mammal", "bird"])
    clf.fit(X, y)

    assert list(clf.classes_) == ["bird", "mammal"]
    mammal_code = clf._code_of_["mammal"]
    assert mammal_code == 1  # sorts second -> code 1, but is the majority
    assert clf.majority_class_ == mammal_code
    assert f"training code {mammal_code}" in clf.fit_prompt_
    assert '"mammal"' in clf.fit_prompt_


# ---------------------------------------------------------------------------
# Pre-pass prompt forbids markdown
# ---------------------------------------------------------------------------


def test_prepass_prompt_forbids_markdown(monkeypatch):
    """The pre-pass prompt must explicitly prohibit markdown formatting."""
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"age": [25, 40, 35], "income": [30000, 80000, 50000]})
    y = pd.Series([0, 1, 0])
    clf.fit(X, y, dataset_description="UCI Adult.")

    prepass = clf.context_prepass_prompt_
    assert prepass is not None
    # Must forbid the specific markdown elements we care about
    assert "**" in prepass or "no **" in prepass or "no markdown" in prepass.lower()
    assert "#" in prepass or "no #" in prepass or "no markdown" in prepass.lower()


def test_prepass_prompt_says_downstream_prompt(monkeypatch):
    """Pre-pass prompt must tell the LLM its output will be used in a downstream prompt."""
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"age": [25, 40, 35]})
    y = pd.Series([0, 1, 0])
    clf.fit(X, y, dataset_description="Some dataset.")

    assert "downstream prompt" in clf.context_prepass_prompt_


def test_prepass_prompt_forbids_fences(monkeypatch):
    """Pre-pass prompt must explicitly ban code fences."""
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"age": [25, 40]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="Some dataset.")

    assert "fences" in clf.context_prepass_prompt_ or "fence" in clf.context_prepass_prompt_


# ---------------------------------------------------------------------------
# Value summary caps at 20 unique values
# ---------------------------------------------------------------------------


def test_prepass_value_summary_caps_high_cardinality_columns(monkeypatch):
    """Columns above the cap show a bounded preview, not every unique value.

    Previously there was no cap at all, which meant a single high-cardinality
    or near-continuous column (e.g. free-text names, raw timestamps) could
    blow the pre-pass prompt past the model's context window regardless of
    row count — this happened for real on the spotify-genre dataset (a
    track_name column with 23,449 uniques produced a ~392k-token prompt
    against gpt-4o's 128k-token limit, failing every time).
    """
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    monkeypatch.setattr(
        clf,
        "_call_llm",
        lambda p, web_search=False: "def predict(**f): return int(int(f.get('val', 0)) > 14)",
    )

    # 30 unique values in each column — over the 20-value cap.
    X = pd.DataFrame({"code": [str(i) for i in range(30)], "val": range(30)})
    y = pd.Series([i % 2 for i in range(30)])
    clf.fit(X, y, dataset_description="Some dataset.")

    prompt = clf.context_prepass_prompt_
    assert "more unique values" in prompt
    assert "29" not in prompt


def test_prepass_value_summary_no_ellipsis_when_few_values(monkeypatch):
    """Columns with few unique values appear without truncation."""
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"code": list("abcde"), "val": range(5)})
    y = pd.Series([0, 1, 0, 1, 0])
    clf.fit(X, y, dataset_description="Some dataset.")

    prompt = clf.context_prepass_prompt_
    code_line = [l for l in prompt.splitlines() if "code:" in l]
    assert code_line, "Expected a 'code:' line in value summary"
    assert ", ..." not in code_line[0]
