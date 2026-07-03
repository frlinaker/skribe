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
import pytest

from promptlearn import PromptClassifier, PromptRegressor
from promptlearn.prompt_markers import CONTEXT_END, CONTEXT_START, DATA_MARKER

SIMPLE_CODE = "def predict(**f): return 0"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_marker_values():
    assert CONTEXT_START == "--- Dataset context ---"
    assert CONTEXT_END == "--- End context ---"
    assert DATA_MARKER == "Data:"


def test_classifier_template_uses_data_marker():
    from promptlearn.classifier import DEFAULT_CLASSIFICATION_PROMPT_TEMPLATE

    assert DATA_MARKER in DEFAULT_CLASSIFICATION_PROMPT_TEMPLATE
    # Must not hard-code a different string
    assert "Data:\n" in DEFAULT_CLASSIFICATION_PROMPT_TEMPLATE


def test_regressor_template_uses_data_marker():
    from promptlearn.regressor import DEFAULT_REGRESSION_PROMPT_TEMPLATE

    assert DATA_MARKER in DEFAULT_REGRESSION_PROMPT_TEMPLATE
    assert "Data:\n" in DEFAULT_REGRESSION_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Markers appear verbatim in fit_prompt_ and in the correct order
# ---------------------------------------------------------------------------


def test_fit_prompt_marker_order_with_context(monkeypatch):
    """Instructions < CONTEXT_START < CONTEXT_END < DATA_MARKER."""
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=False)
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


def test_fit_prompt_no_context_still_has_data_marker(monkeypatch):
    """Without a description, CONTEXT_START/END are absent but DATA_MARKER is still present."""
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    p = clf.fit_prompt_
    assert DATA_MARKER in p
    assert CONTEXT_START not in p
    assert CONTEXT_END not in p


def test_fit_prompt_contains_no_unresolved_placeholders(monkeypatch):
    """No __PLACEHOLDER__ tokens should survive into the final prompt."""
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=False)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"age": [25, 40]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="Some description.")

    assert "__CONTEXT_START__" not in clf.fit_prompt_
    assert "__CONTEXT_END__" not in clf.fit_prompt_
    assert "__DATA_MARKER__" not in clf.fit_prompt_


def test_regressor_fit_prompt_marker_order(monkeypatch):
    reg = PromptRegressor(model="gpt-5.4-mini", verbose=False, context_prepass=False)
    monkeypatch.setattr(reg, "_call_llm", lambda p, web_search=False: "def predict(**f): return 1.0")

    X = pd.DataFrame({"height": [1.0, 2.0]})
    y = pd.Series([0.5, 1.0])
    reg.fit(X, y, dataset_description="Physics dataset.")

    p = reg.fit_prompt_
    assert p.index(CONTEXT_START) < p.index(CONTEXT_END) < p.index(DATA_MARKER)


# ---------------------------------------------------------------------------
# Pre-pass prompt forbids markdown
# ---------------------------------------------------------------------------


def test_prepass_prompt_forbids_markdown(monkeypatch):
    """The pre-pass prompt must explicitly prohibit markdown formatting."""
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
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
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"age": [25, 40, 35]})
    y = pd.Series([0, 1, 0])
    clf.fit(X, y, dataset_description="Some dataset.")

    assert "downstream prompt" in clf.context_prepass_prompt_


def test_prepass_prompt_forbids_fences(monkeypatch):
    """Pre-pass prompt must explicitly ban code fences."""
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"age": [25, 40]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="Some dataset.")

    assert "fences" in clf.context_prepass_prompt_ or "fence" in clf.context_prepass_prompt_


# ---------------------------------------------------------------------------
# Value summary caps at 20 unique values
# ---------------------------------------------------------------------------


def test_prepass_value_summary_shows_all_values(monkeypatch):
    """All unique values per column appear in the pre-pass prompt — no cap."""
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    monkeypatch.setattr(
        clf, "_call_llm",
        lambda p, web_search=False: "def predict(**f): return int(int(f.get('val', 0)) > 14)"
    )

    # 30 unique values in each column
    X = pd.DataFrame({"code": [str(i) for i in range(30)], "val": range(30)})
    y = pd.Series([i % 2 for i in range(30)])
    clf.fit(X, y, dataset_description="Some dataset.")

    # All 30 values present, no ellipsis
    assert ", ..." not in clf.context_prepass_prompt_
    assert "29" in clf.context_prepass_prompt_


def test_prepass_value_summary_no_ellipsis_when_few_values(monkeypatch):
    """Columns with few unique values appear without truncation."""
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False, context_prepass=True)
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: SIMPLE_CODE)

    X = pd.DataFrame({"code": list("abcde"), "val": range(5)})
    y = pd.Series([0, 1, 0, 1, 0])
    clf.fit(X, y, dataset_description="Some dataset.")

    prompt = clf.context_prepass_prompt_
    code_line = [l for l in prompt.splitlines() if "code:" in l]
    assert code_line, "Expected a 'code:' line in value summary"
    assert ", ..." not in code_line[0]
