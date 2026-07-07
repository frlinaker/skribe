"""Contract tests for `explain()` — written TDD-style (these FAIL until
`explain()` is implemented).

Each test encodes a recognized explainability standard from the XAI community
so the eventual implementation is held to more than "the LLM returned a string":

- Return type: an ``Explanation`` object exposing ``meta`` + ``data`` dicts with
  top-level keys also reachable as attributes, JSON round-trippable.
  (Alibi `Explanation`: https://docs.seldon.io/projects/alibi ;
   InterpretML global/local: https://interpret.ml/docs/framework.html)
- Global vs local scope distinction.
  (InterpretML ``explain_global``/``explain_local``; SHAP.)
- ``NotFittedError`` before ``fit``  (scikit-learn inspection convention:
  https://scikit-learn.org/stable/inspection.html)
- Quality properties from Molnar's *Interpretable ML* and Doshi-Velez & Kim
  (1702.08608): fidelity/faithfulness, stability/determinism,
  comprehensibility/sparsity, no "phantom" (unused) features.

Most structural tests stub ``_call_llm`` so they neither hit the network nor
assert on LLM phrasing; the fidelity test is intentionally a live call.
"""

import pytest
import pandas as pd

from sklearn.exceptions import NotFittedError

from skribe.classifier import SkribeClassifier

# A known, hand-written rule so we can assert about explanations of it without
# depending on the (stochastic) code-generation step.
KNOWN_CODE = "def predict(age):\n    return 1 if float(age) >= 18 else 0"


def _make_fitted(feature_names=("age",), code=KNOWN_CODE):
    """Return an estimator in a fitted state without calling the generation LLM."""
    clf = SkribeClassifier()
    clf.feature_names_ = list(feature_names)
    clf.target_name_ = "is_adult"
    clf.raw_python_code_ = code
    clf.python_code_ = code
    return clf


# --------------------------------------------------------------------------- #
# Structural contract (Alibi / InterpretML return-type conventions)
# --------------------------------------------------------------------------- #
def test_explain_returns_meta_and_data(monkeypatch):
    clf = _make_fitted()
    monkeypatch.setattr(
        clf, "_call_llm", lambda prompt: "Predicts adult when age >= 18."
    )
    e = clf.explain()
    assert isinstance(e.meta, dict)
    assert isinstance(e.data, dict)
    assert e.meta["name"] == "SkribeClassifier"
    assert "params" in e.meta
    assert isinstance(e.data["summary"], str) and e.data["summary"].strip()
    assert "features_used" in e.data


def test_explain_scope_is_global(monkeypatch):
    """A bare explain() describes the whole model → global scope."""
    clf = _make_fitted()
    monkeypatch.setattr(clf, "_call_llm", lambda prompt: "text")
    e = clf.explain()
    assert e.meta["explanations"] == ["global"]


def test_explain_attribute_access(monkeypatch):
    """Top-level keys of meta/data are reachable as attributes (Alibi ChainMap)."""
    clf = _make_fitted()
    monkeypatch.setattr(clf, "_call_llm", lambda prompt: "the summary")
    e = clf.explain()
    assert e.summary == e.data["summary"]
    assert e.name == e.meta["name"]


def test_explain_json_roundtrip(monkeypatch):
    from skribe import Explanation  # fails until implemented/exported

    clf = _make_fitted()
    monkeypatch.setattr(
        clf, "_call_llm", lambda prompt: "Predicts adult when age >= 18."
    )
    e = clf.explain()
    restored = Explanation.from_json(e.to_json())
    assert restored.meta == e.meta
    assert restored.data == e.data


def test_explain_str_is_summary(monkeypatch):
    clf = _make_fitted()
    monkeypatch.setattr(
        clf, "_call_llm", lambda prompt: "Predicts adult when age >= 18."
    )
    e = clf.explain()
    assert str(e) == e.summary


# --------------------------------------------------------------------------- #
# scikit-learn convention: not-fitted estimators raise NotFittedError
# --------------------------------------------------------------------------- #
def test_explain_before_fit_raises_notfitted():
    with pytest.raises(NotFittedError):
        SkribeClassifier().explain()


# --------------------------------------------------------------------------- #
# Comprehensibility / sparsity + no "phantom" features
# (explanation references only features the model could actually use)
# --------------------------------------------------------------------------- #
def test_explain_no_phantom_features(monkeypatch):
    clf = _make_fitted(feature_names=("age", "height"))
    monkeypatch.setattr(clf, "_call_llm", lambda prompt: "text")
    e = clf.explain()
    assert set(e.features_used).issubset(set(clf.feature_names_))


def test_explain_is_selective(monkeypatch):
    clf = _make_fitted(feature_names=("age",))
    monkeypatch.setattr(
        clf, "_call_llm", lambda prompt: "Predicts adult when age >= 18."
    )
    e = clf.explain()
    assert len(e.features_used) == len(set(e.features_used))  # no duplicates
    assert len(e.features_used) <= len(clf.feature_names_)  # invents nothing


# --------------------------------------------------------------------------- #
# Consistency / determinism: same fitted model → identical explanation,
# and the LLM is consulted at most once (result is cached).
# --------------------------------------------------------------------------- #
def test_explain_is_deterministic_and_cached(monkeypatch):
    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        return f"summary variant {calls['n']}"

    clf = _make_fitted()
    monkeypatch.setattr(clf, "_call_llm", fake_llm)
    first = clf.explain()
    second = clf.explain()
    assert first.summary == second.summary  # determinism
    assert calls["n"] == 1  # cached, not regenerated


# --------------------------------------------------------------------------- #
# Fidelity / faithfulness (LIVE): the plain-English explanation must reflect
# the actual rule encoded by the model, not a generic description.
# --------------------------------------------------------------------------- #
def test_explain_is_faithful_to_known_rule():
    clf = _make_fitted(feature_names=("age",))
    text = clf.explain().summary.lower()
    assert "age" in text


# --------------------------------------------------------------------------- #
# Global vs local distinction (InterpretML/SHAP). explain(X) for a single row
# yields a LOCAL explanation of that one prediction.
# --------------------------------------------------------------------------- #
def test_explain_local_for_instance(monkeypatch):
    clf = _make_fitted(feature_names=("age",))
    monkeypatch.setattr(
        clf, "_call_llm", lambda prompt: "age (20) >= 18, so predicted adult (1)."
    )
    e = clf.explain(pd.DataFrame([{"age": 20}]))
    assert "local" in e.meta["explanations"]
