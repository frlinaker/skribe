"""Security regression tests: a serialized (joblib/pickle) model must never
carry API keys or other credentials.

The estimator talks to providers through litellm/openai, which read keys from
the environment at call time. None of that should ever be captured in the
estimator's state. These tests fail loudly if a future change starts storing a
client object (or anything secret) on the estimator.
"""

import io

import joblib
import pandas as pd

from promptlearn.classifier import PromptClassifier
from promptlearn.regressor import PromptRegressor
from promptlearn.explain import Explanation

# A sentinel that looks like a real key; placed in the environment so that, if
# anything captured the key, it would show up verbatim in the dump.
SECRET = "sk-FAKE-secret-key-do-not-leak-0123456789abcdef"

# Only these (non-secret) fields are expected in serialized state. A new field
# outside this set must be reviewed for credential leakage before being added.
ALLOWED_STATE_FIELDS = {
    "model",
    "verbose",
    "max_train_rows",
    "max_retries",
    "web_search",
    "context_prepass",
    "vertex_location",
    "target_name_",
    "feature_names_",
    "raw_python_code_",
    "python_code_",
    "explanation_",
    "context_summary_",
    "context_prepass_prompt_",
    "new_feature_names_",  # PromptFeatureEngineer
}


def _fitted(cls=PromptClassifier):
    """A fitted-looking estimator (incl. a cached explanation) with no network."""
    est = cls()
    est.feature_names_ = ["age"]
    est.target_name_ = "label"
    code = "def predict(age):\n    return 1 if float(age) >= 18 else 0"
    est.raw_python_code_ = code
    est.python_code_ = code
    # Set the cached explanation directly (no LLM call, no instance-level
    # monkeypatching that would pollute the serialized state).
    est.explanation_ = Explanation(
        meta={
            "name": cls.__name__,
            "type": ["whitebox"],
            "explanations": ["global"],
            "params": est.get_params(),
        },
        data={
            "summary": "Predicts 1 when age >= 18.",
            "features_used": ["age"],
            "code": code,
        },
    )
    return est


def test_getstate_only_contains_safe_fields():
    est = _fitted()
    state = est.__getstate__()
    assert "llm_client" not in state  # no raw LLM client should ever be serialized
    assert set(state).issubset(
        ALLOWED_STATE_FIELDS
    ), f"Unexpected serialized field(s): {set(state) - ALLOWED_STATE_FIELDS}"


def test_joblib_dump_contains_no_credentials(monkeypatch):
    # Configure provider keys in the environment, as a real user would.
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)

    est = _fitted()

    buffer = io.BytesIO()
    joblib.dump(est, buffer)
    blob = buffer.getvalue()

    assert SECRET.encode() not in blob
    for marker in (
        b"OPENAI_API_KEY",
        b"ANTHROPIC_API_KEY",
        b"api_key",
        b"authorization",
        b"Bearer ",
    ):
        assert marker not in blob, f"credential marker {marker!r} found in dump"

    # The reloaded model must still work (predict recompiles from python_code_).
    buffer.seek(0)
    restored = joblib.load(buffer)
    assert int(restored.predict(pd.DataFrame([{"age": 20}]))[0]) == 1


def test_regressor_dump_contains_no_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    est = _fitted(cls=PromptRegressor)

    buffer = io.BytesIO()
    joblib.dump(est, buffer)
    assert SECRET.encode() not in buffer.getvalue()
