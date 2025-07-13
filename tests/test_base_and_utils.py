import pytest
from promptlearn.base import BasePromptEstimator


def test_get_set_params():
    est = BasePromptEstimator(model="gpt-4", verbose=True, max_train_rows=10)
    params = est.get_params()
    assert params["model"] == "gpt-4"
    est.set_params(model="gpt-3.5-turbo")
    assert est.model == "gpt-3.5-turbo"


def test_call_llm_raises(monkeypatch):
    est = BasePromptEstimator(model="gpt-4", verbose=False, max_train_rows=1)
    monkeypatch.setattr(est, "llm_client", None)
    with pytest.raises(Exception):
        est._call_llm("this should fail")


def test_extend_code_handles_llm_failure(monkeypatch):
    class DummyEstimator(BasePromptEstimator):
        def __init__(self):
            super().__init__(model="dummy-model", verbose=False, max_train_rows=10)

        def _call_llm(self, prompt: str):
            raise RuntimeError("Mocked LLM failure")

    estimator = DummyEstimator()
    # Should log a warning and return original code unchanged
    result = estimator._extend_code("def predict(**features): return 42")
    assert result.strip() == "def predict(**features): return 42"
