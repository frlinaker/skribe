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
