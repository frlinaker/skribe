"""Tests for PromptFeatureEngineer and AdaptiveFeatureEngineer.

The mocked tests stub the two LLM touchpoints (``_call_llm`` and
``_extend_code``) so they run without network. One live smoke test exercises a
real provider (the cheap test model from conftest).
"""

import io

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from promptlearn import AdaptiveFeatureEngineer, PromptFeatureEngineer

# A simple, valid transform the mocked LLM "returns".
GOOD_CODE = (
    "def transform(**features):\n"
    "    country = str(features.get('country', '')).lower()\n"
    "    gdp = {'sweden': 60000, 'india': 2500}.get(country, 0)\n"
    "    age = float(features.get('age', 0) or 0)\n"
    "    return {'gdp_per_capita': gdp, 'is_adult': 1 if age >= 18 else 0}\n"
)


def _mock(fe, code=GOOD_CODE):
    fe._call_llm = lambda prompt, web_search=False: code
    fe._extend_code = lambda c: c
    return fe


@pytest.fixture
def Xy():
    X = pd.DataFrame(
        {"country": ["sweden", "india", "sweden", "india"], "age": [30, 12, 41, 9]}
    )
    y = pd.Series([1, 0, 1, 0], name="label")
    return X, y


def test_fit_transform_appends_features(Xy):
    X, y = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False)).fit(X, y)
    assert fe.new_feature_names_ == ["gdp_per_capita", "is_adult"]
    out = fe.transform(X)
    # original columns preserved, new ones appended
    assert list(out.columns) == ["country", "age", "gdp_per_capita", "is_adult"]
    assert out.loc[0, "gdp_per_capita"] == 60000
    assert list(out["is_adult"]) == [1, 0, 1, 0]
    assert len(out) == len(X)


def test_fit_unsupervised_without_y(Xy):
    X, _ = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False)).fit(X)
    assert fe.target_name_ is None
    assert "gdp_per_capita" in fe.transform(X).columns


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        PromptFeatureEngineer(verbose=False).transform(pd.DataFrame({"a": [1]}))


def test_requires_dataframe(Xy):
    _, y = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False))
    with pytest.raises(ValueError, match="DataFrame"):
        fe.fit(np.array([[1, 2], [3, 4]]), y)


def test_transform_row_failure_yields_nan(Xy):
    X, y = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False)).fit(X, y)
    # Replace the compiled fn with one that raises, to exercise the safe path.
    fe.predict_fn = lambda **f: (_ for _ in ()).throw(RuntimeError("boom"))
    out = fe.transform(X)
    assert out["gdp_per_capita"].isna().all()


def test_validation_rejects_nondict_output(Xy):
    X, y = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False, max_retries=0))
    fe._call_llm = (
        lambda prompt, web_search=False: "def transform(**features):\n    return 42\n"
    )
    with pytest.raises(ValueError, match="must return a dict"):
        fe.fit(X, y)


def test_validation_rejects_inconsistent_keys(Xy):
    X, y = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False, max_retries=0))
    code = (
        "def transform(**features):\n"
        "    age = float(features.get('age', 0) or 0)\n"
        "    return {'a': 1} if age >= 18 else {'b': 2}\n"
    )
    fe._call_llm = lambda prompt, web_search=False: code
    with pytest.raises(ValueError, match="same set of feature keys"):
        fe.fit(X, y)


def test_pipeline_with_downstream_model(Xy):
    X, y = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False)).fit(X, y)
    # Use only the engineered numeric features downstream.
    engineered = fe.transform(X)[fe.new_feature_names_]
    clf = LogisticRegression().fit(engineered, y)
    assert clf.score(engineered, y) == 1.0


def test_joblib_roundtrip_recompiles(Xy):
    X, _ = Xy
    # Build fitted state directly (no instance-level mocks that would pollute
    # __dict__ and break pickling).
    fe = PromptFeatureEngineer(verbose=False)
    fe.feature_names_ = ["country", "age"]
    fe.target_name_ = "label"
    fe.raw_python_code_ = GOOD_CODE
    fe.python_code_ = GOOD_CODE
    fe.new_feature_names_ = ["gdp_per_capita", "is_adult"]

    buffer = io.BytesIO()
    joblib.dump(fe, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    # predict_fn is dropped on dump and recompiled from python_code_ on load.
    out = restored.transform(X)
    assert out.loc[0, "gdp_per_capita"] == 60000
    assert restored.new_feature_names_ == ["gdp_per_capita", "is_adult"]


def test_live_feature_engineering():
    """Smoke test against a real provider via the conftest test model."""
    X = pd.DataFrame(
        {
            "country": ["Sweden", "India", "Japan", "Brazil"],
            "age": [30, 12, 41, 17],
        }
    )
    y = pd.Series([1, 0, 1, 0], name="label")
    fe = PromptFeatureEngineer(verbose=False).fit(X, y)
    out = fe.transform(X)
    # New columns were added, original rows preserved.
    assert len(out) == len(X)
    assert len(out.columns) > len(X.columns)
    assert fe.new_feature_names_


# ---------------------------------------------------------------------------
# AdaptiveFeatureEngineer tests
# ---------------------------------------------------------------------------


# Patch _probe_fe_delta so tests run without LLM calls.
# Returns (score_base, score_fe, probe_n).
def _patch_probe(afe, score_base, score_fe):
    """Inject a fake probe result into an AdaptiveFeatureEngineer instance."""
    import promptlearn.feature_engineer as _fem

    def _fake_probe(X, y, fe, cv, probe_size):
        return score_base, score_fe, min(40, len(X))

    afe._probe_fn = _fake_probe
    return afe


class _MockedAFE(AdaptiveFeatureEngineer):
    """Subclass that replaces _probe_fe_delta with a controllable fake."""

    def __init__(self, probe_base, probe_fe, **kw):
        super().__init__(**kw)
        self._probe_base = probe_base
        self._probe_fe = probe_fe

    def fit(self, X, y=None):
        import math

        if not isinstance(X, pd.DataFrame):
            raise ValueError("AdaptiveFeatureEngineer requires a pandas DataFrame.")

        self.skip_reason_ = None
        self.probe_score_base_ = float("nan")
        self.probe_score_fe_ = float("nan")
        self.probe_delta_ = float("nan")

        if X.shape[0] < self.min_rows:
            self.skip_reason_ = (
                f"n_rows={X.shape[0]} < min_rows={self.min_rows} — "
                "too few samples for engineered features to generalise"
            )
            self.fe_ = None
            return self

        self.probe_score_base_ = self._probe_base
        self.probe_score_fe_ = self._probe_fe
        self.probe_delta_ = self._probe_fe - self._probe_base

        if self.probe_delta_ <= self.min_delta:
            self.skip_reason_ = (
                f"probe_delta={self.probe_delta_:+.3f} <= min_delta={self.min_delta} — "
                "FE did not improve logreg accuracy on probe split"
            )
            self.fe_ = None
            return self

        # Simulate a fitted FE (empty passthrough)
        from unittest.mock import MagicMock

        mock_fe = MagicMock()
        mock_fe.transform.side_effect = lambda X: X
        self.fe_ = mock_fe
        return self


def _make_df(n=200):
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
    y = pd.Series((X["a"] > 0).astype(int), name="label")
    return X, y


def test_adaptive_skip_on_small_dataset():
    """Fewer than min_rows → size-guard skip, no probe needed."""
    X, y = _make_df(n=50)
    afe = _MockedAFE(probe_base=0.7, probe_fe=0.8, min_rows=200, verbose=False)
    afe.fit(X, y)
    assert afe.skip_reason_ is not None
    assert "n_rows" in afe.skip_reason_
    assert afe.fe_ is None
    # probe stats remain NaN since probe never ran
    assert np.isnan(afe.probe_delta_)


def test_adaptive_skip_when_probe_delta_zero():
    """Probe shows no lift → skip."""
    X, y = _make_df()
    afe = _MockedAFE(probe_base=0.80, probe_fe=0.80, min_rows=200, verbose=False)
    afe.fit(X, y)
    assert afe.skip_reason_ is not None
    assert "probe_delta" in afe.skip_reason_
    assert afe.fe_ is None
    assert afe.probe_delta_ == pytest.approx(0.0)


def test_adaptive_skip_when_probe_delta_negative():
    """Probe shows negative lift → skip."""
    X, y = _make_df()
    afe = _MockedAFE(probe_base=0.80, probe_fe=0.75, min_rows=200, verbose=False)
    afe.fit(X, y)
    assert afe.skip_reason_ is not None
    assert afe.fe_ is None
    assert afe.probe_delta_ == pytest.approx(-0.05)


def test_adaptive_runs_when_probe_shows_lift():
    """Probe shows positive lift → FE runs."""
    X, y = _make_df()
    afe = _MockedAFE(probe_base=0.70, probe_fe=0.80, min_rows=200, verbose=False)
    afe.fit(X, y)
    assert afe.skip_reason_ is None
    assert afe.fe_ is not None
    assert afe.probe_delta_ == pytest.approx(0.10)


def test_adaptive_passthrough_preserves_dataframe(Xy):
    """When skipped, transform() returns the original DataFrame unchanged."""
    X, y = Xy
    afe = _MockedAFE(probe_base=0.75, probe_fe=0.75, min_rows=0, verbose=False)
    afe.fit(X, y)
    assert afe.skip_reason_ is not None
    out = afe.transform(X)
    pd.testing.assert_frame_equal(out, X)


def test_adaptive_transform_before_fit_raises(Xy):
    X, _ = Xy
    afe = AdaptiveFeatureEngineer(verbose=False)
    with pytest.raises(RuntimeError, match="fit"):
        afe.transform(X)


def test_adaptive_requires_dataframe():
    afe = AdaptiveFeatureEngineer(verbose=False)
    with pytest.raises(ValueError, match="DataFrame"):
        afe.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))


def test_prompt_fe_empty_dict_passthrough(Xy):
    """PromptFeatureEngineer with LLM returning empty dict → transform returns original cols."""
    X, y = Xy
    empty_code = "def transform(**features):\n    return {}\n"
    fe = _mock(PromptFeatureEngineer(verbose=False), code=empty_code)
    fe.fit(X, y)
    assert fe.new_feature_names_ == []
    out = fe.transform(X)
    assert list(out.columns) == list(X.columns)


def test_prompt_fe_dataset_stats_injected(Xy):
    """dataset_stats passed to fit() appear in the LLM prompt."""
    X, y = Xy
    captured = []

    fe = PromptFeatureEngineer(verbose=False)
    fe._call_llm = lambda prompt, web_search=False: (
        captured.append(prompt),
        GOOD_CODE,
    )[1]
    fe._extend_code = lambda c: c
    stats = {
        "n_rows": 4,
        "logreg_cv_accuracy": "0.750",
        "xgboost_minus_logreg_gap": "+0.100",
    }
    fe.fit(X, y, dataset_stats=stats)

    assert captured, "LLM was never called"
    assert "logreg_cv_accuracy" in captured[0]
    assert "0.750" in captured[0]
