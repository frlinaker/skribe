"""skribe estimators must satisfy the scikit-learn estimator protocol so
that meta-estimators (GridSearchCV, MultiOutputRegressor, ...) accept them.

These checks don't fit a model, so they make no LLM calls.
"""

import pytest
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
    is_classifier,
    is_regressor,
)

from skribe import SkribeClassifier, SkribeFeatureEngineer, SkribeRegressor

ALL_ESTIMATORS = [SkribeClassifier, SkribeRegressor, SkribeFeatureEngineer]


def test_estimators_inherit_baseestimator_and_mixins():
    assert isinstance(SkribeClassifier(verbose=False), BaseEstimator)
    assert isinstance(SkribeRegressor(verbose=False), BaseEstimator)
    assert isinstance(SkribeClassifier(verbose=False), ClassifierMixin)
    assert isinstance(SkribeRegressor(verbose=False), RegressorMixin)
    assert isinstance(SkribeFeatureEngineer(verbose=False), TransformerMixin)


def test_estimator_type_tags():
    assert is_classifier(SkribeClassifier(verbose=False))
    assert is_regressor(SkribeRegressor(verbose=False))


@pytest.mark.parametrize("cls", ALL_ESTIMATORS)
def test_sklearn_tags_available(cls):
    # sklearn>=1.6 meta-estimators call __sklearn_tags__; it must not raise.
    tags = cls(verbose=False).__sklearn_tags__()
    assert tags is not None


@pytest.mark.parametrize("cls", ALL_ESTIMATORS)
def test_clone_roundtrips_params(cls):
    est = cls(model="gpt-4o", verbose=False, max_train_rows=42, max_retries=1)
    cloned = clone(est)
    assert cloned.get_params() == est.get_params()
    assert cloned is not est
