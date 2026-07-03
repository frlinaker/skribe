"""Tests for the promptlearn CLI (no LLM calls — mocks the estimators)."""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from promptlearn.cli import main


@pytest.fixture()
def iris_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "iris.csv"
    csv.write_text(textwrap.dedent("""\
        sepal_length,sepal_width,petal_length,petal_width,species
        5.1,3.5,1.4,0.2,0
        4.9,3.0,1.4,0.2,0
        6.3,3.3,6.0,2.5,2
        6.5,3.0,5.8,2.2,2
        5.5,2.4,3.7,1.0,1
        5.7,2.8,4.5,1.3,1
        """))
    return csv


@pytest.fixture()
def regression_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "reg.csv"
    csv.write_text(textwrap.dedent("""\
        x1,x2,y
        1.0,2.0,3.0
        2.0,3.0,5.0
        3.0,4.0,7.0
        4.0,5.0,9.0
        5.0,6.0,11.0
        """))
    return csv


def _mock_classifier():
    clf = MagicMock()
    clf.fit.return_value = clf
    clf.predict.side_effect = lambda X: np.zeros(len(X), dtype=int)
    return clf


def _mock_regressor():
    reg = MagicMock()
    reg.fit.return_value = reg
    reg.predict.side_effect = lambda X: np.ones(len(X), dtype=float)
    return reg


def test_no_args_prints_help(capsys):
    rc = main([])
    assert rc == 0


def test_fit_missing_target(iris_csv):
    rc = main(["fit", str(iris_csv), "--target", "nonexistent"])
    assert rc == 1


def test_fit_classification(iris_csv):
    with patch("promptlearn.cli.PromptClassifier") as MockClf:
        MockClf.return_value = _mock_classifier()
        rc = main(["fit", str(iris_csv), "--target", "species", "--test-size", "0.2"])
    assert rc == 0


def test_fit_regression(regression_csv):
    with patch("promptlearn.cli.PromptRegressor") as MockReg:
        MockReg.return_value = _mock_regressor()
        rc = main(
            [
                "fit",
                str(regression_csv),
                "--target",
                "y",
                "--task",
                "regression",
                "--test-size",
                "0.2",
            ]
        )
    assert rc == 0


def test_fit_no_test_split(iris_csv):
    with patch("promptlearn.cli.PromptClassifier") as MockClf:
        MockClf.return_value = _mock_classifier()
        rc = main(["fit", str(iris_csv), "--target", "species", "--test-size", "0"])
    assert rc == 0
