import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys
import tempfile

from promptlearn.classifier import PromptClassifier
from promptlearn.utils import *


def is_int(val):
    return isinstance(val, (int, np.integer))


@pytest.fixture
def sample_Xy():
    X = pd.DataFrame({"x1": [0, 1, 2], "x2": [2, 3, 4]})
    y = pd.Series([0, 1, 0], name="y")
    return X, y


def test_stuff():
    assert is_int(3)
    assert is_int(np.int64(2))


def test_fit_predict_dataframe(sample_Xy):
    X, y = sample_Xy
    clf = PromptClassifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)
    assert all(is_int(p) for p in preds)


def test_fit_predict_ndarray(sample_Xy):
    X, y = sample_Xy
    clf = PromptClassifier()
    clf.fit(X.values, y.values)
    preds = clf.predict(X.values)
    assert len(preds) == len(y)


def test_score_accuracy(sample_Xy):
    X, y = sample_Xy
    clf = PromptClassifier()
    clf.fit(X, y)
    acc = clf.score(X, y)
    assert 0.0 <= acc <= 1.0


def test_zero_row_fit_predict():
    X = pd.DataFrame(columns=["x"])
    y = pd.Series(name="y", dtype=int)
    clf = PromptClassifier()
    clf.fit(X, y)
    preds = clf.predict(pd.DataFrame([{"x": 1}]))
    assert is_int(preds[0])


def test_predict_missing_column(sample_Xy):
    X, y = sample_Xy
    clf = PromptClassifier()
    clf.fit(X, y)
    # Remove one column
    X2 = X.copy().drop("x2", axis=1)
    preds = clf.predict(X2)
    assert len(preds) == len(X2)


def test_predict_extra_column(sample_Xy):
    X, y = sample_Xy
    clf = PromptClassifier()
    clf.fit(X, y)
    X2 = X.copy()
    X2["extra"] = 99
    preds = clf.predict(X2)
    assert len(preds) == len(X2)


def test_predict_without_fit_raises():
    clf = PromptClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(pd.DataFrame({"x": [1, 2]}))


def test_predict_invalid_type_after_fit():
    clf = PromptClassifier()
    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)
    with pytest.raises(ValueError):
        clf.predict("not a dataframe or array")


def test_joblib_save_load(sample_Xy):
    X, y = sample_Xy
    clf = PromptClassifier()
    clf.fit(X, y)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "clf.joblib")
        joblib.dump(clf, path)
        clf2 = joblib.load(path)
        preds = clf2.predict(X)
        assert len(preds) == len(y)


def test_missing_api_key(monkeypatch):
    # Remove env var and force reload
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import importlib
    import promptlearn.base

    importlib.reload(promptlearn.base)
    with pytest.raises(RuntimeError):
        PromptClassifier()


def test_importerror_openai(monkeypatch):
    """Test missing openai dependency error path."""
    # Patch sys.modules to simulate openai missing
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError):
        clf = PromptClassifier(model="gpt-4o")


def test_setstate_broken_code(monkeypatch):
    """Test __setstate__ with broken python_code_ triggers warning and fallback."""
    clf = PromptClassifier()
    bad_state = dict(
        python_code_="def not_valid_code !@#", predict_fn=None, model="gpt-4o"
    )
    # Should warn but not crash
    with pytest.warns(UserWarning):
        clf.__setstate__(bad_state)
    assert clf.predict_fn is None


def test_fit_too_many_rows(monkeypatch):
    """Test .fit() samples down if input too large."""
    import promptlearn.utils
    import pandas as pd
    import numpy as np
    from promptlearn.classifier import PromptClassifier

    # Patch make_predict_fn at the module level
    monkeypatch.setattr(
        promptlearn.utils, "make_predict_fn", lambda code: lambda **features: 0
    )
    clf = PromptClassifier(max_train_rows=5)
    # 10 rows will trigger down-sampling
    X = pd.DataFrame({"a": np.arange(10)})
    y = pd.Series(np.arange(10), name="target")
    # Patch _call_llm to return a stub function
    monkeypatch.setattr(
        clf, "_call_llm", lambda prompt: "def predict(**features): return 0"
    )
    # Now .fit should use the patched function
    clf.fit(X, y)
    assert clf.predict_fn is not None


def test_fit_blank_llm_output(monkeypatch):
    """Test .fit() with empty/whitespace LLM output triggers error."""
    clf = PromptClassifier()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 2, 3], name="target")
    monkeypatch.setattr(clf, "_call_llm", lambda prompt: "   \n   ")
    with pytest.raises(ValueError, match="No code to exec from LLM output"):
        clf.fit(X, y)


def test_fit_nonstring_llm_output(monkeypatch):
    """Test .fit() with non-string LLM output is handled robustly."""
    import pandas as pd
    import promptlearn.utils

    # Patch BEFORE importing PromptClassifier
    monkeypatch.setattr(
        promptlearn.utils, "make_predict_fn", lambda code: lambda **features: 0
    )
    from promptlearn.classifier import PromptClassifier

    clf = PromptClassifier()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 2, 3], name="target")
    monkeypatch.setattr(clf, "_call_llm", lambda prompt: 12345)
    # Expect ValueError because the LLM output isn't code
    with pytest.raises(ValueError, match="No valid function named 'predict'"):
        clf.fit(X, y)


def test_sample_calls_llm_and_parses(monkeypatch):
    """Test .sample() exercises LLM and TSV parsing logic."""
    clf = PromptClassifier()
    # Patch _call_llm to return a simple TSV
    clf.feature_names_ = ["x1"]
    clf.target_name_ = "y"
    monkeypatch.setattr(clf, "_call_llm", lambda prompt: "a\tb\n1\t2\n3\t4")
    df = clf.sample(2)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"a", "b"}
    assert len(df) == 2


def test_sample_raises_before_fit():
    clf = PromptClassifier()
    with pytest.raises(RuntimeError, match="Call fit.*before sample"):
        clf.sample(2)


def test_normalize_feature_name_various():
    assert normalize_feature_name("A B C") == "a_b_c"
    assert normalize_feature_name("foo-bar.baz") == "foo_bar_baz"
    assert normalize_feature_name("__x__y__") == "x_y"
    assert normalize_feature_name("foo123") == "foo123"
    assert normalize_feature_name("foo__bar") == "foo_bar"


def test_safe_exec_fn_handles_errors_and_coercion():
    # Broken function, should fallback to default
    def broken(**features):
        raise RuntimeError("fail!")

    out = safe_exec_fn(broken, {"a": 1}, output_type=int, default=42, label="T")
    assert out == 42

    # Correct function, but returns None
    def returns_none(**features):
        return None

    assert safe_exec_fn(returns_none, {"a": 1}, output_type=int, default=13) == 13

    # Test coercion of string numbers
    def just_return(**features):
        return features["x"]

    assert (
        safe_exec_fn(just_return, {"x": "3.0"}, output_type=float, default=0.0) == 3.0
    )
    assert safe_exec_fn(just_return, {"x": "5"}, output_type=int, default=0) == 5


def test_generate_feature_dicts_dataframe_and_ndarray():
    df = pd.DataFrame({"foo bar": [1], "baz": [2]})
    results = list(generate_feature_dicts(df, df.columns))
    assert results == [{"foo_bar": 1, "baz": 2}]
    arr = np.array([[3, 4]])
    names = ["x", "y"]
    results = list(generate_feature_dicts(arr, names))
    assert results == [{"x": 3, "y": 4}]


def test_prepare_training_data_various_inputs():
    df = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([3, 4], name="Y Tar")
    data, feature_names, target_name = prepare_training_data(df, y)
    assert target_name == "y_tar"
    assert "y_tar" in data.columns
    arr = np.array([[1, 2], [3, 4]])
    yarr = np.array([5, 6])
    data2, feature_names2, target_name2 = prepare_training_data(arr, yarr)
    assert target_name2 == "target"
    assert data2.shape[1] == arr.shape[1] + 1


def test_parse_tsv_parses_and_errors():
    tsv = "a\tb\n1\t2\n3\t4"
    df = parse_tsv(tsv)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_make_predict_fn_error_handling():
    # Not valid python
    try:
        make_predict_fn("def no_colon\n  pass")
    except ValueError as e:
        assert "Could not exec LLM code" in str(e)
    # No 'predict' function
    try:
        make_predict_fn("def something(): pass")
    except ValueError as e:
        assert "No valid function named 'predict'" in str(e)


def test_sample_generates_examples(monkeypatch):
    clf = PromptClassifier()
    clf.feature_names_ = ["foo", "bar"]
    clf.target_name_ = "baz"
    clf.python_code_ = "def predict(foo, bar): return 1"
    # Patch LLM call to return TSV
    monkeypatch.setattr(
        clf, "_call_llm", lambda prompt: "foo\tbar\tbaz\n1\t2\t3\n4\t5\t6"
    )
    df = clf.sample(2)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2


def test_generate_feature_dicts_invalid():
    # X is neither DataFrame nor ndarray
    with pytest.raises(ValueError, match="X must be a DataFrame or ndarray."):
        list(generate_feature_dicts("bad input", []))


def test_prepare_training_data_invalid():
    with pytest.raises(
        ValueError, match="X must be a pandas DataFrame or numpy array."
    ):
        prepare_training_data("not a table", [1, 2, 3])


def test_safe_exec_fn_none_and_str_fallback():
    # None value path (returns None because that's what the function returns)
    def fn(**features):
        return features.get("val", 42)

    assert safe_exec_fn(fn, {"val": None}, default=None) is None

    # Simulate error in function, fallback to 0
    def fn_error(**features):
        raise TypeError("bad input")

    assert safe_exec_fn(fn_error, {"val": None}) == 0

    # If feature value is a non-numeric string, fallback is 0
    assert safe_exec_fn(fn, {"val": "not_a_number"}) == 0


def test_make_predict_fn_exec_error():
    with pytest.raises(ValueError, match="Could not exec LLM code"):
        make_predict_fn("def bad_code :")  # Syntax error


def test_make_predict_fn_missing_predict():
    with pytest.raises(ValueError, match="No valid function named 'predict'"):
        make_predict_fn("def not_predict(): pass")


def test_parse_tsv_bad_input():
    # An empty string should trigger the error
    with pytest.raises(ValueError, match="Failed to parse TSV output"):
        parse_tsv("")


def test_safe_exec_fn_non_number_string():
    def fn(**features):
        return "not_a_number"

    # Should fall back to default (0) because int("not_a_number") fails
    assert safe_exec_fn(fn, {"val": "not_a_number"}) == 0
