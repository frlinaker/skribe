import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest

from skribe.classifier import SkribeClassifier
from skribe.utils import *


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
    clf = SkribeClassifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)
    assert all(is_int(p) for p in preds)


def test_fit_predict_ndarray(sample_Xy):
    X, y = sample_Xy
    clf = SkribeClassifier()
    clf.fit(X.values, y.values)
    preds = clf.predict(X.values)
    assert len(preds) == len(y)


def test_score_accuracy(sample_Xy):
    X, y = sample_Xy
    clf = SkribeClassifier()
    clf.fit(X, y)
    acc = clf.score(X, y)
    assert 0.0 <= acc <= 1.0


def test_zero_row_fit_predict():
    X = pd.DataFrame(columns=["x"])
    y = pd.Series(name="y", dtype=int)
    clf = SkribeClassifier()
    clf.fit(X, y)
    preds = clf.predict(pd.DataFrame([{"x": 1}]))
    assert is_int(preds[0])


def test_predict_missing_column(sample_Xy):
    X, y = sample_Xy
    clf = SkribeClassifier()
    clf.fit(X, y)
    # Remove one column
    X2 = X.copy().drop("x2", axis=1)
    preds = clf.predict(X2)
    assert len(preds) == len(X2)


def test_predict_extra_column(sample_Xy):
    X, y = sample_Xy
    clf = SkribeClassifier()
    clf.fit(X, y)
    X2 = X.copy()
    X2["extra"] = 99
    preds = clf.predict(X2)
    assert len(preds) == len(X2)


def test_predict_without_fit_raises():
    clf = SkribeClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(pd.DataFrame({"x": [1, 2]}))


def test_predict_invalid_type_after_fit():
    clf = SkribeClassifier()
    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)
    with pytest.raises(ValueError):
        clf.predict("not a dataframe or array")


def test_joblib_save_load(sample_Xy):
    X, y = sample_Xy
    clf = SkribeClassifier()
    clf.fit(X, y)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "clf.joblib")
        joblib.dump(clf, path)
        clf2 = joblib.load(path)
        preds = clf2.predict(X)
        assert len(preds) == len(y)


def test_construction_does_not_require_api_key(monkeypatch):
    """Credentials are resolved lazily per-provider by litellm at call time,
    so constructing an estimator must not require any API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    clf = SkribeClassifier()
    assert clf.predict_fn is None


def test_fit_retries_then_succeeds(monkeypatch):
    """A function that errors when run on the sample is retried, then accepted."""
    clf = SkribeClassifier(max_retries=2)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)
    outputs = iter(
        [
            "def predict(**features): raise ValueError('boom')",  # compiles, fails at run
            "def predict(**features): return 0",  # valid
        ]
    )
    monkeypatch.setattr(clf, "_call_llm", lambda prompt, web_search=False: next(outputs))
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0], name="target")
    clf.fit(X, y)
    assert clf.predict_fn is not None
    assert clf.predict_fn(a=5) == 0


def test_fit_feedback_includes_error(monkeypatch):
    """The retry prompt carries the previous error message back to the LLM."""
    clf = SkribeClassifier(max_retries=1)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)
    prompts = []
    outputs = iter(
        [
            "def predict(**features): raise ValueError('kaboom')",
            "def predict(**features): return 1",
        ]
    )

    def fake_llm(prompt, web_search=False):
        prompts.append(prompt)
        return next(outputs)

    monkeypatch.setattr(clf, "_call_llm", fake_llm)
    clf.fit(pd.DataFrame({"a": [1]}), pd.Series([1], name="target"))
    assert len(prompts) == 2
    assert "kaboom" in prompts[1]


def test_fit_records_retry_history_on_success(monkeypatch):
    """Every validation failure/retry along the way to a successful fit must
    be recorded on clf.fit_log_, not just logged and discarded -- so a
    benchmark harness (or any caller) can persist what actually happened
    during fit() (which errors were hit, how many retries, what feedback
    the LLM got) instead of only ever seeing the final accuracy number.
    Regression for the cache audit's fit-time errors being visible only in
    ephemeral log files, invisible in the cached result JSON."""
    clf = SkribeClassifier(max_retries=1)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)
    outputs = iter(
        [
            "def predict(**features): raise ValueError('kaboom')",
            "def predict(**features): return 1",
        ]
    )
    monkeypatch.setattr(clf, "_call_llm", lambda prompt, web_search=False: next(outputs))
    clf.fit(pd.DataFrame({"a": [1]}), pd.Series([1], name="target"))

    assert len(clf.fit_log_) == 1
    entry = clf.fit_log_[0]
    assert entry["attempt"] == 1
    assert "kaboom" in entry["error"]


def test_fit_records_retry_history_on_exhaustion(monkeypatch):
    """Same as above, but when every attempt fails -- the retry history
    (one entry per failed attempt) must still be attached to the raised
    exception's estimator state so a caller that catches the exception can
    still recover what happened, not just the final error string."""
    clf = SkribeClassifier(max_retries=1)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)
    monkeypatch.setattr(
        clf,
        "_call_llm",
        lambda prompt, web_search=False: "def predict(**features): raise ValueError('nope')",
    )
    with pytest.raises(Exception):
        clf.fit(pd.DataFrame({"a": [1]}), pd.Series([1], name="target"))

    assert len(clf.fit_log_) == 2
    assert all("nope" in entry["error"] for entry in clf.fit_log_)
    assert [entry["attempt"] for entry in clf.fit_log_] == [1, 2]


def test_fit_feedback_includes_name_suggestion_for_typos(monkeypatch):
    """A NameError from a typo'd variable (e.g. 'ea' instead of the real loop
    variable 'ra') should feed back Python's own "Did you mean: 'ra'?"
    suggestion to the LLM, not just the bare "name 'ea' is not defined" --
    regression for the cache audit's finding that generated code sometimes
    has exactly this class of typo (spotify-genre x gemini-3.5-flash's
    'ea'/'ra' mixup, hepatitis x gpt-4o-mini's 'antiviral'/'antivirals'
    mixup), which a plain NameError message doesn't help the LLM self-correct
    as directly as the interpreter's own suggestion would."""
    clf = SkribeClassifier(max_retries=1)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)
    prompts = []
    outputs = iter(
        [
            "def predict(**features):\n"
            "    ra = features.get('a')\n"
            "    if ea == ra:\n"  # typo: should be 'ra == ra' or similar
            "        return 1\n"
            "    return 0\n",
            "def predict(**features): return 1",
        ]
    )

    def fake_llm(prompt, web_search=False):
        prompts.append(prompt)
        return next(outputs)

    monkeypatch.setattr(clf, "_call_llm", fake_llm)
    clf.fit(pd.DataFrame({"a": [1]}), pd.Series([1], name="target"))
    assert len(prompts) == 2
    assert "ea" in prompts[1] and "is not defined" in prompts[1]
    assert "Did you mean" in prompts[1] and "ra" in prompts[1]


def test_fit_catches_typo_in_untriggered_branch(monkeypatch):
    """A typo'd name inside a branch that no validation row happens to take
    must still be caught at fit time -- regression for row-execution-only
    validation being "hit-and-miss": _validate_predict_fn only calls
    predict_fn on the rows it has, so a NameError inside an untaken if/elif
    branch silently passes validation today and only surfaces later as a
    safe_predict fallback in production. A static AST check for references
    to names that are never bound anywhere in the function catches this
    regardless of which branch the validation rows exercise."""
    clf = SkribeClassifier(max_retries=1)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)
    prompts = []
    outputs = iter(
        [
            "def predict(**features):\n"
            "    ra = features.get('a')\n"
            "    if ra > 100:\n"  # validation row has a=1, never enters this branch
            "        return 1 if ea > 0 else 0\n"  # typo: 'ea' should be 'ra'
            "    return 0\n",
            "def predict(**features): return 1",
        ]
    )

    def fake_llm(prompt, web_search=False):
        prompts.append(prompt)
        return next(outputs)

    monkeypatch.setattr(clf, "_call_llm", fake_llm)
    clf.fit(pd.DataFrame({"a": [1]}), pd.Series([1], name="target"))
    assert len(prompts) == 2
    assert "ea" in prompts[1]
    assert "Did you mean" in prompts[1] and "ra" in prompts[1]


def test_fit_feedback_includes_argument_types_for_attribute_errors(monkeypatch):
    """An AttributeError from calling a string method on a non-string feature
    (e.g. a numeric-looking column pandas parsed as int, then .lower()'d by
    generated code that assumed it was always a string) should tell the LLM
    which argument had the wrong runtime type -- regression for the cache
    audit's "retry loop fails to converge within 3 attempts" finding
    (hepatitis x gpt-4o-mini, lymph x gpt-4o-mini: 'float' object has no
    attribute 'lower', 'int' object has no attribute 'isdigit'). The bare
    AttributeError message names the type but not which feature carried it,
    forcing the LLM to guess; surfacing "track_name=1979 (int)" points
    straight at the fix."""
    clf = SkribeClassifier(max_retries=1)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)
    prompts = []
    outputs = iter(
        [
            "def predict(**features):\n"
            "    return 1 if features['a'].lower().startswith('x') else 0\n",
            "def predict(**features): return 1",
        ]
    )

    def fake_llm(prompt, web_search=False):
        prompts.append(prompt)
        return next(outputs)

    monkeypatch.setattr(clf, "_call_llm", fake_llm)
    clf.fit(pd.DataFrame({"a": [1979]}), pd.Series([1], name="target"))
    assert len(prompts) == 2
    assert "'int' object has no attribute 'lower'" in prompts[1]
    assert "a=1979" in prompts[1] and "(int)" in prompts[1]


def test_fit_raises_after_exhausting_retries(monkeypatch):
    """When every attempt fails validation, the last error is surfaced."""
    clf = SkribeClassifier(max_retries=1)
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)
    monkeypatch.setattr(
        clf,
        "_call_llm",
        lambda prompt, web_search=False: "def predict(**features): raise ValueError('always broken')",
    )
    with pytest.raises(ValueError, match="always broken"):
        clf.fit(pd.DataFrame({"a": [1, 2]}), pd.Series([0, 1], name="target"))


def test_setstate_broken_code(monkeypatch):
    """Test __setstate__ with broken python_code_ triggers warning and fallback."""
    clf = SkribeClassifier()
    bad_state = dict(python_code_="def not_valid_code !@#", predict_fn=None, model="gpt-4o")
    # Should warn but not crash
    with pytest.warns(UserWarning):
        clf.__setstate__(bad_state)
    assert clf.predict_fn is None


def test_fit_too_many_rows(monkeypatch):
    """Test .fit() samples down if input too large."""
    import numpy as np
    import pandas as pd

    import skribe.utils
    from skribe.classifier import SkribeClassifier

    # Patch make_predict_fn at the module level
    monkeypatch.setattr(skribe.utils, "make_predict_fn", lambda code: lambda **features: 0)
    clf = SkribeClassifier(max_train_rows=5)
    # 10 rows will trigger down-sampling
    X = pd.DataFrame({"a": np.arange(10)})
    y = pd.Series(np.arange(10), name="target")
    # Patch _call_llm to return a stub function
    monkeypatch.setattr(
        clf,
        "_call_llm",
        lambda prompt, web_search=False: "def predict(**features): return 0",
    )
    # Now .fit should use the patched function
    clf.fit(X, y)
    assert clf.predict_fn is not None


def test_fit_blank_llm_output(monkeypatch):
    """Test .fit() with empty/whitespace LLM output triggers error."""
    clf = SkribeClassifier()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 2, 3], name="target")
    monkeypatch.setattr(clf, "_call_llm", lambda prompt, web_search=False: "   \n   ")
    with pytest.raises(ValueError, match="No code to exec from LLM output"):
        clf.fit(X, y)


def test_fit_nonstring_llm_output(monkeypatch):
    """Test .fit() with non-string LLM output is handled robustly."""
    import pandas as pd

    import skribe.utils

    # Patch BEFORE importing SkribeClassifier
    monkeypatch.setattr(skribe.utils, "make_predict_fn", lambda code: lambda **features: 0)
    from skribe.classifier import SkribeClassifier

    clf = SkribeClassifier()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 2, 3], name="target")
    monkeypatch.setattr(clf, "_call_llm", lambda prompt, web_search=False: 12345)
    # Expect ValueError because the LLM output isn't code
    with pytest.raises(ValueError, match="No valid function named 'predict'"):
        clf.fit(X, y)


def test_sample_calls_llm_and_parses(monkeypatch):
    """Test .sample() exercises LLM and TSV parsing logic."""
    clf = SkribeClassifier()
    # Patch _call_llm to return a simple TSV
    clf.feature_names_ = ["x1"]
    clf.target_name_ = "y"
    monkeypatch.setattr(clf, "_call_llm", lambda prompt, web_search=False: "a\tb\n1\t2\n3\t4")
    df = clf.sample(2)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"a", "b"}
    assert len(df) == 2


def test_sample_raises_before_fit():
    clf = SkribeClassifier()
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

    assert safe_exec_fn(just_return, {"x": "3.0"}, output_type=float, default=0.0) == 3.0
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
    clf = SkribeClassifier()
    clf.feature_names_ = ["foo", "bar"]
    clf.target_name_ = "baz"
    clf.python_code_ = "def predict(foo, bar): return 1"
    # Patch LLM call to return TSV
    monkeypatch.setattr(
        clf,
        "_call_llm",
        lambda prompt, web_search=False: "foo\tbar\tbaz\n1\t2\t3\n4\t5\t6",
    )
    df = clf.sample(2)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2


def test_generate_feature_dicts_invalid():
    # X is neither DataFrame nor ndarray
    with pytest.raises(ValueError, match="X must be a DataFrame or ndarray."):
        list(generate_feature_dicts("bad input", []))


def test_prepare_training_data_invalid():
    with pytest.raises(ValueError, match="X must be a pandas DataFrame or numpy array."):
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


def test_safe_exec_fn_preserves_numeric_looking_text_feature():
    """A free-text feature that happens to look numeric (e.g. a song title
    like "1979") must reach the generated function as the original string,
    not get silently coerced to an int -- regression for the spotify-genre
    'int' object has no attribute 'lower' bug found via the cache audit,
    where safe_exec_fn's numeric-string coercion ran unconditionally on
    every string feature, including free-text ones."""

    def fn(**features):
        return 1 if features["track_name"].lower().startswith("1") else 0

    result = safe_exec_fn(
        fn, {"track_name": "1979", "track_artist": "The Smashing Pumpkins"}, default=99
    )
    assert result == 1  # must NOT hit the except-block fallback of 99


def test_safe_exec_fn_still_coerces_numeric_string_when_fn_needs_it():
    """Existing safety net: a feature that's semantically numeric but arrived
    as a string, and whose function assumes it's already numeric, should
    still be coerced -- this is the fallback path, exercised only when
    calling the function with the raw string fails."""

    def fn(**features):
        return features["age"] + 1  # would TypeError on a str without coercion

    assert safe_exec_fn(fn, {"age": "5"}, output_type=int, default=0) == 6


# ---------------------------------------------------------------------------
# Signature and class-coverage validation (regression for commit 20ad44d)
# ---------------------------------------------------------------------------


def test_fit_rejects_positional_arg_missing_feature(monkeypatch):
    """predict() with fixed args that omit an expected feature must trigger a retry
    and ultimately raise with a clear message naming the missing argument."""
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, max_retries=0)
    # Returns a function that only accepts 'age', missing 'income'
    monkeypatch.setattr(clf, "_call_llm", lambda p, web_search=False: "def predict(age): return 0")
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)

    with pytest.raises(ValueError, match="missing expected feature arguments"):
        clf.fit(
            pd.DataFrame({"age": [25, 40], "income": [30000, 80000]}),
            pd.Series([0, 1]),
        )


def test_fit_kwargs_signature_passes_validation(monkeypatch):
    """predict(**features) always passes the signature check regardless of column names."""
    clf = SkribeClassifier(model="gpt-5.4-mini", verbose=False, max_retries=0)
    monkeypatch.setattr(
        clf,
        "_call_llm",
        lambda p, web_search=False: "def predict(**features): return int(features.get('a', 0) > 10)",
    )
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)

    X = pd.DataFrame({"a": list(range(20))})
    y = pd.Series([0] * 10 + [1] * 10)
    clf.fit(X, y)  # must not raise
