import joblib
import pandas as pd
from promptlearn import PromptClassifier

def test_zero_row_classifier_runs():
    X = pd.DataFrame(columns=["country_name"])
    y = pd.Series(name="has_blue_in_flag", dtype=int)

    clf = PromptClassifier(verbose=False)
    clf.fit(X, y)

    result = clf.predict(pd.DataFrame([{"country_name": "France"}]))
    assert isinstance(result[0], int)

def test_promptclassifier_joblib_roundtrip(tmp_path):
    # Define a minimal "trained" classifier manually
    clf = PromptClassifier(model="o4-mini", verbose=False)
    clf.feature_names_in_ = ["name"]
    clf.target_name_ = "is_animal"
    clf.heuristic_ = "IF name in {cat, dog, tiger} THEN is_animal = 1 ELSE 0"

    # Save to disk
    model_path = tmp_path / "clf.joblib"
    joblib.dump(clf, model_path)

    # Load it back
    loaded = joblib.load(model_path)

    # Confirm state round-tripped
    assert loaded.model == "o4-mini"
    assert loaded.target_name_ == "is_animal"
    assert loaded.heuristic_.startswith("IF name")

    # Run prediction
    X = pd.DataFrame([{"name": "cat"}, {"name": "car"}])
    preds = loaded.predict(X)

    assert isinstance(preds, list)
    assert all(isinstance(p, int) for p in preds)
