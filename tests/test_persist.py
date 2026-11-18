"""LLM-output persistence: every call and every code candidate (including the
ones that fail validation) lands on disk by default, and can be turned off."""

import json
import pickle

import pandas as pd
import pytest
from sklearn.base import clone

from skribe import SkribeClassifier
from skribe.persist import resolve_log_dir

GOOD = "def predict(**f):\n    return 1 if f['a'] > 1 else 0\n"
BAD = "def predict(**f):\n    return undefined_thing\n"


def _data():
    X = pd.DataFrame({"a": [0, 1, 2, 3, 0, 1, 2, 3]})
    y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    return X, y


def _clf(monkeypatch, responses, **kw):
    clf = SkribeClassifier(model="stub-model", verbose=False, context_prepass=False, **kw)
    calls = iter(responses)
    monkeypatch.setattr(clf, "_dispatch_llm_call", lambda *a, **k: next(calls))
    monkeypatch.setattr(clf, "_extend_code", lambda code, web_search=False: code)
    return clf


def _runs(base):
    return [d for d in base.iterdir() if d.is_dir()]


def _events(run):
    return [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]


def test_calls_and_code_persisted_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("SKRIBE_LOG_DIR", str(tmp_path))
    clf = _clf(monkeypatch, [GOOD])
    clf.fit(*_data())

    (run,) = _runs(tmp_path)
    assert json.loads((run / "meta.json").read_text())["model"] == "stub-model"
    events = _events(run)
    call = next(e for e in events if e["kind"] == "llm_call")
    assert call["status"] == "ok"
    assert (run / call["files"]["response"]).read_text() == GOOD
    assert "predict" in (run / call["files"]["prompt"]).read_text()
    attempt = next(e for e in events if e["kind"] == "code_attempt")
    assert attempt["status"] == "ok"
    done = next(e for e in events if e["kind"] == "fit_done")
    assert (run / done["files"]["final_code"]).read_text() == GOOD


def test_failed_code_attempt_is_persisted_with_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SKRIBE_LOG_DIR", str(tmp_path))
    clf = _clf(monkeypatch, [BAD, GOOD])
    clf.fit(*_data())

    (run,) = _runs(tmp_path)
    attempts = [e for e in _events(run) if e["kind"] == "code_attempt"]
    assert [a["status"] for a in attempts] == ["failed", "ok"]
    failed = attempts[0]
    assert "undefined_thing" in (run / failed["files"]["raw"]).read_text()
    assert "undefined_thing" in (run / failed["files"]["extended"]).read_text()
    assert "is not defined" in (run / failed["files"]["error"]).read_text()
    assert "is not defined" in failed["error"]


def test_failed_llm_call_is_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("SKRIBE_LOG_DIR", str(tmp_path))
    clf = SkribeClassifier(model="stub-model", verbose=False, context_prepass=False)

    def boom(*a, **k):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(clf, "_dispatch_llm_call", boom)
    with pytest.raises(RuntimeError):
        clf.fit(*_data())

    (run,) = _runs(tmp_path)
    call = next(e for e in _events(run) if e["kind"] == "llm_call")
    assert call["status"] == "error"
    assert "provider exploded" in call["error"]
    assert "prompt" in call["files"]


def test_each_fit_gets_its_own_run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SKRIBE_LOG_DIR", str(tmp_path))
    clf = _clf(monkeypatch, [GOOD, GOOD])
    clf.fit(*_data())
    clf.fit(*_data())
    assert len(_runs(tmp_path)) == 2


def test_log_dir_argument_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SKRIBE_LOG_DIR", str(tmp_path / "env"))
    clf = _clf(monkeypatch, [GOOD], log_dir=tmp_path / "arg")
    clf.fit(*_data())
    assert _runs(tmp_path / "arg")
    assert not (tmp_path / "env").exists()


@pytest.mark.parametrize("via", ["arg", "env"])
def test_can_be_disabled(tmp_path, monkeypatch, via):
    monkeypatch.chdir(tmp_path)
    if via == "env":
        monkeypatch.setenv("SKRIBE_LOG_DIR", "off")
        clf = _clf(monkeypatch, [GOOD])
    else:
        monkeypatch.delenv("SKRIBE_LOG_DIR", raising=False)
        clf = _clf(monkeypatch, [GOOD], log_dir=False)
    clf.fit(*_data())
    assert list(tmp_path.iterdir()) == []


def test_default_is_skribe_logs_in_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SKRIBE_LOG_DIR", raising=False)
    assert resolve_log_dir(None).name == "skribe_logs"
    clf = _clf(monkeypatch, [GOOD])
    clf.fit(*_data())
    assert _runs(tmp_path / "skribe_logs")


def test_unwritable_dir_does_not_break_fit(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    clf = _clf(monkeypatch, [GOOD], log_dir=blocker / "logs")
    clf.fit(*_data())  # must not raise
    assert clf.predict(pd.DataFrame({"a": [3]}))[0] == 1


def test_clone_and_pickle_keep_working(tmp_path, monkeypatch):
    monkeypatch.setenv("SKRIBE_LOG_DIR", str(tmp_path))
    clf = _clf(monkeypatch, [GOOD], log_dir=tmp_path / "x")
    assert clone(clf).get_params()["log_dir"] == tmp_path / "x"
    clf.fit(*_data())
    monkeypatch.undo()  # drop the unpicklable test stubs; logging state is under test
    restored = pickle.loads(pickle.dumps(clf))
    assert restored._run_log is None
    assert restored.predict(pd.DataFrame({"a": [3]}))[0] == 1
