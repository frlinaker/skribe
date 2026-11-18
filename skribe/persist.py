"""On-disk persistence of everything an LLM returns during ``fit()``.

Every LLM call (prompt, raw response or error, timing) and every generated
code candidate -- including the ones that failed validation, together with the
error -- is written under a per-fit run directory, so a failed or surprising
fit can be investigated after the fact instead of being lost with the process.

Layout (one directory per ``fit()``)::

    <log_dir>/<utc-timestamp>_<model>_<pid>_<n>/
        meta.json                      estimator params, versions, start time
        events.jsonl                   one JSON line per event, in order
        003_llm_call_prompt.txt        large text lives in sidecar files that
        003_llm_call_response.txt      events.jsonl references by name
        004_code_attempt_raw.py
        004_code_attempt_extended.py   the code that actually failed/passed
        ...

Persistence is on by default. ``log_dir`` (estimator argument) wins over the
``SKRIBE_LOG_DIR`` environment variable, which wins over ``./skribe_logs``.
``log_dir=False`` or ``SKRIBE_LOG_DIR=off`` disables it. Writing never raises:
a full disk or unwritable directory must not break a fit, so the first
failure logs one warning and disables further writes for that run.

Note the logs contain the full prompts, which include a sample of the
training data, plus any web-search results. Treat the directory like the
data itself.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger("skribe")

DEFAULT_LOG_DIRNAME = "skribe_logs"
ENV_VAR = "SKRIBE_LOG_DIR"
_OFF_VALUES = {"", "0", "off", "false", "none", "no"}
_run_counter = itertools.count()


def resolve_log_dir(log_dir: Union[str, os.PathLike, bool, None]) -> Optional[Path]:
    """Return the base directory to write under, or None when disabled."""
    if log_dir is False:
        return None
    if log_dir is None or log_dir is True:
        env = os.environ.get(ENV_VAR)
        if env is not None:
            return None if env.strip().lower() in _OFF_VALUES else Path(env).expanduser()
        return Path(DEFAULT_LOG_DIRNAME)
    return Path(log_dir).expanduser()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60] or "model"


class RunLog:
    """Append-only record of one ``fit()``. All methods are failure-safe."""

    def __init__(self, base_dir: Path, model: str, meta: Optional[dict] = None):
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.dir = (
            Path(base_dir).resolve() / f"{stamp}_{_slug(model)}_{os.getpid()}_{next(_run_counter)}"
        )
        self._seq = 0
        self._broken = False
        self._write(lambda: self.dir.mkdir(parents=True, exist_ok=True))
        if meta is not None:
            self._write(
                lambda: (self.dir / "meta.json").write_text(
                    json.dumps({"started_utc": stamp, **meta}, indent=2, default=str)
                )
            )

    def _write(self, fn) -> None:
        if self._broken:
            return
        try:
            fn()
        except OSError as e:
            self._broken = True
            logger.warning("skribe: could not write LLM logs under %s (%s); disabled.", self.dir, e)

    def event(self, kind: str, files: Optional[dict] = None, **fields) -> None:
        """Record one event.

        ``files`` maps a short name to ``(text, extension)``; each is written
        to its own sidecar file and referenced by name from the event line.
        Everything else goes inline in ``events.jsonl``.
        """
        self._seq += 1
        seq = self._seq
        record = {"seq": seq, "time_utc": time.strftime("%H:%M:%S", time.gmtime()), "kind": kind}
        record.update(fields)
        for name, (text, ext) in (files or {}).items():
            if text is None:
                continue
            fname = f"{seq:03d}_{kind}_{name}.{ext}"
            self._write(lambda f=fname, t=text: (self.dir / f).write_text(str(t)))
            record.setdefault("files", {})[name] = fname
        line = json.dumps(record, default=str) + "\n"
        self._write(lambda: self._append("events.jsonl", line))

    def _append(self, name: str, text: str) -> None:
        with open(self.dir / name, "a") as f:
            f.write(text)
