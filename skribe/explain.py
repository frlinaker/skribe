"""A small, JSON-serializable explanation container.

The shape mirrors the de-facto convention used by libraries such as Alibi: an
``Explanation`` holds two dicts — ``meta`` (explainer metadata: name, type,
scope, params) and ``data`` (the explanation itself, e.g. a plain-English
``summary`` and the ``features_used``). Top-level keys of either dict are also
exposed as attributes for convenience, and the object round-trips through JSON.
"""

import json
from collections import ChainMap
from typing import Any, Dict


class Explanation:
    def __init__(self, meta: Dict[str, Any], data: Dict[str, Any]):
        self.meta = dict(meta)
        self.data = dict(data)

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal attribute lookup fails. Expose top-level
        # keys of data/meta as attributes; __dict__.get avoids recursion while
        # the object is still being constructed or unpickled.
        chain = ChainMap(self.__dict__.get("data", {}), self.__dict__.get("meta", {}))
        if name in chain:
            return chain[name]
        raise AttributeError(name)

    def __str__(self) -> str:
        return str(self.data.get("summary", ""))

    def __repr__(self) -> str:
        return f"Explanation(meta={self.meta!r}, data={self.data!r})"

    def to_json(self) -> str:
        return json.dumps({"meta": self.meta, "data": self.data})

    @classmethod
    def from_json(cls, text: str) -> "Explanation":
        obj = json.loads(text)
        return cls(meta=obj["meta"], data=obj["data"])
