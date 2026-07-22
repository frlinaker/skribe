"""Pure postprocessing of LLM-generated predict()/transform() code: find bare
numeric literals that sit in a threshold, coefficient, or dict-lookup
position, fit a real value for each against the training sample, and splice
it in place of the LLM's original literal -- but only when doing so does not
reduce accuracy on a validation sample.

This is deliberately a standalone, dependency-injected component: it takes no
implicit state from the estimator that owns it (the caller passes
``is_classification`` explicitly rather than this module reaching back into
``BaseSkribeEstimator._task_type``), and it requires no cooperation from the
prompt -- there is no marker convention (e.g. a ``calibrate()`` call) for the
LLM to use or forget to use. Every qualifying literal in the generated code
is a candidate; the accuracy safety rail is what keeps this from being
harmful rather than any declared LLM intent, since a literal that was already
correct (e.g. a genuine domain constant) simply won't score better after
being "fit" on a training sample and will be rejected.
"""

import ast
from typing import Callable, Optional

from .utils import make_predict_fn

# Small integers in threshold/coefficient position are almost always
# structural (counts, indices, boolean-like flags -- e.g. "== 2" meaning
# "exactly two matches") rather than a dataset-dependent guess worth
# refitting. Floats and larger integers are fair game. This bound is
# deliberately conservative -- it only needs to exclude the common
# structural cases, not every possible one, since the accuracy safety rail
# (see ConstantPostProcessor.process) rejects any refit that doesn't
# actually help regardless of which literals were considered.
_SMALL_INT_SKIP_BOUND = 3


def _bare_feature_name(node: ast.AST) -> Optional[str]:
    """A feature reference with no coercion wrapper: either a bare local
    variable name (``age``) or a direct ``features["age"]``/``features['age']``
    subscript, which the coercion-first prompt instruction is meant to
    discourage but doesn't strictly prevent."""
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "features"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    return None


def _feature_name_from_operand(node: ast.AST) -> Optional[str]:
    """Extract the feature variable name from the "other side" of a
    comparison or multiplication involving a numeric literal, e.g.
    ``age`` from a bare ``Name``, from ``float(age)``/``int(age)``, or from
    ``float(features["age"])``/a direct ``features["age"]`` subscript."""
    name = _bare_feature_name(node)
    if name is not None:
        return name
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("float", "int")
        and len(node.args) == 1
    ):
        return _bare_feature_name(node.args[0])
    return None


def _numeric_default_value(node: ast.AST) -> Optional[float]:
    """Extract a numeric literal, including a negative one -- Python parses
    ``-0.85`` as ``UnaryOp(USub, Constant(0.85))``, not a single Constant, so
    a negative default (a very common value, e.g. any negative risk weight)
    would otherwise silently fail to be recognized at all."""
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        value = float(node.operand.value)
        return -value if isinstance(node.op, ast.USub) else value
    return None


def _is_tunable_literal(node: ast.AST) -> bool:
    """Whether a numeric literal is worth treating as a calibration
    candidate at all -- excludes small integers (see
    ``_SMALL_INT_SKIP_BOUND``) and booleans, which ``_numeric_default_value``
    already excludes."""
    value = _numeric_default_value(node)
    if value is None:
        return False
    if isinstance(node, ast.Constant):
        raw = node.value
    elif isinstance(node, ast.UnaryOp):
        raw = node.operand.value
    else:
        return False
    is_int_literal = isinstance(raw, int)
    return not (is_int_literal and abs(value) <= _SMALL_INT_SKIP_BOUND)


def _outermost_sum(node: ast.AST, parent_of: dict) -> ast.AST:
    """Walk up through a chain of ``BinOp(Add/Sub)`` nodes so that every
    coefficient term in the same linear combination is grouped together for
    joint fitting, e.g. all three numeric-literal coefficient terms in
    ``0.3*a + 0.7*b - 0.4*c``."""
    current = node
    while True:
        parent = parent_of.get(id(current))
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, (ast.Add, ast.Sub)):
            current = parent
            continue
        break
    return current


def _find_calibration_sites(tree: ast.AST) -> list:
    """Find every bare numeric literal in ``tree`` that sits in a position
    worth calibrating against training data: one side of a comparison
    against a feature (a "threshold"), multiplied against a feature inside a
    sum (a "coefficient"), or a value inside a string-keyed dict literal
    matched against a feature's categories (a "dict_entry").

    A site only qualifies if it can actually be tied to a feature (or, for
    dict_entry, to the enclosing dict), since a literal with no identifiable
    feature has nothing to fit against and is left alone.
    """
    parent_of: dict = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_of[id(child)] = parent

    sites = []
    site_index = 0
    for node in ast.walk(tree):
        if not _is_tunable_literal(node):
            continue
        default = _numeric_default_value(node)
        parent = parent_of.get(id(node))

        feature = None
        kind = None
        group = None
        dict_key = None

        if isinstance(parent, ast.Compare):
            candidates = [parent.left] + list(parent.comparators)
            other = next((c for c in candidates if c is not node), None)
            feature = _feature_name_from_operand(other) if other is not None else None
            if feature is not None:
                kind = "threshold"
        elif (
            isinstance(parent, ast.BinOp)
            and isinstance(parent.op, ast.Mult)
            and (parent.left is node or parent.right is node)
        ):
            other = parent.right if parent.left is node else parent.left
            feature = _feature_name_from_operand(other)
            if feature is not None:
                kind = "coefficient"
                group = _outermost_sum(parent, parent_of)
        elif isinstance(parent, ast.Dict) and node in parent.values:
            key_index = parent.values.index(node)
            key_node = parent.keys[key_index]
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                kind = "dict_entry"
                dict_key = key_node.value
                group = parent  # the enclosing Dict node groups sibling entries

        if kind is None:
            continue

        site_index += 1
        sites.append(
            {
                "node": node,
                "name": f"site_{site_index}",
                "default": float(default),
                "feature": feature,
                "kind": kind,
                "group": group if group is not None else node,
                "dict_key": dict_key,
            }
        )
    return sites


def _fit_threshold(values: list, labels: list, default: float, is_classification: bool) -> float:
    """Best single-feature split point: scan candidate splits (midpoints of
    sorted unique values) and keep whichever maximizes training accuracy
    (classification) or minimizes MSE of the two-sided constant fit
    (regression). Falls back to ``default`` if the feature has fewer than 2
    distinct usable values."""
    pairs = [(v, y) for v, y in zip(values, labels) if v is not None]
    if len(pairs) < 2:
        return default
    xs = sorted(set(v for v, _ in pairs))
    if len(xs) < 2:
        return default
    candidates = [(xs[i] + xs[i + 1]) / 2 for i in range(len(xs) - 1)]

    best_split = default
    best_score = None
    for split in candidates:
        if is_classification:
            # Orient each side to whichever class is more common there --
            # this mirrors what the LLM's generated branch does (predict one
            # class above the split, another below), without needing to know
            # which side the generated code calls "true".
            above = [y for v, y in pairs if v > split]
            below = [y for v, y in pairs if v <= split]
            correct = 0
            for side in (above, below):
                if side:
                    majority = max(set(side), key=side.count)
                    correct += sum(1 for y in side if y == majority)
            score = correct / len(pairs)
            better = best_score is None or score > best_score
        else:
            above = [y for v, y in pairs if v > split]
            below = [y for v, y in pairs if v <= split]
            sse = 0.0
            for side in (above, below):
                if side:
                    mean = sum(side) / len(side)
                    sse += sum((y - mean) ** 2 for y in side)
            score = -sse
            better = best_score is None or score > best_score
        if better:
            best_score = score
            best_split = split
    return best_split


def _fit_coefficients(
    group_sites: list, feature_values: dict, labels: list, is_classification: bool
) -> dict:
    """Fit a joint linear model over every feature in one ``literal *
    feature`` sum, using sklearn LogisticRegression (classification) or
    LinearRegression (regression). Returns {name: fitted_coefficient}. Falls
    back to each site's own default if fitting isn't possible (too few rows,
    a single class, a non-numeric feature)."""
    defaults = {s["name"]: s["default"] for s in group_sites}
    features = [s["feature"] for s in group_sites]
    columns = [feature_values.get(f) for f in features]
    if not labels or any(col is None for col in columns):
        return defaults

    n = len(labels)
    if any(len(col) != n for col in columns):
        return defaults

    rows = list(zip(*columns))
    usable = [(row, y) for row, y in zip(rows, labels) if all(v is not None for v in row)]
    if len(usable) < max(4, len(features) + 1):
        return defaults

    X = [list(row) for row, _ in usable]
    y = [label for _, label in usable]

    try:
        if is_classification:
            if len(set(y)) < 2:
                return defaults
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression()
            model.fit(X, y)
            coefs = model.coef_[0]
        else:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(X, y)
            coefs = model.coef_
    except Exception:
        return defaults

    return {s["name"]: float(c) for s, c in zip(group_sites, coefs)}


def _normalize_categorical(value) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _find_indexing_feature(dict_node: ast.Dict, rows: list) -> Optional[str]:
    """Guess which raw feature a dict-literal lookup table (e.g.
    ``{"critical/other existing credit": -0.85, ...}``) is keyed by, by
    checking which feature's raw training values actually match the dict's
    string keys most often -- more robust than tracing the ``.get()`` call
    site back through whatever normalization helper the LLM wrote
    (``_norm()``, ``.lower()``, etc.), since this matches against the real
    data directly. Returns None if no feature matches well enough to trust.
    """
    keys = {
        _normalize_categorical(k.value)
        for k in dict_node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    if not keys or not rows:
        return None

    best_feature, best_rate = None, 0.0
    for feature in rows[0].keys():
        matches = sum(1 for row in rows if _normalize_categorical(row.get(feature)) in keys)
        rate = matches / len(rows)
        if rate > best_rate:
            best_feature, best_rate = feature, rate

    # Require most rows to actually hit a key in this dict -- a feature that
    # only coincidentally matches a handful of rows isn't the real index.
    return best_feature if best_rate >= 0.5 else None


def _fit_dict_weights(
    dict_node: ast.Dict, group_sites: list, rows: list, labels: list, is_classification: bool
) -> dict:
    """Fit a joint one-hot linear model over every key in a numeric-valued
    dict-literal lookup table, e.g.
    ``{"critical/other existing credit": -0.85, ...}``.

    Every key in the dict becomes a one-hot indicator column (not just the
    ones ``group_sites`` proposes refitting) so the fit captures each
    category's contribution relative to the others, then returns fitted
    values only for the sites actually passed in -- other numeric entries in
    the same dict may exist but aren't in ``group_sites`` if they didn't
    qualify as tunable (see ``_is_tunable_literal``).

    This regresses directly against the training label rather than the
    function's own intermediate running total (which isn't observable
    without re-executing the whole function), so it recovers each
    category's relative sign and ordering correctly -- enough to fix a
    reversed-polarity category -- even though the absolute scale may not
    exactly match the rest of the function's additive scoring. Falls back
    to each site's own default if no feature can be matched to this dict's
    keys or fitting is otherwise degenerate.
    """
    defaults = {s["name"]: s["default"] for s in group_sites}
    all_keys = [
        k.value for k in dict_node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
    ]
    if len(all_keys) < 2:
        return defaults

    feature = _find_indexing_feature(dict_node, rows)
    if feature is None or not labels or len(rows) != len(labels):
        return defaults

    normalized_keys = [_normalize_categorical(k) for k in all_keys]
    X = [
        [1.0 if _normalize_categorical(row.get(feature)) == nk else 0.0 for nk in normalized_keys]
        for row in rows
    ]
    y = list(labels)
    # Drop rows whose feature value matched none of this dict's keys -- a
    # zero one-hot row provides no signal about the dict's own weights.
    usable = [(x, label) for x, label in zip(X, y) if any(x)]
    if len(usable) < max(4, len(all_keys) + 1):
        return defaults
    X = [x for x, _ in usable]
    y = [label for _, label in usable]

    try:
        if is_classification:
            if len(set(y)) < 2:
                return defaults
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression()
            model.fit(X, y)
            coefs = model.coef_[0]
        else:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(X, y)
            coefs = model.coef_
    except Exception:
        return defaults

    fitted_by_key = dict(zip(normalized_keys, coefs))
    result = {}
    for s in group_sites:
        nk = _normalize_categorical(s["dict_key"])
        result[s["name"]] = float(fitted_by_key.get(nk, s["default"]))
    return result


def _score_accuracy(fn: Callable, rows: list, labels: list, is_classification: bool) -> float:
    """Score a compiled predict/transform function against a validation
    sample: accuracy for classification, negative MSE for regression (so
    "higher is better" holds in both cases). Any row that raises is scored
    as wrong/maximally-off rather than aborting the whole comparison, since
    a candidate that crashes on some input is exactly the kind of
    regression the safety rail exists to catch.
    """
    if not rows or not labels or len(rows) != len(labels):
        return float("-inf")
    if is_classification:
        correct = 0
        for row, label in zip(rows, labels):
            try:
                pred = fn(**row)
            except Exception:
                continue
            if pred is not None and int(pred) == int(label):
                correct += 1
        return correct / len(rows)
    sse = 0.0
    for row, label in zip(rows, labels):
        try:
            pred = fn(**row)
            sse += (float(pred) - float(label)) ** 2
        except Exception:
            sse += float(label) ** 2 + 1.0  # penalize a crash more than predicting 0
    return -sse / len(rows)


# Fraction of rows held out purely for accept/reject scoring, never used to
# fit a candidate constant -- see ConstantPostProcessor.process.
_HOLDOUT_FRACTION = 0.3
# Below this many rows, splitting leaves too little on either side to fit or
# score meaningfully; process() falls back to reusing all rows for both.
_MIN_ROWS_FOR_SPLIT = 20


def _fit_holdout_split(rows: list, labels: list) -> tuple:
    """Deterministically split ``rows``/``labels`` into a fit slice and a
    holdout slice using an every-Nth-row stride, so the holdout set is spread
    across the whole sample (e.g. across whatever class/value ordering the
    caller's data happens to have) rather than being one contiguous block
    that could be systematically unrepresentative.

    Returns ``(fit_rows, fit_labels, holdout_rows, holdout_labels)``. The
    holdout slice is empty if there isn't enough data to split meaningfully.
    """
    n = len(rows)
    if n < _MIN_ROWS_FOR_SPLIT:
        return rows, labels, [], []

    stride = max(2, round(1 / _HOLDOUT_FRACTION))
    holdout_idx = set(range(0, n, stride))
    fit_rows = [r for i, r in enumerate(rows) if i not in holdout_idx]
    fit_labels = [v for i, v in enumerate(labels) if i not in holdout_idx]
    holdout_rows = [r for i, r in enumerate(rows) if i in holdout_idx]
    holdout_labels = [v for i, v in enumerate(labels) if i in holdout_idx]
    return fit_rows, fit_labels, holdout_rows, holdout_labels


def _splice_literals(code: str, replacements: list) -> str:
    """Replace each numeric literal's exact source span with a new literal,
    using absolute character offsets (computed from each node's line/column
    position) rather than per-line splicing, so a literal that happens to
    sit inside a call wrapping across multiple lines -- e.g. a long dict
    literal reformatted by the LLM -- is still spliced correctly instead of
    being silently left in place.

    Replacing a site here is optional, not mandatory: a bare numeric literal
    is already valid Python on its own, so a site that the accuracy safety
    rail rejects (see ``ConstantPostProcessor.process``) simply isn't
    included in ``replacements`` and is left completely untouched.
    """
    if not replacements:
        return code
    line_starts = [0]
    for line in code.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    def offset(lineno: int, col: int) -> int:
        return line_starts[lineno - 1] + col

    spans = sorted(
        (
            (
                offset(node.lineno, node.col_offset),
                offset(node.end_lineno, node.end_col_offset),
                value,
            )
            for node, value in replacements
        ),
        key=lambda s: s[0],
        reverse=True,
    )
    for start, end, value in spans:
        code = code[:start] + repr(value) + code[end:]
    return code


def find_unverified_thresholds(code: str) -> list:
    """Find calibration-worthy numeric literals in ``code`` -- thresholds and
    coefficients only (dict_entry sites are excluded, since a category
    lookup table isn't the kind of single fact a web search verifies).

    This is the deterministic backstop for the fit-time verification gate: a
    threshold or coefficient the LLM hardcoded without evidence gets caught
    here purely structurally, using the exact same detector
    ``ConstantPostProcessor.process`` uses to find calibration candidates in
    general. Returns a list of ``{"feature": str|None, "literal": float,
    "kind": str}`` dicts, one per site found; empty if the code has none.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    sites = _find_calibration_sites(tree)
    return [
        {"feature": s["feature"], "literal": s["default"], "kind": s["kind"]}
        for s in sites
        if s["kind"] in ("threshold", "coefficient")
    ]


class NoOpPostProcessor:
    """The default postprocessor: returns ``code`` unchanged.

    Constant tuning is opt-in -- pass ``postprocessor=ConstantPostProcessor()``
    to a ``SkribeClassifier``/``SkribeRegressor``/``SkribeFeatureEngineer`` to
    enable it. ``__eq__``/``__hash__`` mirror ``ConstantPostProcessor``'s, for
    the same sklearn ``clone()`` reason.
    """

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))

    def process(self, code: str, rows: list, labels: list, is_classification: bool) -> str:
        return code


class ConstantPostProcessor:
    """Standalone, dependency-injected postprocessor for generated
    predict()/transform() code. See the module docstring for the overall
    approach; :meth:`process` is the single entry point.

    Not enabled by default -- see ``NoOpPostProcessor``, which
    ``BaseSkribeEstimator`` uses unless a caller explicitly injects this
    class instead.

    Stateless, so any two instances are interchangeable -- ``__eq__`` reflects
    that (rather than falling back to identity) so sklearn's ``clone()``,
    which reconstructs a fresh instance from ``get_params()``, round-trips
    equal params instead of failing an identity comparison against the
    original instance it copied.
    """

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))

    def process(self, code: str, rows: list, labels: list, is_classification: bool) -> str:
        """Find bare numeric literals in ``code`` that sit in a threshold,
        coefficient, or dict-lookup position, fit a real value for each
        against a held-out-aware split of ``rows``/``labels``, and splice it
        in place of the original literal -- but ONLY when doing so does not
        reduce accuracy on the held-out slice versus the code as it stood
        before that group's fit.

        ``rows``/``labels`` are internally split into a fit slice (used to
        fit each candidate constant) and a holdout slice (used to score
        accept/reject) so a group can't be judged on the same rows it was
        fit against -- fitting and scoring on identical data would make
        every fit look like an improvement even when it's just memorizing
        that slice.

        Coefficient/dict-entry sites sharing the same enclosing sum or dict
        literal are fit and accepted/rejected jointly, as one group -- their
        values interact, so evaluating them one at a time would compare
        against an ill-defined baseline. Groups are otherwise applied
        greedily and sequentially: each accepted group's splice becomes the
        baseline the next group is compared against.

        Returns plain Python source with only the literals that passed the
        accuracy check changed; falls back to the original ``code``
        unchanged if nothing qualifies, if there are no rows/labels to score
        against, or if nothing improves.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        sites = _find_calibration_sites(tree)
        if not sites:
            return code

        if not rows or not labels or len(rows) != len(labels):
            return code

        fit_rows, fit_labels, holdout_rows, holdout_labels = _fit_holdout_split(rows, labels)
        if not holdout_rows:
            # Too little data for a meaningful split -- fit and score on the
            # same rows rather than skipping postprocessing entirely.
            fit_rows, fit_labels = rows, labels
            holdout_rows, holdout_labels = rows, labels

        feature_values: dict = {}
        for row in fit_rows:
            for k, v in row.items():
                try:
                    feature_values.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    feature_values.setdefault(k, []).append(None)

        try:
            label_values = [float(v) for v in fit_labels]
        except (TypeError, ValueError):
            label_values = []

        # Coefficient/dict-entry sites that share the same enclosing sum (or
        # the same dict literal) are fit jointly; group by a structural key
        # (kind + ordered site identities), not id() of any AST node -- node
        # objects go stale the moment code is re-parsed after a splice, so
        # groups are re-discovered fresh from scratch on each pass and
        # matched back to their processing order by this key instead.
        def _group_key(group_sites: list) -> tuple:
            return tuple(s["name"] for s in group_sites)

        first_pass_groups: dict = {}
        for site in sites:
            first_pass_groups.setdefault(id(site["group"]), (site["kind"], []))[1].append(site)
        ordered_keys = [
            (kind, _group_key(group_sites)) for kind, group_sites in first_pass_groups.values()
        ]

        current_code = code
        try:
            current_fn = make_predict_fn(current_code)
        except Exception:
            return code
        current_score = _score_accuracy(
            current_fn, holdout_rows, holdout_labels, is_classification=is_classification
        )

        for target_key in ordered_keys:
            try:
                current_tree = ast.parse(current_code)
            except SyntaxError:
                break
            current_sites = _find_calibration_sites(current_tree)
            current_groups: dict = {}
            for site in current_sites:
                current_groups.setdefault(id(site["group"]), (site["kind"], site["group"], []))[
                    2
                ].append(site)
            match = next(
                (
                    (kind, group_node, group_sites)
                    for kind, group_node, group_sites in current_groups.values()
                    if (kind, _group_key(group_sites)) == target_key
                ),
                None,
            )
            if match is None:
                # This group's sites no longer appear (e.g. a prior accepted
                # splice happened to remove them) -- nothing left to do.
                continue
            kind, group_node, group_sites = match

            if kind == "threshold":
                site = group_sites[0]
                feature = site["feature"]
                default = site["default"]
                fitted = default
                values = feature_values.get(feature)
                if values and label_values and len(values) == len(label_values):
                    fitted = _fit_threshold(
                        values, label_values, default, is_classification=is_classification
                    )
                candidate_replacements = [(site["node"], fitted)]
            elif kind == "coefficient":
                fitted_map = _fit_coefficients(
                    group_sites, feature_values, label_values, is_classification=is_classification
                )
                candidate_replacements = [
                    (site["node"], fitted_map.get(site["name"], site["default"]))
                    for site in group_sites
                ]
            else:  # dict_entry
                fitted_map = _fit_dict_weights(
                    group_node,
                    group_sites,
                    fit_rows,
                    fit_labels,
                    is_classification=is_classification,
                )
                candidate_replacements = [
                    (site["node"], fitted_map.get(site["name"], site["default"]))
                    for site in group_sites
                ]

            candidate_code = _splice_literals(current_code, candidate_replacements)
            if candidate_code == current_code:
                continue
            try:
                candidate_fn = make_predict_fn(candidate_code)
            except Exception:
                continue
            candidate_score = _score_accuracy(
                candidate_fn, holdout_rows, holdout_labels, is_classification=is_classification
            )
            if candidate_score < current_score:
                continue

            # Accept: this group's fit becomes part of the working code for
            # every subsequent group's comparison.
            current_code = candidate_code
            current_score = candidate_score

        return current_code
