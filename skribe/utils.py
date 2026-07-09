import logging
import re
from io import StringIO
from typing import Any, Callable, Dict, Type

import numpy as np
import pandas as pd

logger = logging.getLogger("skribe")


# Helper for robust Python identifier normalization
def normalize_feature_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    name = re.sub(r"__+", "_", name)
    return name.strip("_").lower()


def generate_feature_dicts(X, feature_names):
    """
    Returns an iterable of feature dicts with normalized keys, from X (DataFrame or ndarray).
    """

    def normalize_keys(d):
        return {normalize_feature_name(k): v for k, v in d.items()}

    if isinstance(X, pd.DataFrame):
        for _, row in X.iterrows():
            yield normalize_keys(row.to_dict())
    elif isinstance(X, np.ndarray):
        cols = [normalize_feature_name(c) for c in feature_names]
        for arr in X:
            yield dict(zip(cols, arr))
    else:
        raise ValueError("X must be a DataFrame or ndarray.")


def sanitize_dataset_description(text: str) -> str:
    """Clean user-supplied dataset description before embedding it in an LLM prompt.

    Strips leading/trailing whitespace, removes curly braces (which would break
    prompt template substitution), and collapses internal whitespace runs.
    Prompt injection cannot be fully prevented, but we avoid making it worse.
    """
    text = text.strip()
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def extract_python_code(text: str) -> str:
    # Remove code fences and cut at any obvious example markers
    if "```python" in text:
        text = text.split("```python", 1)[-1]
    if "```" in text:
        text = text.split("```", 1)[0]
    return text


def prepare_training_data(X, y):
    """
    Returns: data (pd.DataFrame), feature_names (list), target_name (str)
    """
    if isinstance(X, pd.DataFrame):
        data = X.copy()
        target_name = normalize_feature_name(y.name if hasattr(y, "name") and y.name else "target")
        data[target_name] = y.values if hasattr(y, "values") else y
        # Normalize all columns (including target)
        data.columns = [normalize_feature_name(col) for col in data.columns]
        # Feature names: all except target
        feature_names = [col for col in data.columns if col != target_name]
    elif isinstance(X, np.ndarray):
        n_features = X.shape[1]
        feature_names = [f"col{i}" for i in range(n_features)]
        target_name = "target"
        data = pd.DataFrame(X, columns=feature_names)
        data[target_name] = y
        # Already safe names
    else:
        raise ValueError("X must be a pandas DataFrame or numpy array.")
    return data, feature_names, target_name


def make_predict_fn(code: str, fn_names=("predict", "transform")):
    """Exec LLM-generated code and return the entry-point function.

    Looks for a function named ``predict`` (estimators) or ``transform``
    (the feature engineer), in that order.
    """
    # Use a shared dictionary for globals/locals
    local_vars = {}
    try:
        exec(code, local_vars, local_vars)
    except Exception as e:
        raise ValueError(f"Could not exec LLM code: {e}\nCode was:\n{code}")
    for name in fn_names:
        fn = local_vars.get(name, None)
        if callable(fn):
            return fn
    raise ValueError("No valid function named 'predict' or any callable found in LLM output.")


def _coerce_numeric_strings(features: Dict[str, Any], output_type: Type) -> Dict[str, Any]:
    """Best-effort: turn number-looking strings into int/float, for callers that
    forgot to coerce a semantically-numeric feature themselves. Used only as a
    fallback (see safe_exec_fn) since it can't distinguish a numeric column
    represented as text from a free-text column that merely looks numeric
    (e.g. a song title like "1979")."""
    clean = {}
    for k, v in features.items():
        if isinstance(v, str):
            try:
                # Only convert to float if there's a dot, else int
                if "." in v:
                    f = float(v)
                    clean[k] = int(f) if output_type is int and f.is_integer() else f
                else:
                    clean[k] = int(v)
            except Exception:
                clean[k] = v
        else:
            clean[k] = v
    return clean


def safe_exec_fn(
    fn: Callable,
    features: Dict[str, Any],
    output_type: Type = int,
    default: Any = 0,
    label: str = "PredictFn",
) -> Any:
    """
    Safely executes a function with the given features, coercing output to desired type.

    Features are passed through as-is first -- the generated function is
    prompted to do its own numeric coercion, so an unconditional pre-coercion
    here would corrupt free-text features that happen to look numeric (e.g. a
    song title of "1979" would become the int 1979, breaking any string
    method the generated code calls on it). If the first call raises, retry
    once with number-looking strings coerced to int/float, as a safety net for
    generated code that assumed a semantically-numeric feature would already
    be numeric.
    """
    try:
        res = fn(**features)
        return output_type(res) if res is not None else default
    except Exception:
        pass
    try:
        clean = _coerce_numeric_strings(features, output_type)
        res = fn(**clean)
        return output_type(res) if res is not None else default
    except Exception as e:
        logger.error(f"[{label} ERROR] {e} on features={features}")
        return default


# Typed convenience wrappers around safe_exec_fn.
def safe_predict(fn: Callable, features: dict) -> int:
    return safe_exec_fn(fn, features, output_type=int, default=0, label="PredictFn")


def safe_regress(fn: Callable, features: dict) -> float:
    return safe_exec_fn(fn, features, output_type=float, default=0.0, label="RegressFn")


def parse_tsv(tsv: str) -> pd.DataFrame:
    """Parse tab-separated values (TSV) into a pandas DataFrame."""
    try:
        # Clean common LLM output artifacts
        tsv_cleaned = tsv.strip().replace("```", "").strip()

        # Use StringIO to treat the string like a file
        df = pd.read_csv(StringIO(tsv_cleaned), sep="\t")

        # Optionally: strip whitespace from column names
        df.columns = df.columns.str.strip()

        return df

    except Exception as e:
        raise ValueError(f"Failed to parse TSV output:\n{tsv}\nError: {e}")
