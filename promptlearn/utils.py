import logging
import re

from io import StringIO

import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, Type

logger = logging.getLogger("promptlearn")


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
        target_name = normalize_feature_name(
            y.name if hasattr(y, "name") and y.name else "target"
        )
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


def make_predict_fn(code: str):
    # Use a shared dictionary for globals/locals
    local_vars = {}
    try:
        exec(code, local_vars, local_vars)
    except Exception as e:
        raise ValueError(f"Could not exec LLM code: {e}\nCode was:\n{code}")
    # Look for 'predict' function
    fn = local_vars.get("predict", None)
    if not callable(fn):
        raise ValueError(
            "No valid function named 'predict' or any callable found in LLM output."
        )
    return fn


def safe_exec_fn(
    fn: Callable,
    features: Dict[str, Any],
    output_type: Type = int,
    default: Any = 0,
    label: str = "PredictFn",
) -> Any:
    """
    Safely executes a function with cleaned features, coercing output to desired type.
    """
    clean = {}
    for k, v in features.items():
        if v is None:
            clean[k] = v
            continue
        if isinstance(v, (float, int)):
            clean[k] = v
        elif isinstance(v, str):
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
    try:
        res = fn(**clean)
        return output_type(res) if res is not None else default
    except Exception as e:
        logger.error(f"[{label} ERROR] {e} on features={features}")
        return default


# For compatibility with previous usage:
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
