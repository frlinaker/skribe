#!/usr/bin/env python
"""Stress-test: send a massive prompt to OpenAI and Gemini and report what happens.

Builds the largest realistic fit prompt we'd send in production (adult dataset,
full 75% training split, ~5MB CSV) and fires it at a mid-tier model from each
provider.  Reports: accepted / truncated / error, response time, token counts.

Usage:
    python benchmarks/test_large_prompt.py
    python benchmarks/test_large_prompt.py --dataset adult --rows 5000
    python benchmarks/test_large_prompt.py --dataset adult --rows all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

import litellm
import pandas as pd

from benchmark_utils import DEFAULT_DATASETS, load_dataset

MODELS = [
    {"label": "OpenAI GPT-4.1",         "model_id": "gpt-4.1"},
    {"label": "Google Gemini 2.5 Flash", "model_id": "vertex_ai/gemini-2.5-flash",
     "extra": {"vertex_location": "us-central1"}},
]

PREAMBLE = """\
You are an expert Python programmer and machine learning engineer.
Your task is to write a Python function that classifies records from a tabular dataset.

--- DATASET CONTEXT ---
This is the "Adult Income" dataset (also known as Census Income). It contains census data
for ~48,000 individuals. The goal is to predict whether a person earns more than $50,000/year.
Features include age, workclass, education, marital-status, occupation, relationship,
race, sex, capital-gain, capital-loss, hours-per-week, and native-country.

--- INSTRUCTIONS ---
Write a Python function with this exact signature:

    def predict(age, workclass, fnlwgt, education, education_num, marital_status,
                occupation, relationship, race, sex, capital_gain, capital_loss,
                hours_per_week, native_country):
        ...
        return <int>  # 0 = <=50K, 1 = >50K

Rules:
- Return ONLY an int (0 or 1). Never return a string.
- Do not import anything — write pure Python with no dependencies.
- Use the training data below to infer patterns and decision rules.
- You may hard-code lookup tables, thresholds, or rule trees.

--- TRAINING DATA (CSV) ---
{data}
--- END TRAINING DATA ---

Now write the predict() function:
"""


def build_prompt(df: pd.DataFrame) -> str:
    return PREAMBLE.replace("{data}", df.to_csv(index=False))


def count_tokens(model_id: str, prompt: str) -> int:
    try:
        return litellm.token_counter(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        return len(prompt) // 4  # rough fallback


def probe_model(label: str, model_id: str, prompt: str, extra: dict | None = None) -> None:
    extra = extra or {}
    n_chars = len(prompt)
    n_tokens = count_tokens(model_id, prompt)

    print(f"\n{'='*70}")
    print(f"  Model : {label}  ({model_id})")
    print(f"  Prompt: {n_chars:,} chars  ~{n_tokens:,} tokens")
    print(f"{'='*70}")

    t0 = time.time()
    try:
        response = litellm.completion(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            **extra,
        )
        elapsed = time.time() - t0
        content = response.choices[0].message.content or ""
        usage = response.usage

        print(f"  STATUS : OK  ({elapsed:.1f}s)")
        print(f"  USAGE  : prompt_tokens={getattr(usage, 'prompt_tokens', '?')}  "
              f"completion_tokens={getattr(usage, 'completion_tokens', '?')}  "
              f"total={getattr(usage, 'total_tokens', '?')}")
        print(f"  REPLY  : {content[:300].strip()!r}{'...' if len(content) > 300 else ''}")

    except litellm.BadRequestError as e:
        elapsed = time.time() - t0
        print(f"  STATUS : BadRequestError  ({elapsed:.1f}s)")
        print(f"  ERROR  : {str(e)[:500]}")
    except litellm.ContextWindowExceededError as e:
        elapsed = time.time() - t0
        print(f"  STATUS : ContextWindowExceededError  ({elapsed:.1f}s)")
        print(f"  ERROR  : {str(e)[:500]}")
    except litellm.ServiceUnavailableError as e:
        elapsed = time.time() - t0
        print(f"  STATUS : ServiceUnavailableError (503)  ({elapsed:.1f}s)")
        print(f"  ERROR  : {str(e)[:500]}")
    except litellm.RateLimitError as e:
        elapsed = time.time() - t0
        print(f"  STATUS : RateLimitError  ({elapsed:.1f}s)")
        print(f"  ERROR  : {str(e)[:500]}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  STATUS : {type(e).__name__}  ({elapsed:.1f}s)")
        print(f"  ERROR  : {str(e)[:500]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="adult",
                        choices=list(DEFAULT_DATASETS), help="Dataset to use")
    parser.add_argument("--rows", default="all",
                        help="Number of training rows to include (default: all = full dataset)")
    parser.add_argument("--train-frac", type=float, default=0.75,
                        help="Fraction used as training split (default: 0.75)")
    args = parser.parse_args()

    spec = DEFAULT_DATASETS[args.dataset]
    openml_name, version = spec[0], spec[1]
    csv_path = spec[2] if len(spec) > 2 else None
    target_col = spec[3] if len(spec) > 3 else None
    description = spec[4] if len(spec) > 4 else None

    print(f"Loading dataset: {args.dataset} ...")
    X, y, class_map, _ = load_dataset(
        openml_name, version, max_rows=None,
        csv_path=csv_path, target_col=target_col,
        description=description, require_description=False,
    )

    # Reconstruct a labelled dataframe (add target back for the CSV)
    inv_map = {v: k for k, v in class_map.items()}
    df_full = X.copy()
    df_full["label"] = y.map(inv_map)

    # Training split
    n_train = int(len(df_full) * args.train_frac)
    df_train = df_full.iloc[:n_train].copy()

    if args.rows != "all":
        n = int(args.rows)
        df_train = df_train.iloc[:n].copy()

    print(f"Dataset: {args.dataset}  total_rows={len(df_full):,}  "
          f"training_rows_in_prompt={len(df_train):,}  cols={X.shape[1]}")

    prompt = build_prompt(df_train)

    for m in MODELS:
        probe_model(
            label=m["label"],
            model_id=m["model_id"],
            prompt=prompt,
            extra=m.get("extra"),
        )

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
