import logging
import os
import warnings
import joblib
from typing import Any, Dict, Optional
import re

import openai

logger = logging.getLogger("promptlearn")

class BasePromptEstimator:
    def __init__(self, model: str, prompt_template: str, verbose: bool = False):
        self.model = model
        self.prompt_template = prompt_template
        self.verbose = verbose
        self.heuristic_ = None
        self.heuristic_history_ = []
        self.aux_data_ = {}

        self._init_llm_client()

    def _init_llm_client(self):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            openai.api_key = api_key
            self.llm_client = openai.OpenAI()
        except ImportError:
            warnings.warn("openai package is not installed; llm_client not initialized", RuntimeWarning)
            self.llm_client = None
        except Exception as e:
            warnings.warn(f"Failed to initialize llm_client: {e}", RuntimeWarning)
            self.llm_client = None

    def save_clean(self, path: str):
        if hasattr(self, "llm_client"):
            del self.llm_client
        joblib.dump(self, path)
        if self.verbose:
            logger.info(f"Saved clean model to {path}")

    def safe_format(self, template, **kwargs):
        from collections import defaultdict
        class SafeDict(defaultdict):
            def __missing__(self, key):
                return ""
        return template.format_map(SafeDict(str, kwargs))

    def _call_llm(self, prompt: str) -> str:
        if self.llm_client is None:
            raise RuntimeError("llm_client not initialized or OPENAI_API_KEY not set.")

        if self.verbose:
            logger.info(f"[LLM Prompt]\n{prompt}")

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}]
        )
        # For OpenAI v1, choices is a list, and code is .message.content
        code = response.choices[0].message.content.strip()

        if self.verbose:
            logger.info(f"[LLM Output]\n{code}")

        return code

    def _make_predict_fn(self, code: str, aux_data: dict = None):
        aux_data = aux_data or {}
        local_env = {}
        # Remove markdown/code-fence wrappers if present
        code = code.strip()
        # Strip code block markers if present
        if code.startswith("```"):
            code = re.sub(r"^```(python)?", "", code)
            code = code.rstrip("`").rstrip()
        code = re.sub(r"^python\s+", "", code, flags=re.MULTILINE)
        if not code:
            logger.error("LLM output is empty after removing markdown/code block.")
            raise ValueError("No code to exec from LLM output.")

        try:
            # PATCH: Use same dict for globals/locals so all definitions share scope!
            exec(code, local_env, local_env)
            func = local_env.get('predict') or local_env.get('regress') \
                or next((v for v in local_env.values() if callable(v)), None)
            if not func:
                raise ValueError("No valid function named 'predict', 'regress', or any callable found in LLM output.")
        except Exception as e:
            logger.error(f"[MakePredictFn ERROR] Could not exec LLM code: {e}\nCODE WAS:\n{code}")
            raise

        def safe_predict_fn(features: dict):
            try:
                return func(**{**features, **aux_data})
            except Exception as e:
                logger.error(f"[PredictFn ERROR] {e} on features={features}")
                raise
        return safe_predict_fn

    def __getstate__(self):
        state = self.__dict__.copy()
        if "llm_client" in state:
            del state["llm_client"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_llm_client()

    def show_heuristic_evolution(self):
        print("🧠 Heuristic Evolution:\n")
        for i, h in enumerate(self.heuristic_history_):
            print(f"--- After chunk {i + 1} ---")
            print(h.strip() if isinstance(h, str) else str(h))
            print()
