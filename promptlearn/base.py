import logging
import os
import re

# Helper for robust Python identifier normalization
def normalize_feature_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    name = re.sub(r"__+", "_", name)
    return name.strip("_").lower()

class BasePromptEstimator:
    def __init__(self, model: str = "o4-mini", verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.llm_client = self._init_llm_client()
        self._log = logging.getLogger("promptlearn")
        if not self._log.hasHandlers():
            logging.basicConfig(level=logging.INFO)

    def _init_llm_client(self):
        try:
            import openai
        except ImportError:
            raise ImportError("You must install the 'openai' package to use PromptEstimator classes.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable must be set to use LLM models.")
        openai.api_key = api_key
        return openai.OpenAI()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove client on serialization
        state.pop("llm_client", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.llm_client = self._init_llm_client()

    def _call_llm(self, prompt: str) -> str:
        """Call the language model, return the code as string."""
        if self.verbose:
            self._log.info("[Prompt to LLM]\n%s", prompt)
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = str(response.choices[0].message.content).strip()
            if self.verbose:
                self._log.info("[LLM Response]\n%s", content)
            return content
        except Exception as e:
            self._log.error("LLM call failed: %s", e)
            raise RuntimeError(f"LLM call failed: {e}")
