import logging
import os
import warnings

from typing import Callable, Optional

from .utils import (
    make_predict_fn,
    prepare_training_data,
    extract_python_code,
    parse_tsv,
)

logger = logging.getLogger("promptlearn")


class BasePromptEstimator:
    def __init__(self, model: str, verbose: bool, max_train_rows: int):
        self.model = model
        self.verbose = verbose
        self.max_train_rows = max_train_rows
        self.llm_client = self._init_llm_client()
        self.predict_fn: Optional[Callable] = None
        self.target_name_: Optional[str] = None
        self.feature_names_: Optional[list] = None
        self.raw_python_code_: Optional[str] = None
        self.python_code_: Optional[str] = None

    # used by GridSearchCV
    def get_params(self, deep=True):
        # Only include arguments that are accepted by __init__
        return {
            "model": self.model,
            "verbose": self.verbose,
            "max_train_rows": self.max_train_rows,
        }

    # used by GridSearchCV
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _init_llm_client(self):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "You must install the 'openai' package to use PromptEstimator classes."
            )
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable must be set to use LLM models."
            )
        openai.api_key = api_key
        return openai.OpenAI()

    # used by joblib
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("llm_client", None)  # Remove openai client on serialization
        state.pop("predict_fn", None)  # Remove predict_fn on serialization
        return state

    # used by joblib
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.llm_client = (
            self._init_llm_client()
        )  # Re-initialize LLM client on re-creation of object
        # Now do any additional setup specific to this class
        if getattr(self, "python_code_", None):
            try:
                self.predict_fn = make_predict_fn(self.python_code_)
            except Exception as e:
                warnings.warn(
                    f"Failed to recompile regression function: {e}", UserWarning
                )
                self.predict_fn = None

    def _call_llm(self, prompt: str) -> str:
        """Call the language model, return the code as string."""
        if self.verbose:
            logger.info("[Prompt to LLM]\n%s", prompt)
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            content = str(response.choices[0].message.content).strip()
            if self.verbose:
                logger.info("[LLM Response]\n%s", content)
            return content
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise RuntimeError(f"LLM call failed: {e}")

    def _fit(self, X, y, prompt: str):
        data, self.feature_names_, self.target_name_ = prepare_training_data(X, y)

        # Use a small sample for LLM to avoid expensive calls
        if len(data) > self.max_train_rows:
            logger.info(
                f"Reducing training data from {data.shape[0]:,} to {self.max_train_rows:,} rows for LLM."
            )
            sample_df = data.sample(self.max_train_rows, random_state=42)
        else:
            sample_df = data

        csv_data = sample_df.to_csv(index=False)

        prompt = prompt.format(data=csv_data)
        logger.info(f"[LLM Prompt]\n{prompt}")

        # Call LLM and get code
        code = self._call_llm(prompt)
        if not isinstance(code, str):
            code = str(code)
        logger.info(f"[LLM Output]\n{code}")

        # Remove markdown/code block if present (triple backticks)
        code = extract_python_code(code)
        if not code.strip():
            logger.error("LLM output is empty after removing markdown/code block.")
            raise ValueError("No code to exec from LLM output.")

        print(f"the cleaned up code is: [START]{code}[END]")
        self.raw_python_code_ = code

        extended_code = self._extend_code(code)
        self.python_code_ = extended_code
        print(f"the extended code is: [START]{extended_code}[END]")

        # Compile the code into a function
        self.predict_fn = make_predict_fn(extended_code)

        return self

    def _extend_code(self, code: str) -> str:
        logger.info("[Post-Process] Expanding code via second LLM pass...")
        refinement_prompt = (
            "The following function may use a dictionary, set, or mapping based on domain knowledge (e.g., country names, animal types).\n"
            "Please re-write the function to extend any such mappings with many more possible real-world keys, if applicable.\n"
            "Try to figure out the logic of the function based on the variable names and values that are processed in the function.\n"
            "Avoid changing the logic or structure beyond extending categorical support.\n"
            "Only return valid Python code.\n\n"
            f"{code}"
        )
        try:
            refined_code = self._call_llm(refinement_prompt)
            refined_code = extract_python_code(str(refined_code))
            logger.info("[Post-Process] Successfully extended function.")
            return refined_code
        except Exception as e:
            logger.warning(f"[Post-Process] Skipping refinement: {e}")
        return code  # fallback: pass back the original code

    def sample(self, n: int = 5):
        """Generate n synthetic examples that illustrate the heuristic."""
        # Check that columns have some sort of names
        if (
            not hasattr(self, "feature_names_")
            or self.feature_names_ is None
            or not hasattr(self, "target_name_")
            or self.target_name_ is None
        ):
            raise RuntimeError(
                "Call fit() before sample(): feature names or target name not set."
            )
        prompt = (
            f"{self.python_code_}\n\n"
            f"Please generate {n} example rows in tabular format with the following columns:\n"
            f"{', '.join(self.feature_names_ + [self.target_name_])}.\n"
            f"Use tab-separated format. Do not explain."
        )
        text = self._call_llm(prompt)
        return parse_tsv(text)
