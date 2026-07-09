import ast
import builtins
import difflib
import logging
import os
import re
import time
import traceback
import warnings
from typing import Callable, Optional

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from .explain import Explanation
from .prompt_markers import CONTEXT_END, CONTEXT_START, DATA_MARKER
from .utils import (
    extract_python_code,
    generate_feature_dicts,
    make_predict_fn,
    parse_tsv,
    prepare_training_data,
    sanitize_dataset_description,
)

logger = logging.getLogger("skribe")

# The library default model. Used when no model is passed and the
# SKRIBE_MODEL environment variable is unset.
DEFAULT_MODEL = "gpt-5.5"

# Fraction of the model's max_input_tokens budget used for the prompt.
# Start high to maximise training rows; shrink by this step each time the
# model signals finish_reason="length" (output was cut off mid-generation).
_CONTEXT_HEADROOM = 0.92
_CONTEXT_HEADROOM_STEP = 0.05


class _OutputTruncated(Exception):
    """Raised when the LLM signals finish_reason='length' (output cut off)."""


class _ContextWindowExceeded(Exception):
    """Raised when the API rejects the prompt as exceeding the model's real
    input token limit. Carries that limit (parsed from the provider's error
    message) so callers can correct a wrong/stale value from
    litellm.get_model_info() instead of guessing via headroom shrink steps.
    """

    def __init__(self, message: str, real_max_input_tokens: Optional[int] = None):
        super().__init__(message)
        self.real_max_input_tokens = real_max_input_tokens


# Matches known provider context-window error phrasings, e.g.:
#   "Input tokens exceed the configured limit of 272000 tokens."           (Responses API)
#   "This model's maximum context length is 128000 tokens."               (Chat Completions)
_CONTEXT_LIMIT_RE = re.compile(
    r"(?:configured limit of|maximum context length is) (\d[\d,]*)\s*tokens",
    re.IGNORECASE,
)


def _format_error_with_suggestion(e: Exception) -> str:
    """str(e) drops Python's own "Did you mean: 'x'?" suggestion for typo'd
    names (NameError/AttributeError) -- that suggestion is only computed by
    the traceback formatter, which inspects the frame at the point of
    failure. Retry feedback is far more actionable with it: telling the LLM
    "name 'ea' is not defined. Did you mean: 'ra'?" points straight at the
    fix, instead of just "name 'ea' is not defined."

    Also appends the runtime types of the failing call's arguments when the
    innermost frame is a generated predict/transform function (recognized
    by its universal ``features`` kwarg-dict local). This targets the class
    of bug behind "retry loop fails to converge within 3 attempts"
    (e.g. AttributeError: 'int' object has no attribute 'lower' on a
    numeric-looking column pandas parsed as int) -- the bare message names
    the wrong type but not which feature carried it, forcing the LLM to
    guess across every column instead of fixing the one that's actually
    mistyped.
    """
    if e.__traceback__ is None:
        return str(e)
    # format_exception_only alone doesn't include the suggestion -- only the
    # full traceback (which does frame introspection) computes it.
    full = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    lines = full.strip().splitlines()
    message = lines[-1] if lines else str(e)

    tb = e.__traceback__
    while tb.tb_next is not None:
        tb = tb.tb_next
    features = tb.tb_frame.f_locals.get("features")
    if isinstance(features, dict) and isinstance(e, (AttributeError, TypeError)):
        args_repr = ", ".join(f"{k}={v!r} ({type(v).__name__})" for k, v in features.items())
        message += f"\nArguments passed to predict(): {args_repr}"
    return message


def _check_unresolved_names(code: str) -> None:
    """Statically flag references to names that are never bound anywhere in
    the generated ``predict``/``transform`` function, e.g. a typo'd variable
    used only inside a branch that a validation row never happens to take.

    ``_validate_predict_fn`` only exercises the code paths its sample rows
    reach, so a NameError inside an untaken if/elif branch would otherwise
    pass fit-time validation and only surface later as a silent
    ``safe_predict``/``safe_regress`` fallback in production. This walks the
    AST instead of executing it, so it catches the typo regardless of which
    branch the sample rows happen to exercise.

    Mirrors the audit's ``check_signature_mismatch`` heuristic: nested
    ``def``/lambda params, comprehension/for-loop targets, ``with ... as``,
    ``except ... as``, and walrus targets all count as bound names, so
    legitimate nested helper functions don't false-positive.
    """
    tree = ast.parse(code)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in ("predict", "transform")
    ]
    for func in functions:
        if func.args.kwarg is None and not func.args.args:
            continue
        declared = {a.arg for a in func.args.args}
        bound = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        bound.add(t.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(node.name)
                bound |= {a.arg for a in node.args.args}
                if node.args.vararg:
                    bound.add(node.args.vararg.arg)
                if node.args.kwarg:
                    bound.add(node.args.kwarg.arg)
            elif isinstance(node, ast.Lambda):
                bound |= {a.arg for a in node.args.args}
            elif isinstance(node, (ast.For, ast.comprehension)):
                for n in ast.walk(node.target):
                    if isinstance(n, ast.Name):
                        bound.add(n.id)
            elif isinstance(node, ast.withitem) and node.optional_vars:
                for n in ast.walk(node.optional_vars):
                    if isinstance(n, ast.Name):
                        bound.add(n.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
        loaded = {
            n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        known = declared | bound | set(dir(builtins)) | {"self"}
        unknown = sorted(n for n in loaded - known if not n.startswith("__"))
        if unknown:
            name = unknown[0]
            candidates = difflib.get_close_matches(name, known, n=1)
            suggestion = f" Did you mean: '{candidates[0]}'?" if candidates else ""
            raise NameError(f"name '{name}' is not defined.{suggestion}")


def resolve_model(model: Optional[str]) -> str:
    """Resolve the model string for an estimator.

    An explicit ``model`` always wins. Otherwise the ``SKRIBE_MODEL``
    environment variable is used when set (handy for pointing tests/CI at a
    cheaper, faster model), falling back to :data:`DEFAULT_MODEL`.
    """
    if model is not None:
        return model
    return os.environ.get("SKRIBE_MODEL", DEFAULT_MODEL)


class BaseSkribeEstimator(BaseEstimator):
    def __init__(
        self,
        model: str,
        verbose: bool,
        max_train_rows: Optional[int],
        max_retries: int = 2,
        web_search: bool = False,
        context_prepass: bool = True,
        vertex_location: Optional[str] = None,
        llm_timeout: float = 120,
        reasoning_effort: Optional[str] = None,
    ):
        self.model = model
        self.verbose = verbose
        self.max_train_rows = max_train_rows
        self.max_retries = max_retries
        self.web_search = web_search
        self.context_prepass = context_prepass
        self.vertex_location = vertex_location
        self.llm_timeout = llm_timeout
        self.reasoning_effort = reasoning_effort
        self.predict_fn: Optional[Callable] = None
        self.target_name_: Optional[str] = None
        self.feature_names_: Optional[list] = None
        self.raw_python_code_: Optional[str] = None
        self.python_code_: Optional[str] = None
        self.explanation_: Optional[Explanation] = None
        self.context_summary_: Optional[str] = None
        self.context_prepass_prompt_: Optional[str] = None
        self.fit_log_: list = []

    # used by GridSearchCV
    def get_params(self, deep=True):
        # Only include arguments that are accepted by __init__
        return {
            "model": self.model,
            "verbose": self.verbose,
            "max_train_rows": self.max_train_rows,
            "max_retries": self.max_retries,
            "web_search": self.web_search,
            "context_prepass": self.context_prepass,
            "vertex_location": self.vertex_location,
            "llm_timeout": self.llm_timeout,
            "reasoning_effort": self.reasoning_effort,
        }

    # used by GridSearchCV
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # used by joblib
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("predict_fn", None)  # Remove predict_fn on serialization
        return state

    # used by joblib
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recompile the generated heuristic on re-creation of the object
        if getattr(self, "python_code_", None):
            try:
                self.predict_fn = make_predict_fn(self.python_code_)
            except Exception as e:
                warnings.warn(f"Failed to recompile regression function: {e}", UserWarning)
                self.predict_fn = None

    # Models that use the Responses API for web search (tools=[{"type": "web_search"}]).
    # GPT-5+ models on OpenAI use the Responses API; web_search_options is NOT supported.
    _WEB_SEARCH_RESPONSES_API_MODELS = {
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-5.5",
        "gpt-5.4-mini",
        "gpt-5.4",
        "gpt-5.3",
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
    }

    # Models that support web search via web_search_options in the Chat Completions API.
    # OpenAI: dedicated search-preview models.
    # Google: Gemini models via Vertex AI (Google Search grounding).
    _WEB_SEARCH_CHAT_COMPLETIONS_MODELS = {
        "gpt-4o-search-preview",
        "gpt-4o-mini-search-preview",
        "vertex_ai/gemini-2.5-pro",
        "vertex_ai/gemini-2.5-flash",
        "vertex_ai/gemini-2.5-flash-lite",
        "vertex_ai/gemini-3.5-flash",
    }

    # Union for external checks (e.g. warnings when model is unsupported).
    _WEB_SEARCH_MODELS = _WEB_SEARCH_RESPONSES_API_MODELS | _WEB_SEARCH_CHAT_COMPLETIONS_MODELS

    def _record_web_search_evidence(
        self, search_call_count: Optional[int], citations: list
    ) -> None:
        """Log whether a web-search-enabled call actually searched anything.

        Without this, ``web_search=True`` is a black box — there was no way to
        tell whether the model searched at all or what it found. Recorded
        unconditionally (even when empty) so "requested search but found zero
        citations" is visible too, not just silently indistinguishable from
        "never called with web_search=True".
        """
        entry: dict = {"stage": "web_search"}
        if search_call_count is not None:
            entry["search_call_count"] = search_call_count
        entry["citations"] = citations
        self.fit_log_.append(entry)

    def _call_llm(
        self,
        prompt: str,
        web_search: bool = False,
        reasoning_effort: Optional[str] = None,
        web_search_config: Optional[dict] = None,
    ) -> str:
        """Call the language model via litellm, return the response text.

        The provider is selected by the model string, e.g. ``gpt-5.5`` (OpenAI),
        ``claude-sonnet-4-6`` (Anthropic), or ``ollama:llama3.1`` (local Ollama).
        API keys are read from the usual per-provider environment variables.

        ``reasoning_effort`` (``"low"``/``"medium"``/``"high"``/etc, provider-
        dependent) defaults to ``self.reasoning_effort`` when not overridden by
        the caller — passed through to litellm uniformly; unsupported models
        simply ignore it (litellm no-ops rather than erroring).

        ``web_search_config`` merges extra keys into the Responses API's
        ``web_search`` tool dict (e.g. ``search_context_size``, ``filters``)
        for the OpenAI-only Responses API path — Gemini's grounding tool has
        no equivalent knobs, so this is a no-op there.
        """
        import litellm

        if reasoning_effort is None:
            reasoning_effort = self.reasoning_effort

        if self.verbose:
            logger.info("[Prompt to LLM]\n%s", prompt)
        # Accept the documented ``ollama:model`` syntax; litellm expects ``ollama/model``.
        model = self.model
        if model.startswith("ollama:"):
            model = "ollama/" + model[len("ollama:") :]

        if web_search and model in self._WEB_SEARCH_RESPONSES_API_MODELS:
            # GPT-5+ uses the Responses API with tools=[{"type": "web_search"}].
            responses_kwargs: dict = {}
            if reasoning_effort is not None:
                responses_kwargs["reasoning_effort"] = reasoning_effort
            web_search_tool = {"type": "web_search", **(web_search_config or {})}
            response = litellm.responses(
                prompt,
                model,
                tools=[web_search_tool],
                timeout=self.llm_timeout,
                **responses_kwargs,
            )
            content = ""
            search_call_count = 0
            citations: list = []
            for item in response.output:
                item_type = getattr(item, "type", None)
                if item_type == "web_search_call":
                    search_call_count += 1
                elif item_type == "message":
                    for c in item.content:
                        if getattr(c, "type", None) == "output_text":
                            content += c.text
                            for ann in getattr(c, "annotations", None) or []:
                                url = getattr(ann, "url", None)
                                if url:
                                    citations.append(url)
            self._record_web_search_evidence(search_call_count, citations)
            content = content.strip()
            if self.verbose:
                logger.info("[LLM Response]\n%s", content)
            return content

        kwargs: dict = {}
        if web_search:
            if model in self._WEB_SEARCH_CHAT_COMPLETIONS_MODELS:
                kwargs["web_search_options"] = {}
            else:
                logger.warning(
                    "web_search=True requested but model %r is not in the known "
                    "supported list %s — proceeding without web search.",
                    model,
                    self._WEB_SEARCH_MODELS,
                )
        if self.vertex_location:
            kwargs["vertex_location"] = self.vertex_location
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.llm_timeout,
                **kwargs,
            )
            choice = response.choices[0]
            content = str(choice.message.content).strip()
            finish_reason = getattr(choice, "finish_reason", None)
            if kwargs.get("web_search_options") is not None:
                citations = [
                    ann.get("url_citation", {}).get("url")
                    for ann in getattr(choice.message, "annotations", None) or []
                    if ann.get("url_citation", {}).get("url")
                ]
                self._record_web_search_evidence(None, citations)
            if self.verbose:
                logger.info("[LLM Response (finish_reason=%s)]\n%s", finish_reason, content)
            if finish_reason == "length":
                logger.warning(
                    "LLM output was truncated (finish_reason='length') — "
                    "response may be incomplete."
                )
                raise _OutputTruncated(content)
            return content
        except _OutputTruncated:
            raise
        except litellm.ContextWindowExceededError as e:
            # Local token counter (or litellm's get_model_info) undercounted or
            # is stale — the provider's error message carries the real limit,
            # so parse it out rather than guessing via headroom shrink steps.
            match = _CONTEXT_LIMIT_RE.search(str(e))
            real_max_input_tokens = int(match.group(1).replace(",", "")) if match else None
            logger.warning(
                "Context window exceeded at API level (local token count was inaccurate) — "
                "will reduce dataset and retry%s. Error: %s",
                (
                    f" using real limit {real_max_input_tokens:,} tokens"
                    if real_max_input_tokens
                    else ""
                ),
                e,
            )
            raise _ContextWindowExceeded(str(e), real_max_input_tokens)
        except litellm.RateLimitError as e:
            logger.warning("Rate limit hit — sleeping 60s before re-raising. Error: %s", e)
            self.fit_log_.append({"stage": "llm_call", "error": f"RateLimitError: {e}"})
            time.sleep(60)
            raise RuntimeError(f"LLM call failed: {e}")
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            self.fit_log_.append({"stage": "llm_call", "error": str(e)})
            raise RuntimeError(f"LLM call failed: {e}")

    def _build_dataset_context(
        self,
        dataset_description: str,
        feature_names: list,
        sample_df,
        target_name: str,
        label_names: Optional[dict] = None,
    ) -> str:
        """Run a pre-pass LLM call to produce a clean, structured dataset context block.

        Takes the raw description, column names, and a small sample of values and
        asks the LLM to:
          - Strip boilerplate, citations, and metadata noise
          - Explain what each column measures (decode acronyms, expand short names)
          - Map known categorical short-values to their meanings (e.g. 'f'/'t', codes)
          - Note the prediction target and what its values mean
          - Output a compact, self-contained paragraph or table — no code

        If web_search is True the same flag is forwarded so the model can look up
        domain schemas (e.g. UCI attribute glossaries).

        ``label_names`` maps each integer class code actually present in
        ``sample_df[target_name]`` to its true original label (e.g.
        ``{0: "bird", 1: "fish", 2: "mammal"}``). When given, the target-value
        line states the real labels instead of the bare training codes — the
        target column itself has already been integer-encoded for training by
        the time this runs, so without ``label_names`` there would be nothing
        but bare ints to go on and the LLM would have to guess what they mean.
        """

        # Build a value-summary: unique values per column, capped so a single
        # high-cardinality or near-continuous column (e.g. free-text track
        # names, raw timestamps) can't blow up the prompt past the model's
        # context window — this is independent of row count, since such
        # columns keep most of their distinct values even after sampling.
        _MAX_UNIQUE_PREVIEW = 20
        value_lines = []
        for col in feature_names:
            uniq = sample_df[col].dropna().unique()
            preview = sorted(str(v) for v in uniq)
            if len(preview) > _MAX_UNIQUE_PREVIEW:
                shown = preview[:_MAX_UNIQUE_PREVIEW]
                value_lines.append(
                    f"  {col}: {', '.join(shown)}, "
                    f"... and {len(preview) - _MAX_UNIQUE_PREVIEW:,} more unique values"
                )
            else:
                value_lines.append(f"  {col}: {', '.join(preview)}")
        value_summary = "\n".join(value_lines)

        target_codes = sample_df[target_name].dropna().unique()
        if label_names:
            target_uniq = sorted(str(label_names.get(code, code)) for code in target_codes)
        else:
            target_uniq = sorted(str(v) for v in target_codes)

        prompt = (
            "You are preparing a structured dataset summary that will be embedded in a "
            "machine-learning prompt. Your output will be read by an LLM that will write "
            "a Python predict() function — so precision and brevity matter more than prose.\n\n"
            "Given the raw dataset description and column information below, produce a clean "
            "context block that:\n"
            "1. States in one sentence what the dataset predicts and what each target value means.\n"
            "2. For every column, explains what it measures. Decode any abbreviations or acronyms. "
            "If values are short codes (e.g. 'f'/'t', single letters, integers used as categories), "
            "map them to their real meaning.\n"
            "3. Strips all boilerplate: citations, donor info, file format descriptions, "
            "download notices, and instance/attribute counts.\n"
            "4. Is compact: aim for under 400 words. For columns, use plain lines in the format "
            "'- column_name: description' — no markdown bold, headers, or tables.\n"
            "--- Raw dataset description ---\n"
            f"{dataset_description.strip()}\n"
            "--- End raw description ---\n\n"
            f"Target column: {target_name}\n"
            f"Target values seen: {', '.join(target_uniq)}\n\n"
            "Feature columns and their observed values:\n"
            f"{value_summary}\n\n"
            "Output only the plain-text context block. No markdown of any kind (no **, no #, "
            "no tables, no fences). This text will be embedded verbatim in a downstream prompt."
        )

        self.context_prepass_prompt_ = prompt
        logger.info("[Context pre-pass] Calling LLM to summarize dataset context...")
        try:
            # This call benefits most from a thorough search (finding a
            # dataset's real documentation/schema) more than the code-gen or
            # extend calls do, which mostly need a quick fact lookup -- so it
            # asks for the OpenAI Responses API's highest search-context tier.
            # No equivalent knob exists for Gemini's grounding tool.
            result = self._call_llm(
                prompt,
                web_search=self.web_search,
                web_search_config={"search_context_size": "high"},
            )
            result = result.strip()
            logger.info("[Context pre-pass] Result:\n%s", result)
            return result
        except Exception as e:
            logger.warning("[Context pre-pass] Failed (%s); falling back to raw description.", e)
            return sanitize_dataset_description(dataset_description)

    def _build_prompt_without_data(
        self,
        prompt: str,
        synthetic_features: Optional[list],
        context_block: Optional[str],
        label_names: Optional[dict] = None,
        target_name: Optional[str] = None,
        majority_class: Optional[float] = None,
    ) -> str:
        """Build the prompt with {data} still present as a placeholder.

        ``context_block`` is the already-processed dataset context string (either
        the output of ``_build_dataset_context`` or a sanitized raw description).

        ``label_names`` (target column code -> true label) is stated here
        verbatim and unconditionally — not just fed into the context pre-pass
        — because the pre-pass is a second LLM call summarizing in free text;
        it may drop or reorder the explicit code->label correspondence even
        when it correctly lists the label names. The training data CSV itself
        only ever contains the bare integer codes, so without this line the
        code-generation LLM has nothing authoritative to tie e.g. `1` back to
        `bird` and has to guess (see test_fit_prompt_states_label_mapping).
        """
        if synthetic_features:
            synthetic_note = (
                f"\nNote: the following columns are SYNTHETIC features pre-computed by a "
                f"feature engineering step: {', '.join(synthetic_features)}. "
                "Treat them as weak hints only — do NOT build your primary logic around them. "
                "Base your prediction mainly on the original (non-synthetic) columns.\n"
            )
            prompt = synthetic_note + prompt

        # Only worth stating when a code actually differs from its own label
        # (e.g. string labels encoded to ints) — skip the no-op case where y
        # was already plain ints 0..n-1, to avoid adding a noisy, redundant
        # section to every fit() call regardless of dataset_description.
        needs_mapping_line = label_names and any(
            str(code) != str(label) for code, label in label_names.items()
        )
        if needs_mapping_line:
            # Quote the original label and keep the training code bare, so a
            # label that happens to itself look like a small integer (e.g.
            # OpenML datasets that store category codes as the strings '1',
            # '2') can never be visually confused with the training code —
            # otherwise the LLM tends to write `return 1` / `return 2` (the
            # quoted-looking label) instead of `return 0` / `return 1` (the
            # actual code), which silently inverts/scrambles every prediction.
            mapping_str = ", ".join(
                f'{code}="{label}"' for code, label in sorted(label_names.items())
            )
            mapping_line = (
                f"The {target_name or 'target'} column in the training data below has been "
                f"encoded to integer training codes 0..{len(label_names) - 1}. The original "
                f"dataset label for each code (for context only — your function must still "
                f"return the training code, never the original label) is: {mapping_str}."
            )
            context_block = f"{mapping_line}\n\n{context_block}" if context_block else mapping_line

        if majority_class is not None:
            # The template's generic "fallback such as 0" wording steers the
            # LLM toward always defaulting to code 0 / value 0.0 regardless
            # of what's actually common — 0 is just whichever class sorts
            # first (or a fixed constant for regression), not necessarily
            # representative. A function whose primary logic is a set of
            # narrow/memorized branches executes its fallback on every
            # unmatched input, so a bad fallback choice can dominate
            # accuracy even when the matched-branch logic is otherwise fine.
            if label_names is not None:
                majority_code = int(majority_class)
                majority_label = label_names.get(majority_code, majority_code)
                label_note = (
                    f' (original label "{majority_label}")'
                    if str(majority_label) != str(majority_code)
                    else ""
                )
                fallback_line = (
                    f"If your function has a final fallback/default case for when no "
                    f"other rule matches, use training code {majority_code}{label_note} — "
                    f"this is the most common class in the training data, not "
                    f"necessarily code 0."
                )
            else:
                fallback_line = (
                    f"If your function has a final fallback/default case for when no "
                    f"other rule matches, use {majority_class!r} — this is a "
                    f"representative typical value (median) of the training target, "
                    f"not necessarily 0.0."
                )
            context_block = (
                f"{fallback_line}\n\n{context_block}" if context_block else fallback_line
            )

        if context_block:
            context_section = f"{CONTEXT_START}\n{context_block}\n{CONTEXT_END}\n\n"
            data_marker_line = DATA_MARKER + "\n"
            if data_marker_line in prompt:
                prompt = prompt.replace(data_marker_line, context_section + data_marker_line, 1)
            else:
                prompt = prompt + "\n" + context_section

        if self.web_search:
            prompt = (
                "You may search the web to look up information about these features "
                "and their real-world relationships before writing the predict function — "
                "e.g. the dataset's own documentation or schema (UCI, Kaggle, OpenML), or "
                "an authoritative reference for what a feature's codes/categories mean. "
                "If search results conflict with an explicit code-to-label mapping already "
                "stated in this prompt, the mapping stated here always wins — never let a "
                "recalled or searched schema override it.\n\n"
            ) + prompt

        return prompt

    def _count_tokens(self, prompt: str) -> int:
        """Count tokens for the given prompt string using the most accurate method available.

        For Gemini/Vertex AI models, calls Google's countTokens API which uses the
        exact same tokenizer as inference. For all other models, falls back to
        litellm.token_counter (tiktoken-based).
        """
        import litellm

        messages = [{"role": "user", "content": prompt}]

        # Gemini models: use Google's exact countTokens API (free, same tokenizer as inference).
        # litellm.token_counter falls back to tiktoken for Gemini which is ~20-30% off.
        if "gemini" in self.model.lower():
            try:
                import os

                from google import genai

                project = os.environ.get("VERTEXAI_PROJECT") or os.environ.get(
                    "GOOGLE_CLOUD_PROJECT"
                )
                location = (
                    self.vertex_location or os.environ.get("VERTEXAI_LOCATION") or "us-central1"
                )
                # Extract bare model name (strip "vertex_ai/" prefix if present)
                model_name = self.model.split("/")[-1].replace("+web", "")
                client = genai.Client(vertexai=True, project=project, location=location)
                resp = client.models.count_tokens(model=model_name, contents=prompt)
                return resp.total_tokens
            except Exception as e:
                logger.warning(
                    "Google countTokens API failed (%s) — falling back to litellm.token_counter.", e
                )

        return litellm.token_counter(model=self.model, messages=messages)

    def _truncate_to_context_window(
        self,
        df,
        prompt_template: str,
        headroom: float = _CONTEXT_HEADROOM,
        max_input_override: Optional[int] = None,
    ) -> "pd.DataFrame":
        """Return df truncated so the full prompt fits within the model's context window.

        Builds the prompt with all rows, counts tokens, and removes rows from the
        bottom until it fits. Warns once if any truncation occurs. If the model's
        context window is unknown, returns df unchanged with a warning.

        ``max_input_override`` takes precedence over litellm.get_model_info() —
        used once the API has told us its real limit, since that registry value
        can be stale or wrong (see _ContextWindowExceeded).
        """
        import litellm

        if max_input_override:
            max_input = max_input_override
        else:
            try:
                info = litellm.get_model_info(self.model)
                max_input = info.get("max_input_tokens")
                if not max_input:
                    raise ValueError("max_input_tokens not available")
            except Exception as e:
                logger.warning(
                    "Could not determine context window for model %r (%s) — "
                    "skipping token-limit check.",
                    self.model,
                    e,
                )
                return df

        budget = int(max_input * headroom)

        csv = df.to_csv(index=False)
        full_prompt = prompt_template.replace("{data}", csv)
        n_tokens = self._count_tokens(full_prompt)

        if n_tokens <= budget:
            return df

        # Estimate target row count linearly then do at most one binary-search
        # correction, using _count_tokens throughout for accuracy.
        original_rows = len(df)
        # Linear estimate: scale rows proportionally to budget/n_tokens ratio.
        estimated_rows = int(original_rows * budget / n_tokens)
        # Binary-search from there to find the exact largest row count that fits.
        lo, hi = max(1, estimated_rows - 500), min(original_rows, estimated_rows + 500)
        # Expand bounds if estimate was off.
        while lo > 1:
            csv = df.iloc[:lo].to_csv(index=False)
            if self._count_tokens(prompt_template.replace("{data}", csv)) <= budget:
                break
            lo = max(1, lo - 500)
            hi = lo + 500
        while hi < original_rows:
            csv = df.iloc[:hi].to_csv(index=False)
            if self._count_tokens(prompt_template.replace("{data}", csv)) > budget:
                break
            hi = min(original_rows, hi + 500)

        while lo < hi:
            mid = (lo + hi + 1) // 2
            csv = df.iloc[:mid].to_csv(index=False)
            if self._count_tokens(prompt_template.replace("{data}", csv)) <= budget:
                lo = mid
            else:
                hi = mid - 1

        kept = lo
        warnings.warn(
            f"Training data ({original_rows:,} rows, ~{n_tokens:,} tokens) exceeds "
            f"{headroom:.0%} of {self.model!r} context window "
            f"({max_input:,} tokens). Truncating to {kept:,} rows.",
            UserWarning,
            stacklevel=4,
        )
        logger.warning(
            "Truncating training data from %d to %d rows to fit context window.",
            original_rows,
            kept,
        )
        return df.iloc[:kept].copy()

    def _fit(
        self,
        X,
        y,
        prompt: str,
        synthetic_features: Optional[list] = None,
        dataset_description: Optional[str] = None,
        label_names: Optional[dict] = None,
        majority_class: Optional[float] = None,
    ):
        data, self.feature_names_, self.target_name_ = prepare_training_data(X, y)
        self.explanation_ = None  # invalidate any cached explanation from a prior fit
        self.context_summary_ = None
        self.context_prepass_prompt_ = None
        self.fit_log_ = []

        if self.max_train_rows is not None and len(data) > self.max_train_rows:
            logger.info(
                "Reducing training data from %d to %d rows (max_train_rows).",
                len(data),
                self.max_train_rows,
            )
            data = data.sample(self.max_train_rows, random_state=42)
        else:
            # Shuffle so that context-window truncation sees a representative
            # class distribution rather than whatever ordering the source data uses
            # (OpenML datasets are often sorted by class label).
            data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Context pre-pass: replace raw description with a clean, structured summary.
        if dataset_description and self.context_prepass:
            self.context_summary_ = self._build_dataset_context(
                dataset_description,
                self.feature_names_,
                data,
                self.target_name_,
                label_names=label_names,
            )
            context_block = self.context_summary_
        elif dataset_description:
            context_block = sanitize_dataset_description(dataset_description)
        else:
            context_block = None

        prompt_template = self._build_prompt_without_data(
            prompt,
            synthetic_features,
            context_block,
            label_names=label_names,
            target_name=self.target_name_,
            majority_class=majority_class,
        )

        headroom = _CONTEXT_HEADROOM
        max_input_override = None
        while True:
            sample_df = self._truncate_to_context_window(
                data,
                prompt_template,
                headroom=headroom,
                max_input_override=max_input_override,
            )
            base_prompt = prompt_template.replace("{data}", sample_df.to_csv(index=False))

            self.fit_prompt_ = base_prompt
            logger.info(f"[LLM Prompt]\n{base_prompt}")

            validation_rows = (
                list(generate_feature_dicts(sample_df[self.feature_names_], self.feature_names_))
                if self.feature_names_
                else []
            )
            validation_labels = list(sample_df[self.target_name_]) if self.target_name_ else []

            try:
                raw_code, extended_code, predict_fn = self._generate_code(
                    base_prompt, validation_rows, validation_labels, web_search=self.web_search
                )
                break
            except _ContextWindowExceeded as e:
                if e.real_max_input_tokens and e.real_max_input_tokens != max_input_override:
                    # We now know the model's actual limit (litellm's registry
                    # value was wrong/stale) — retry once with that instead of
                    # spending headroom-shrink budget on a guess.
                    logger.warning(
                        "Correcting max_input_tokens for %r to the API-reported "
                        "value %d and retrying.",
                        self.model,
                        e.real_max_input_tokens,
                    )
                    self.fit_log_.append(
                        {
                            "stage": "context_window",
                            "error": str(e),
                            "action": f"corrected max_input_tokens to {e.real_max_input_tokens}",
                        }
                    )
                    max_input_override = e.real_max_input_tokens
                    continue
                new_headroom = headroom - _CONTEXT_HEADROOM_STEP
                if new_headroom < 0.50:
                    raise RuntimeError(
                        f"Context window exceeded even at {headroom:.0%} headroom "
                        f"(floor is 50%). The prompt is too large for this model."
                    )
                logger.warning(
                    "Context window still exceeded at headroom=%.0f%% — shrinking to "
                    "%.0f%% and retrying.",
                    headroom * 100,
                    new_headroom * 100,
                )
                self.fit_log_.append(
                    {
                        "stage": "context_window",
                        "error": str(e),
                        "action": f"shrunk headroom {headroom:.0%} -> {new_headroom:.0%}",
                    }
                )
                headroom = new_headroom
            except _OutputTruncated:
                new_headroom = headroom - _CONTEXT_HEADROOM_STEP
                if new_headroom < 0.50:
                    raise RuntimeError(
                        f"LLM output was truncated even at {headroom:.0%} context headroom "
                        f"(floor is 50%). The prompt is too large for this model."
                    )
                logger.warning(
                    "Output truncated at headroom=%.0f%% — shrinking to %.0f%% and retrying.",
                    headroom * 100,
                    new_headroom * 100,
                )
                self.fit_log_.append(
                    {
                        "stage": "output_truncated",
                        "action": f"shrunk headroom {headroom:.0%} -> {new_headroom:.0%}",
                    }
                )
                headroom = new_headroom
        self.raw_python_code_ = raw_code
        self.python_code_ = extended_code
        self.predict_fn = predict_fn
        return self

    def _generate_code(
        self,
        base_prompt: str,
        validation_rows: list,
        validation_labels: list = [],
        web_search: bool = False,
    ):
        """Generate code from the LLM with a validation-and-retry loop.

        Returns ``(raw_code, extended_code, fn)``. Each attempt compiles the
        code and runs ``_validate_predict_fn`` over the sample rows; any failure
        triggers a retry with the error fed back to the LLM. Subclasses can
        override ``_validate_predict_fn`` to add stricter checks. Raises the last
        error if every attempt (one initial plus ``max_retries``) fails.
        """
        feedback = ""
        last_error: Optional[Exception] = None
        # One initial attempt plus up to max_retries corrective re-tries.
        for attempt in range(self.max_retries + 1):
            # Web search on the first attempt and the final retry: a retry
            # triggered by a missing-fact problem (wrong constant, bad value
            # mapping, unresolved name) is exactly the case a lookup helps
            # most, so the last attempt shouldn't go in blind -- but every
            # attempt would just add latency without giving the model new
            # information most of the time (most retries are plain logic
            # bugs, not knowledge gaps).
            # _OutputTruncated propagates immediately to _fit for data reduction.
            is_last_attempt = attempt == self.max_retries
            code = self._call_llm(
                base_prompt + feedback, web_search=web_search and (attempt == 0 or is_last_attempt)
            )
            if not isinstance(code, str):
                code = str(code)
            logger.info(f"[LLM Output]\n{code}")

            # Remove markdown/code block if present (triple backticks)
            code = extract_python_code(code)
            try:
                if not code.strip():
                    raise ValueError("No code to exec from LLM output.")
                raw_code = code
                extended_code = self._extend_code(code, web_search=web_search)
                _check_unresolved_names(extended_code)
                fn = make_predict_fn(extended_code)
                self._validate_predict_fn(fn, validation_rows, validation_labels)
            except Exception as e:
                last_error = e
                error_detail = _format_error_with_suggestion(e)
                logger.warning(
                    f"[Validation] Attempt {attempt + 1}/{self.max_retries + 1} failed: {error_detail}"
                )
                self.fit_log_.append(
                    {
                        "stage": "validation",
                        "attempt": attempt + 1,
                        "max_attempts": self.max_retries + 1,
                        "error": error_detail,
                    }
                )
                feedback = (
                    "\n\nThe Python function you previously returned failed validation "
                    f"with this error:\n{error_detail}\n\n"
                    "Fix the problem and return only the corrected, valid Python code."
                )
                if attempt < self.max_retries:
                    time.sleep(5)
                continue

            return raw_code, extended_code, fn

        # Every attempt failed; surface the most recent error.
        assert last_error is not None
        raise last_error

    def _validate_predict_fn(self, predict_fn: Callable, rows: list, labels: list = []) -> None:
        """Run the compiled function over the training sample to confirm it
        executes without raising. Any exception is treated as a validation
        failure so ``_fit`` can retry with the error fed back to the LLM."""
        import inspect

        sig = inspect.signature(predict_fn)
        params = sig.parameters
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not has_var_keyword:
            if rows:
                expected = set(rows[0].keys())
                accepted = set(params.keys())
                missing = expected - accepted
                if missing:
                    raise ValueError(
                        f"Generated predict() does not accept **kwargs and is missing "
                        f"expected feature arguments: {sorted(missing)}. "
                        f"Rewrite predict() to use **features or include all feature arguments."
                    )
        for row in rows:
            predict_fn(**row)

    _EXTEND_MAX_RETRIES = 5

    def _extend_code(self, code: str, web_search: bool = False) -> str:
        """Expand categorical mappings in the generated code via a second LLM pass.

        Validates the result and retries up to _EXTEND_MAX_RETRIES times when the
        LLM returns broken code, feeding the error back each time.  Falls back to
        the original code if all attempts fail.

        ``web_search`` lets the model look up real-world category values (e.g.
        country names, species) it may not know from training data alone --
        this is the primary use case web search is suited for, since expanding
        a lookup table is exactly a factual-recall task. Only forwarded on the
        first attempt, matching ``_generate_code``'s pattern: retries here are
        almost always caused by broken Python syntax from the previous
        attempt, not missing facts, so a repeat search would just add latency.
        """
        logger.info("[Post-Process] Expanding code via second LLM pass...")
        base_prompt = (
            "The following function may use a dictionary, set, or mapping based on domain knowledge (e.g., country names, animal types).\n"
            "Please re-write the function to extend any such mappings with many more possible real-world keys, if applicable.\n"
            "Try to figure out the logic of the function based on the variable names and values that are processed in the function.\n"
            "Avoid changing the logic or structure beyond extending categorical support.\n"
            "Only return valid Python code.\n\n"
            f"{code}"
        )
        feedback = ""
        for attempt in range(self._EXTEND_MAX_RETRIES):
            try:
                refined_code = self._call_llm(
                    base_prompt + feedback, web_search=web_search and attempt == 0
                )
                refined_code = extract_python_code(str(refined_code))
                compile(refined_code, "<extend>", "exec")
                make_predict_fn(refined_code)
                logger.info("[Post-Process] Successfully extended function.")
                return refined_code
            except Exception as e:
                logger.warning(
                    "[Post-Process] Attempt %d/%d failed: %s",
                    attempt + 1,
                    self._EXTEND_MAX_RETRIES,
                    e,
                )
                feedback = (
                    f"\n\nThe code you returned has an error: {e}\n"
                    "Fix it and return only valid Python code."
                )
        logger.warning("[Post-Process] All extend attempts failed; using original code.")
        return code

    def sample(self, n: int = 5):
        """Generate n synthetic examples that illustrate the heuristic."""
        # Check that columns have some sort of names
        if (
            not hasattr(self, "feature_names_")
            or self.feature_names_ is None
            or not hasattr(self, "target_name_")
            or self.target_name_ is None
        ):
            raise RuntimeError("Call fit() before sample(): feature names or target name not set.")
        prompt = (
            f"{self.python_code_}\n\n"
            f"Please generate {n} example rows in tabular format with the following columns:\n"
            f"{', '.join(self.feature_names_ + [self.target_name_])}.\n"
            f"Use tab-separated format. Do not explain."
        )
        text = self._call_llm(prompt)
        return parse_tsv(text)

    def explain(self, X=None) -> Explanation:
        """Return a plain-English explanation of the fitted heuristic.

        With no argument, returns a **global** explanation of the rule the model
        encodes (cached, so repeated calls are deterministic). Given a single-row
        ``X``, returns a **local** explanation of that one prediction.
        """
        if not getattr(self, "python_code_", None):
            raise NotFittedError("Call fit() before explain().")

        features_used = self._features_used()

        if X is not None:
            instance = next(iter(generate_feature_dicts(X, self.feature_names_)), {})
            summary = self._call_llm(self._local_explain_prompt(instance))
            return Explanation(
                meta=self._explanation_meta(["local"]),
                data={
                    "summary": summary.strip(),
                    "features_used": features_used,
                    "instance": instance,
                },
            )

        # Global explanation is computed once and cached for determinism.
        if self.explanation_ is None:
            summary = self._call_llm(self._global_explain_prompt())
            self.explanation_ = Explanation(
                meta=self._explanation_meta(["global"]),
                data={
                    "summary": summary.strip(),
                    "features_used": features_used,
                    "code": self.python_code_,
                },
            )
        return self.explanation_

    def _explanation_meta(self, scope: list) -> dict:
        # The generated Python heuristic is fully visible, so this is a whitebox
        # explanation in the Alibi sense.
        return {
            "name": type(self).__name__,
            "type": ["whitebox"],
            "explanations": scope,
            "params": self.get_params(),
        }

    def _features_used(self) -> list:
        """Features the heuristic actually references — never invents new ones."""
        names = self.feature_names_ or []
        code = self.python_code_ or ""
        used = [n for n in names if re.search(rf"\b{re.escape(n)}\b", code)]
        return used or list(names)

    def _global_explain_prompt(self) -> str:
        return (
            "You are documenting a trained model for an interpretability report. "
            "Below is the Python function it uses to make predictions. In clear, "
            "concise plain English, describe the rule it encodes: which input "
            "features it uses and how they determine the output. Be faithful to "
            "the code and do not mention features that are not present.\n\n"
            f"Target: {self.target_name_}\n"
            f"Features: {', '.join(self.feature_names_ or [])}\n\n"
            f"{self.python_code_}"
        )

    def _local_explain_prompt(self, instance: dict) -> str:
        return (
            "Below is the Python function a trained model uses to make "
            "predictions, followed by one specific input. In plain English, "
            "explain why this particular input yields its prediction, referring "
            "to the relevant feature values. Be faithful to the code.\n\n"
            f"{self.python_code_}\n\n"
            f"Input: {instance}"
        )
