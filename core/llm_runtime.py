"""
LLM provider runtime: client factories, response cache, token/cost accounting.

Extracted from ``core/llm.py`` (Q1, 2026-06-10) as a pure structural move.

IMPORTANT — patch-surface contract: the orchestration symbols that tests
monkeypatch by module path (``llm_cache``, ``_invoke_with_resilience``,
``get_llm_for_stage``, ``_log_usage_for_message``, ``make_gemini``/
``make_openai``) deliberately REMAIN in ``core.llm``. This module holds only
the implementation layer beneath them. Do not move patched symbols here
without migrating every ``monkeypatch.setattr(llm_core, ...)`` call site.
"""

import hashlib
import logging
import threading as _threading
import time as _time
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config import (
    GEMINI_MODEL,
    GEMINI_TIMEOUT_SECONDS,
    GOOGLE_API_KEY,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_MAX_TOKENS,
    NVIDIA_MODEL,
    NVIDIA_TEMPERATURE,
    NVIDIA_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_SECONDS,
    PROVIDER_MINIMUM_START_BUDGET_MS,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MAX_TOKENS,
    QWEN_MODEL,
    QWEN_TEMPERATURE,
    QWEN_TIMEOUT_SECONDS,
    REPORT_MAX_OUTPUT_TOKENS,
    REPORT_MODEL,
    REPORT_MODEL_TYPE,
    REPORT_REASONING_EFFORT,
    REPORT_TIMEOUT_SECONDS,
    REQUEST_CLEANUP_ALLOWANCE_MS,
)

log = logging.getLogger("Enai")


def _bounded_coalesce_wait_seconds(configured_seconds: float) -> float:
    from utils.request_deadline import current_request_execution_scope

    scope = current_request_execution_scope()
    if scope is None or scope.deadline is None:
        return configured_seconds
    return scope.deadline.bounded_timeout_seconds(
        "llm_coalesce_wait",
        configured_timeout_seconds=configured_seconds,
        cleanup_allowance_ms=REQUEST_CLEANUP_ALLOWANCE_MS,
        minimum_start_ms=PROVIDER_MINIMUM_START_BUDGET_MS,
    )


def _to_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _extract_cached_prompt_tokens(message) -> int:
    """Best-effort extraction of the cached share of the prompt.

    Providers price a cache hit below a fresh prompt token, so the split has to
    be visible before caching can be tuned or costed. OpenAI-compatible
    endpoints report it under ``prompt_tokens_details.cached_tokens``; Gemini
    reports ``cached_content_token_count``. Anything unreported reads as 0,
    which is indistinguishable from a genuine miss — that is the honest answer
    when a provider does not tell us.
    """

    candidates: list[dict] = []
    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        candidates.append(usage_metadata)
        details = usage_metadata.get("input_token_details")
        if isinstance(details, dict):
            candidates.append(details)
    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        for key in ("token_usage", "usage"):
            usage = response_metadata.get(key)
            if isinstance(usage, dict):
                candidates.append(usage)
                details = usage.get("prompt_tokens_details")
                if isinstance(details, dict):
                    candidates.append(details)
        details = response_metadata.get("usage_metadata")
        if isinstance(details, dict):
            candidates.append(details)
    for source in candidates:
        for field in (
            "cached_tokens",
            "cache_read",
            "cache_read_input_tokens",
            "cached_content_token_count",
        ):
            value = _to_int(source.get(field))
            if value > 0:
                return value
    return 0


def _extract_token_usage(message) -> tuple[int, int, int]:
    """Best-effort extraction of prompt/completion/total tokens from LLM message."""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        prompt_tokens = _to_int(usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens"))
        completion_tokens = _to_int(usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens"))
        total_tokens = _to_int(usage_metadata.get("total_tokens"))

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
        if isinstance(token_usage, dict):
            prompt_tokens = max(
                prompt_tokens, _to_int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens"))
            )
            completion_tokens = max(
                completion_tokens, _to_int(token_usage.get("completion_tokens") or token_usage.get("output_tokens"))
            )
            total_tokens = max(total_tokens, _to_int(token_usage.get("total_tokens")))

    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


# NOTE: _is_openai_model_name / _estimate_cost_usd / _provider_from_model_name
# stay in core/llm.py — they read provider config constants that tests
# monkeypatch on the core.llm module (see test_metrics_observability).

# -----------------------------
# LLM Response Cache (Phase 1 Optimization + Request Coalescing)
# -----------------------------


class LLMResponseCache:
    """Thread-safe in-memory cache with request coalescing for LLM responses.

    Phase 1 optimization: Cache identical prompts to avoid repeated LLM calls.
    Phase 2 optimization: Request coalescing ("singleflight") prevents stampeding
    herd cache misses.  When multiple threads request the same prompt concurrently,
    only one thread calls the LLM.  The remaining threads block on a
    ``threading.Event`` until the leader finishes, then read from the cache.

    The public API (``get`` / ``set``) is fully backward-compatible.  The
    coalescing lifecycle is:
      1. ``get()`` → cache miss → returns ``None``
      2. Caller calls ``mark_in_flight(prompt)``
      3. Caller calls the LLM
      4. Caller calls ``set(prompt, response)`` on success (or
         ``cancel_in_flight(prompt)`` on failure)

    If a concurrent thread calls ``get()`` while a key is in-flight, the get
    blocks up to ``coalesce_timeout`` seconds waiting for the leader.
    """

    def __init__(
        self,
        max_size: int = 1000,
        coalesce_timeout: float = 95.0,
    ):
        self._cache: dict[str, str] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._coalesce_hits = 0
        self._coalesce_timeout = coalesce_timeout
        # Guards both _cache and _in_flight mutations.
        self._lock = _threading.Lock()
        # key → Event; set when the leader finishes (success or failure).
        self._in_flight: dict[str, tuple[_threading.Event, object, float]] = {}

    def _make_key(self, prompt: str) -> str:
        """Generate cache key from prompt hash."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    # --- core API (backward-compatible) ---

    def get(self, prompt: str) -> Optional[str]:
        """Return cached response, or ``None`` on a true miss.

        If another thread is currently computing the same key (in-flight), this
        method blocks until the leader finishes, then returns the cached value
        (or ``None`` if the leader failed).
        """
        key = self._make_key(prompt)

        with self._lock:
            if key in self._cache:
                self._hits += 1
                log.info("✅ LLM cache HIT (hit rate: %.1f%%)", self.hit_rate() * 100)
                return self._cache[key]

            flight = self._in_flight.get(key)
            if flight is None:
                # True miss — no cached value, nobody computing it.
                self._misses += 1
                return None

        # Another thread is computing this key — wait for it.
        log.info("⏳ LLM cache: waiting for in-flight result (key=%.8s…)", key)
        event, _token, _started_at = flight
        signaled = event.wait(timeout=_bounded_coalesce_wait_seconds(self._coalesce_timeout))

        with self._lock:
            result = self._cache.get(key)

        if result is not None:
            self._coalesce_hits += 1
            log.info(
                "✅ LLM cache COALESCE HIT (waited=%.0fs, hit rate: %.1f%%)",
                0 if signaled else self._coalesce_timeout,
                self.hit_rate() * 100,
            )
            return result

        # Leader failed — caller should proceed as a fresh miss.
        self._misses += 1
        return None

    def get_or_reserve(self, prompt: str) -> tuple[Optional[str], object | None]:
        """Atomically return a cached value or reserve singleflight ownership.

        The opaque token must be passed to :meth:`set` or
        :meth:`cancel_in_flight`. A waiter can replace a timed-out owner, and a
        late result from that stale owner is then discarded.
        """

        key = self._make_key(prompt)
        deadline = _time.monotonic() + _bounded_coalesce_wait_seconds(self._coalesce_timeout * 2)
        observed_flight = None
        while True:
            with self._lock:
                if key in self._cache:
                    self._hits += 1
                    return self._cache[key], None
                flight = self._in_flight.get(key)
                if flight is None:
                    token = object()
                    self._in_flight[key] = (_threading.Event(), token, _time.monotonic())
                    self._misses += 1
                    return None, token

            event, _token, _started_at = flight
            if flight is not observed_flight:
                log.info("LLM cache: waiting for in-flight result (key=%.8s...)", key)
                observed_flight = flight
            remaining = max(0.0, deadline - _time.monotonic())
            signaled = event.wait(timeout=min(self._coalesce_timeout, remaining))

            with self._lock:
                result = self._cache.get(key)
                if result is not None:
                    self._coalesce_hits += 1
                    return result, None
                current = self._in_flight.get(key)
                if not signaled and current is flight:
                    replacement_token = object()
                    replacement_event = _threading.Event()
                    self._in_flight[key] = (
                        replacement_event,
                        replacement_token,
                        _time.monotonic(),
                    )
                    event.set()
                    self._misses += 1
                    log.warning("LLM cache replaced stale in-flight owner (key=%.8s...)", key)
                    return None, replacement_token

            if _time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for coalesced LLM result key={key[:8]}")

    def set(self, prompt: str, response: str, token: object | None = None):
        """Cache response for prompt and wake any waiting threads."""
        key = self._make_key(prompt)

        with self._lock:
            current = self._in_flight.get(key)
            if token is not None and (current is None or current[1] is not token):
                log.warning("Discarding stale LLM cache result (key=%.8s...)", key)
                return False
            if len(self._cache) >= self._max_size:
                remove_count = max(1, self._max_size // 10)
                for _ in range(remove_count):
                    self._cache.pop(next(iter(self._cache)), None)
                log.info("🗑️ Cache eviction: removed %d oldest entries", remove_count)
            self._cache[key] = response
            flight = self._in_flight.pop(key, None)

        if flight is not None:
            event = flight[0]
            event.set()  # Wake all waiters.
        return True

    # --- coalescing lifecycle ---

    def mark_in_flight(self, prompt: str):
        """Mark *prompt* as being computed.  Must be followed by ``set()`` or
        ``cancel_in_flight()`` (use try/finally)."""
        key = self._make_key(prompt)
        with self._lock:
            if key not in self._in_flight:
                token = object()
                self._in_flight[key] = (_threading.Event(), token, _time.monotonic())
                return token
        return None

    def cancel_in_flight(self, prompt: str, token: object | None = None):
        """Remove the in-flight marker without caching a value.  Wakes any
        waiting threads so they can retry independently."""
        key = self._make_key(prompt)
        with self._lock:
            current = self._in_flight.get(key)
            if token is not None and (current is None or current[1] is not token):
                return False
            flight = self._in_flight.pop(key, None)
        if flight is not None:
            event = flight[0]
            event.set()
        return True

    # --- stats ---

    def hit_rate(self) -> float:
        """Calculate cache hit rate (includes coalesce hits)."""
        total = self._hits + self._coalesce_hits + self._misses
        return (self._hits + self._coalesce_hits) / total if total > 0 else 0.0

    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "coalesce_hits": self._coalesce_hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate(),
            "in_flight": len(self._in_flight),
        }


# -----------------------------
# LLM Instances (Singleton Pattern)
# -----------------------------

_gemini_llm = None
_openai_llm = None
_nvidia_llm = None
_qwen_llm = None
_report_llm = None


def get_gemini() -> ChatGoogleGenerativeAI:
    """Get cached Gemini LLM instance (singleton pattern).

    Note: convert_system_message_to_human=True is required because Gemini
    doesn't natively support SystemMessages in the LangChain interface.

    Retry configuration: max_retries=1 to prevent quota exhaustion from
    aggressive retry behavior (default is 6 retries with exponential backoff).
    """
    global _gemini_llm
    if _gemini_llm is None:
        _gemini_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            convert_system_message_to_human=True,
            max_retries=1,  # attempts includes the first call: no SDK retry
            request_timeout=max(0.001, float(GEMINI_TIMEOUT_SECONDS or 120.0)),
        )
        log.info("✅ Gemini LLM instance cached (max_retries=1, timeout=%ss)", GEMINI_TIMEOUT_SECONDS or 120.0)
    return _gemini_llm


def get_openai() -> ChatOpenAI:
    """Get cached OpenAI LLM instance (singleton pattern).

    Raises:
        RuntimeError: If OPENAI_API_KEY is not configured
    """
    global _openai_llm
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set (fallback needed)")
    if _openai_llm is None:
        client_kwargs = dict(
            model=OPENAI_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            max_retries=0,  # application owns the only safe fallback
        )
        if OPENAI_TIMEOUT_SECONDS:
            # P5.1 (H13): a bounded call keeps a stalled OpenAI request from
            # holding the end-to-end deadline; drop retries to 1 so a timeout
            # fails over once instead of multiplying the wait.
            client_kwargs["request_timeout"] = OPENAI_TIMEOUT_SECONDS
            client_kwargs["max_retries"] = 0
        _openai_llm = ChatOpenAI(**client_kwargs)
        log.info(
            "✅ OpenAI LLM instance cached (timeout=%s, max_retries=%s)",
            OPENAI_TIMEOUT_SECONDS or "request-deadline-bound",
            client_kwargs["max_retries"],
        )
    return _openai_llm


def get_qwen() -> ChatOpenAI:
    """Get cached Qwen LLM instance (singleton pattern).

    qwencloud exposes an OpenAI-compatible API, so it is driven through
    ``ChatOpenAI`` with a custom ``base_url`` — the same shape as
    ``get_nvidia()``, but under its own provider key so cost attribution and the
    circuit breaker do not merge with NVIDIA's.

    ``reasoning_effort`` is deliberately never sent: it is an OpenAI-specific
    parameter and compatible endpoints reject unknown sampling arguments.

    Raises:
        RuntimeError: If the key or the compatible-mode base URL is missing.
    """
    global _qwen_llm
    if not QWEN_API_KEY:
        raise RuntimeError(
            "QWEN_API_KEY (or DASHSCOPE_API_KEY) not set"
        )
    if not QWEN_BASE_URL:
        raise RuntimeError("QWEN_BASE_URL not set")
    if _qwen_llm is None:
        client_kwargs = dict(
            model=QWEN_MODEL,
            temperature=QWEN_TEMPERATURE,
            openai_api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
            max_retries=0,  # application owns the only safe fallback
        )
        # Only send an output cap when one is configured: Qwen's
        # structured-output guide warns that truncated output produces invalid
        # JSON, and every report stage returns JSON.
        if QWEN_MAX_TOKENS:
            client_kwargs["max_tokens"] = QWEN_MAX_TOKENS
        if QWEN_TIMEOUT_SECONDS:
            client_kwargs["request_timeout"] = QWEN_TIMEOUT_SECONDS
        _qwen_llm = ChatOpenAI(**client_kwargs)
        log.info(
            "✅ Qwen LLM instance cached (model=%s, max_tokens=%s, "
            "temperature=%s, timeout=%s)",
            QWEN_MODEL,
            QWEN_MAX_TOKENS,
            QWEN_TEMPERATURE,
            QWEN_TIMEOUT_SECONDS,
        )
    return _qwen_llm


def get_nvidia() -> ChatOpenAI:
    """Get cached NVIDIA LLM instance (singleton pattern).

    build.nvidia.com exposes an OpenAI-compatible API, so it is driven through
    ``ChatOpenAI`` — identical to ``get_openai()`` except for the custom
    ``base_url``. The model id (e.g. ``openai/gpt-oss-120b``), key, and base URL
    all come from env (NVIDIA_MODEL / NVIDIA_API_KEY / NVIDIA_BASE_URL).

    Raises:
        RuntimeError: If NVIDIA_API_KEY is not configured
    """
    global _nvidia_llm
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY not set")
    if _nvidia_llm is None:
        client_kwargs = dict(
            model=NVIDIA_MODEL,
            temperature=NVIDIA_TEMPERATURE,
            max_tokens=NVIDIA_MAX_TOKENS,
            openai_api_key=NVIDIA_API_KEY,
            base_url=NVIDIA_BASE_URL,
            max_retries=0,  # application owns the only safe fallback
        )
        if NVIDIA_TIMEOUT_SECONDS:
            # Bounded call: a timeout must reach the OpenAI fallback after ONE
            # attempt — retrying a slow model just multiplies the wait (see the
            # NVIDIA_TIMEOUT_SECONDS comment in config.py).
            client_kwargs["request_timeout"] = NVIDIA_TIMEOUT_SECONDS
            client_kwargs["max_retries"] = 0
        _nvidia_llm = ChatOpenAI(**client_kwargs)
        log.info(
            "✅ NVIDIA LLM instance cached (model=%s, max_tokens=%s, temperature=%s, timeout=%s, max_retries=%s)",
            NVIDIA_MODEL,
            NVIDIA_MAX_TOKENS,
            NVIDIA_TEMPERATURE,
            NVIDIA_TIMEOUT_SECONDS or "request-deadline-bound",
            client_kwargs["max_retries"],
        )
    return _nvidia_llm


def get_report():
    """Return the dedicated durable-report client selected by REPORT_*.

    The report profile is constructed only inside the report worker path.
    Configuration validation normally catches incomplete profiles at startup;
    the local checks here keep the factory safe when called directly in tests
    or utility processes.
    """
    global _report_llm
    if not REPORT_MODEL_TYPE or not REPORT_MODEL:
        raise RuntimeError(
            "Dedicated report provider requires REPORT_MODEL_TYPE and REPORT_MODEL"
        )
    if _report_llm is not None:
        return _report_llm

    common_output = {
        "max_output_tokens": REPORT_MAX_OUTPUT_TOKENS,
        "timeout": REPORT_TIMEOUT_SECONDS,
    }
    if REPORT_MODEL_TYPE == "gemini":
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "REPORT_MODEL_TYPE=gemini but GOOGLE_API_KEY is missing"
            )
        client_kwargs = {
            "model": REPORT_MODEL,
            "google_api_key": GOOGLE_API_KEY,
            "convert_system_message_to_human": True,
            **common_output,
            "max_retries": 1,
        }
        if REPORT_REASONING_EFFORT:
            client_kwargs["thinking_level"] = REPORT_REASONING_EFFORT
        _report_llm = ChatGoogleGenerativeAI(**client_kwargs)
    elif REPORT_MODEL_TYPE in {"openai", "nvidia", "qwen"}:
        api_key = {
            "openai": OPENAI_API_KEY,
            "nvidia": NVIDIA_API_KEY,
            "qwen": QWEN_API_KEY,
        }[REPORT_MODEL_TYPE]
        if not api_key:
            key_name = {
                "openai": "OPENAI_API_KEY",
                "nvidia": "NVIDIA_API_KEY",
                "qwen": "QWEN_API_KEY",
            }[REPORT_MODEL_TYPE]
            raise RuntimeError(
                f"REPORT_MODEL_TYPE={REPORT_MODEL_TYPE} but {key_name} is missing"
            )
        client_kwargs = {
            "model": REPORT_MODEL,
            "openai_api_key": api_key,
            "max_tokens": REPORT_MAX_OUTPUT_TOKENS,
            "request_timeout": REPORT_TIMEOUT_SECONDS,
            "max_retries": 0,
        }
        if REPORT_MODEL_TYPE == "openai":
            client_kwargs["use_responses_api"] = True
        elif REPORT_MODEL_TYPE == "qwen":
            client_kwargs["base_url"] = QWEN_BASE_URL
        else:
            client_kwargs["base_url"] = NVIDIA_BASE_URL
        if REPORT_REASONING_EFFORT:
            client_kwargs["reasoning_effort"] = REPORT_REASONING_EFFORT
        _report_llm = ChatOpenAI(**client_kwargs)
    else:
        raise RuntimeError(
            "Invalid REPORT_MODEL_TYPE. Expected one of: gemini, openai, nvidia"
        )

    log.info(
        "Report LLM instance cached: provider=%s model=%s "
        "max_output_tokens=%s timeout=%ss reasoning_effort=%s",
        REPORT_MODEL_TYPE,
        REPORT_MODEL,
        REPORT_MAX_OUTPUT_TOKENS,
        REPORT_TIMEOUT_SECONDS,
        REPORT_REASONING_EFFORT or "provider_default",
    )
    return _report_llm
