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
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config import (
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_CHAT_TEMPLATE_KWARGS,
    NVIDIA_MAX_TOKENS,
    NVIDIA_MODEL,
    NVIDIA_TEMPERATURE,
    NVIDIA_TOP_P,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

log = logging.getLogger("Enai")


def _to_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
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
            prompt_tokens = max(prompt_tokens, _to_int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens")))
            completion_tokens = max(completion_tokens, _to_int(token_usage.get("completion_tokens") or token_usage.get("output_tokens")))
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
        coalesce_timeout: float = 180.0,
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
        self._in_flight: dict[str, _threading.Event] = {}

    def _make_key(self, prompt: str) -> str:
        """Generate cache key from prompt hash."""
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]

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

            event = self._in_flight.get(key)
            if event is None:
                # True miss — no cached value, nobody computing it.
                self._misses += 1
                return None

        # Another thread is computing this key — wait for it.
        log.info("⏳ LLM cache: waiting for in-flight result (key=%.8s…)", key)
        signaled = event.wait(timeout=self._coalesce_timeout)

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

    def set(self, prompt: str, response: str):
        """Cache response for prompt and wake any waiting threads."""
        key = self._make_key(prompt)

        with self._lock:
            if len(self._cache) >= self._max_size:
                remove_count = max(1, self._max_size // 10)
                for _ in range(remove_count):
                    self._cache.pop(next(iter(self._cache)), None)
                log.info("🗑️ Cache eviction: removed %d oldest entries", remove_count)
            self._cache[key] = response
            event = self._in_flight.pop(key, None)

        if event is not None:
            event.set()  # Wake all waiters.

    # --- coalescing lifecycle ---

    def mark_in_flight(self, prompt: str):
        """Mark *prompt* as being computed.  Must be followed by ``set()`` or
        ``cancel_in_flight()`` (use try/finally)."""
        key = self._make_key(prompt)
        with self._lock:
            if key not in self._in_flight:
                self._in_flight[key] = _threading.Event()

    def cancel_in_flight(self, prompt: str):
        """Remove the in-flight marker without caching a value.  Wakes any
        waiting threads so they can retry independently."""
        key = self._make_key(prompt)
        with self._lock:
            event = self._in_flight.pop(key, None)
        if event is not None:
            event.set()

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


def _chat_openai_supports_kwarg(name: str) -> bool:
    fields = getattr(ChatOpenAI, "model_fields", None) or getattr(ChatOpenAI, "__fields__", {})
    return name in fields


def _set_chat_openai_request_option(
    kwargs: dict,
    model_kwargs: dict,
    name: str,
    value,
) -> None:
    if _chat_openai_supports_kwarg(name):
        kwargs[name] = value
    else:
        model_kwargs[name] = value


def _build_nvidia_chat_openai_kwargs() -> dict:
    kwargs = {
        "model": NVIDIA_MODEL,
        "temperature": NVIDIA_TEMPERATURE,
        "max_tokens": NVIDIA_MAX_TOKENS,
        "openai_api_key": NVIDIA_API_KEY,
        "base_url": NVIDIA_BASE_URL,
        "max_retries": 2,  # Limit retries to prevent quota exhaustion
    }
    model_kwargs = {}
    if NVIDIA_TOP_P is not None:
        _set_chat_openai_request_option(kwargs, model_kwargs, "top_p", NVIDIA_TOP_P)
    if NVIDIA_CHAT_TEMPLATE_KWARGS:
        _set_chat_openai_request_option(
            kwargs,
            model_kwargs,
            "extra_body",
            {"chat_template_kwargs": dict(NVIDIA_CHAT_TEMPLATE_KWARGS)},
        )
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    return kwargs


def get_gemini() -> ChatGoogleGenerativeAI:
    """Get cached Gemini LLM instance (singleton pattern).

    Note: convert_system_message_to_human=True is required because Gemini
    doesn't natively support SystemMessages in the LangChain interface.

    Retry configuration: max_retries=2 to prevent quota exhaustion from
    aggressive retry behavior (default is 6 retries with exponential backoff).
    """
    global _gemini_llm
    if _gemini_llm is None:
        _gemini_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            convert_system_message_to_human=True,
            max_retries=2,  # Limit retries to prevent quota exhaustion
            timeout=120     # Allow up to 120s for large prompts (default is 60s)
        )
        log.info("✅ Gemini LLM instance cached (max_retries=2, timeout=120s)")
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
        _openai_llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            max_retries=2  # Limit retries to prevent quota exhaustion
        )
        log.info("✅ OpenAI LLM instance cached (max_retries=2)")
    return _openai_llm


def get_nvidia() -> ChatOpenAI:
    """Get cached NVIDIA LLM instance (singleton pattern).

    build.nvidia.com exposes an OpenAI-compatible API, so it is driven through
    ``ChatOpenAI`` — identical to ``get_openai()`` except for the custom
    ``base_url``. The model id and base URL come from env. The key is the
    resolved config value from NVIDIA_OPENAI_API_KEY / NVIDIA_GEMMA_API_KEY,
    falling back to NVIDIA_API_KEY.

    Raises:
        RuntimeError: If NVIDIA_API_KEY is not configured
    """
    global _nvidia_llm
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY not set")
    if _nvidia_llm is None:
        _nvidia_llm = ChatOpenAI(**_build_nvidia_chat_openai_kwargs())
        log.info(
            (
                "NVIDIA LLM instance cached "
                "(model=%s, max_tokens=%s, temperature=%s, top_p=%s, chat_template_kwargs=%s, max_retries=2)"
            ),
            NVIDIA_MODEL,
            NVIDIA_MAX_TOKENS,
            NVIDIA_TEMPERATURE,
            NVIDIA_TOP_P,
            NVIDIA_CHAT_TEMPLATE_KWARGS,
        )
    return _nvidia_llm
