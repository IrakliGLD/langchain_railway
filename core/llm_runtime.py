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
    GOOGLE_API_KEY,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_MAX_TOKENS,
    NVIDIA_MODEL,
    NVIDIA_TEMPERATURE,
    NVIDIA_TIMEOUT_SECONDS,
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

            flight = self._in_flight.get(key)
            if flight is None:
                # True miss — no cached value, nobody computing it.
                self._misses += 1
                return None

        # Another thread is computing this key — wait for it.
        log.info("⏳ LLM cache: waiting for in-flight result (key=%.8s…)", key)
        event, _token, _started_at = flight
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

    def get_or_reserve(self, prompt: str) -> tuple[Optional[str], object | None]:
        """Atomically return a cached value or reserve singleflight ownership.

        The opaque token must be passed to :meth:`set` or
        :meth:`cancel_in_flight`. A waiter can replace a timed-out owner, and a
        late result from that stale owner is then discarded.
        """

        key = self._make_key(prompt)
        deadline = _time.monotonic() + (self._coalesce_timeout * 2)
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
            max_retries=2,  # Limit retries to prevent quota exhaustion
        )
        if NVIDIA_TIMEOUT_SECONDS:
            # Bounded call: a timeout must reach the OpenAI fallback after ONE
            # attempt — retrying a slow model just multiplies the wait (see the
            # NVIDIA_TIMEOUT_SECONDS comment in config.py).
            client_kwargs["request_timeout"] = NVIDIA_TIMEOUT_SECONDS
            client_kwargs["max_retries"] = 1
        _nvidia_llm = ChatOpenAI(**client_kwargs)
        log.info(
            "✅ NVIDIA LLM instance cached (model=%s, max_tokens=%s, temperature=%s, "
            "timeout=%s, max_retries=%s)",
            NVIDIA_MODEL, NVIDIA_MAX_TOKENS, NVIDIA_TEMPERATURE,
            NVIDIA_TIMEOUT_SECONDS or "unbounded", client_kwargs["max_retries"],
        )
    return _nvidia_llm
