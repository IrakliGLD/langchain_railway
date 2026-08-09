"""Bounded native provider invocation behind one delivery-aware interface."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Protocol

from utils.provider_attempts import (
    ProviderDeliveryDisposition,
    ProviderExecutionError,
)


class CircuitBreaker(Protocol):
    def allow_request(self) -> tuple[bool, str]: ...

    def record_success(self) -> None: ...

    def record_failure(self) -> None: ...


# Providers whose SDK accepts ``prompt_cache_key``. langchain-openai documents
# it as an invoke kwarg; nothing else here does.
_PROMPT_CACHE_KEY_PROVIDERS = frozenset({"openai"})


class ProviderInvocationRuntime:
    """Execute exactly one native provider attempt and finalize its outcome."""

    def __init__(
        self,
        *,
        claim_attempt: Callable[[str, str], Any],
        finish_attempt: Callable[[Any, ProviderDeliveryDisposition], Any],
        classify_failure: Callable[[BaseException], ProviderDeliveryDisposition],
        wrap_failure: Callable[..., ProviderExecutionError],
        log_circuit_open: Callable[[str], Any],
    ) -> None:
        self._claim_attempt = claim_attempt
        self._finish_attempt = finish_attempt
        self._classify_failure = classify_failure
        self._wrap_failure = wrap_failure
        self._log_circuit_open = log_circuit_open

    @staticmethod
    def _invoke_kwargs(
        provider: str,
        timeout_seconds: float,
        sampling_temperature: float | None = None,
        prompt_cache_key: str = "",
    ) -> dict[str, float | int | str]:
        if provider == "gemini":
            # langchain-google-genai accepts timeout in seconds and converts it
            # to the google-genai HTTP millisecond value internally. The wrapper
            # also counts the first attempt, so one attempt disables SDK retries.
            kwargs: dict[str, float | int | str] = {
                "timeout": max(0.001, float(timeout_seconds)),
                "max_retries": 1,
            }
        else:
            kwargs = {"timeout": timeout_seconds}
        if sampling_temperature is not None:
            kwargs["temperature"] = float(sampling_temperature)
        # OpenAI-only routing affinity: same key, better odds of landing on a
        # machine that already holds the prefix. It is a hint, not a cache
        # breakpoint — this API has no such control. Allow-listed rather than
        # deny-listed, so a provider added later cannot inherit an argument its
        # SDK would reject.
        if prompt_cache_key and provider in _PROMPT_CACHE_KEY_PROVIDERS:
            kwargs["prompt_cache_key"] = prompt_cache_key
        return kwargs

    @staticmethod
    def _accepts_invoke_kwargs(invoke: Callable[..., Any]) -> bool:
        try:
            signature = inspect.signature(invoke)
        except (TypeError, ValueError):
            return True
        return (
            any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
            or "timeout" in signature.parameters
        )

    def invoke(
        self,
        llm: Any,
        messages: Any,
        *,
        provider: str,
        stage: str,
        timeout_seconds: float,
        breaker: CircuitBreaker,
        sampling_temperature: float | None = None,
        prompt_cache_key: str = "",
    ) -> Any:
        allowed, reason = breaker.allow_request()
        if not allowed:
            self._log_circuit_open(f"llm_{provider}")
            raise ProviderExecutionError(
                f"LLM circuit breaker open for provider={provider} reason={reason}",
                provider=provider,
                stage=stage,
                disposition=ProviderDeliveryDisposition.REJECTED,
            )

        token = self._claim_attempt(provider, stage)
        try:
            kwargs = self._invoke_kwargs(
                provider, timeout_seconds, sampling_temperature, prompt_cache_key
            )
            message = (
                llm.invoke(messages, **kwargs) if self._accepts_invoke_kwargs(llm.invoke) else llm.invoke(messages)
            )
        except Exception as error:
            disposition = self._classify_failure(error)
            self._finish_attempt(token, disposition)
            if disposition == ProviderDeliveryDisposition.PERMANENT_FAILURE:
                breaker.record_success()
            else:
                breaker.record_failure()
            raise self._wrap_failure(
                error,
                provider=provider,
                stage=stage,
                disposition=disposition,
            ) from error

        self._finish_attempt(token, ProviderDeliveryDisposition.COMPLETED)
        breaker.record_success()
        return message
