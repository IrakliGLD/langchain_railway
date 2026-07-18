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
    def _invoke_kwargs(provider: str, timeout_seconds: float) -> dict[str, float | int]:
        if provider == "gemini":
            # google-genai uses milliseconds and counts the first attempt, so
            # one attempt disables SDK-owned retries.
            return {
                "timeout": max(1, int(timeout_seconds * 1000)),
                "max_retries": 1,
            }
        return {"timeout": timeout_seconds}

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
            kwargs = self._invoke_kwargs(provider, timeout_seconds)
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
