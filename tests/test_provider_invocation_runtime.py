"""Characterization tests for the prompt/provider invocation boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.provider_invocation import ProviderInvocationRuntime
from utils.provider_attempts import ProviderDeliveryDisposition, ProviderExecutionError


class _Breaker:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.successes = 0
        self.failures = 0

    def allow_request(self):
        return self.allowed, "open" if not self.allowed else "closed"

    def record_success(self):
        self.successes += 1

    def record_failure(self):
        self.failures += 1


class _Dependencies:
    def __init__(self, disposition=ProviderDeliveryDisposition.TIMED_OUT):
        self.disposition = disposition
        self.claims = []
        self.finishes = []
        self.circuit_open = []

    def claim(self, provider, stage):
        token = (provider, stage, len(self.claims))
        self.claims.append(token)
        return token

    def finish(self, token, disposition):
        self.finishes.append((token, disposition))

    def classify(self, _error):
        return self.disposition

    def wrap(self, error, *, provider, stage, disposition):
        return ProviderExecutionError(
            str(error),
            provider=provider,
            stage=stage,
            disposition=disposition,
        )

    def log_circuit_open(self, name):
        self.circuit_open.append(name)


def _runtime(dependencies):
    return ProviderInvocationRuntime(
        claim_attempt=dependencies.claim,
        finish_attempt=dependencies.finish,
        classify_failure=dependencies.classify,
        wrap_failure=dependencies.wrap,
        log_circuit_open=dependencies.log_circuit_open,
    )


@pytest.mark.parametrize(
    ("provider", "timeout_seconds", "expected_kwargs"),
    [
        ("gemini", 5.0, {"timeout": 5000, "max_retries": 1}),
        ("openai", 7.0, {"timeout": 7.0}),
        ("nvidia", 9.0, {"timeout": 9.0}),
    ],
)
def test_successful_native_call_has_one_attempt_and_provider_specific_timeout(
    provider, timeout_seconds, expected_kwargs
):
    dependencies = _Dependencies()
    runtime = _runtime(dependencies)
    breaker = _Breaker()
    calls = []

    class _LLM:
        def invoke(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return SimpleNamespace(content="ok")

    message = runtime.invoke(
        _LLM(),
        [("system", "rules"), ("user", "prompt")],
        provider=provider,
        stage="planner",
        timeout_seconds=timeout_seconds,
        breaker=breaker,
    )

    assert message.content == "ok"
    assert calls == [
        (
            [("system", "rules"), ("user", "prompt")],
            expected_kwargs,
        )
    ]
    assert len(dependencies.claims) == 1
    assert dependencies.finishes == [(dependencies.claims[0], ProviderDeliveryDisposition.COMPLETED)]
    assert (breaker.successes, breaker.failures) == (1, 0)


def test_clients_without_invoke_kwargs_preserve_the_historical_call_shape():
    dependencies = _Dependencies()
    runtime = _runtime(dependencies)
    breaker = _Breaker()
    calls = []

    class _LLM:
        def invoke(self, messages):
            calls.append(messages)
            return SimpleNamespace(content="ok")

    runtime.invoke(
        _LLM(),
        [("user", "prompt")],
        provider="openai",
        stage="summarizer",
        timeout_seconds=3.0,
        breaker=breaker,
    )

    assert calls == [[("user", "prompt")]]


def test_open_breaker_rejects_before_claiming_or_sending():
    dependencies = _Dependencies()
    runtime = _runtime(dependencies)

    with pytest.raises(ProviderExecutionError) as exc_info:
        runtime.invoke(
            object(),
            [("user", "prompt")],
            provider="nvidia",
            stage="router",
            timeout_seconds=2.0,
            breaker=_Breaker(allowed=False),
        )

    assert exc_info.value.disposition is ProviderDeliveryDisposition.REJECTED
    assert dependencies.claims == []
    assert dependencies.circuit_open == ["llm_nvidia"]


@pytest.mark.parametrize(
    ("disposition", "expected_successes", "expected_failures"),
    [
        (ProviderDeliveryDisposition.PERMANENT_FAILURE, 1, 0),
        (ProviderDeliveryDisposition.TIMED_OUT, 0, 1),
        (ProviderDeliveryDisposition.AMBIGUOUS, 0, 1),
    ],
)
def test_failed_calls_finalize_once_and_update_breaker_by_delivery_disposition(
    disposition, expected_successes, expected_failures
):
    dependencies = _Dependencies(disposition)
    runtime = _runtime(dependencies)
    breaker = _Breaker()

    class _LLM:
        def invoke(self, _messages, **_kwargs):
            raise TimeoutError("provider failed")

    with pytest.raises(ProviderExecutionError) as exc_info:
        runtime.invoke(
            _LLM(),
            [("user", "prompt")],
            provider="openai",
            stage="planner",
            timeout_seconds=4.0,
            breaker=breaker,
        )

    assert exc_info.value.disposition is disposition
    assert dependencies.finishes == [(dependencies.claims[0], disposition)]
    assert (breaker.successes, breaker.failures) == (
        expected_successes,
        expected_failures,
    )
