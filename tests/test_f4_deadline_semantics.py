import time
from types import SimpleNamespace

import pytest

from core import db_gateway
from core import llm as llm_core
from models import QueryContext
from utils.provider_attempts import (
    ProviderDeliveryDisposition,
    ProviderExecutionError,
    reset_provider_attempts_for_tests,
)
from utils.request_deadline import (
    RequestDeadline,
    RequestDeadlineExceeded,
    bind_request_execution_scope,
)


class _Breaker:
    def __init__(self):
        self.failures = 0
        self.successes = 0

    def allow_request(self):
        return True, "closed"

    def record_failure(self):
        self.failures += 1

    def record_success(self):
        self.successes += 1


class _Connection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        self.calls.append((str(statement), params or {}))
        return SimpleNamespace()


class _Manager:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, *_args):
        return False


class _Engine:
    def __init__(self):
        self.connection = _Connection()
        self.connect_count = 0

    def connect(self):
        self.connect_count += 1
        return _Manager(self.connection)

    def begin(self):
        self.connect_count += 1
        return _Manager(self.connection)


@pytest.fixture(autouse=True)
def _reset_attempts():
    reset_provider_attempts_for_tests()
    yield
    reset_provider_attempts_for_tests()


def _deadline(milliseconds=10_000):
    return RequestDeadline.from_budget_ms(
        budget_ms=milliseconds,
        now_monotonic=time.monotonic(),
        source="test",
    )


def test_db_statement_timeout_uses_remaining_budget_minus_cleanup(monkeypatch):
    breaker = _Breaker()
    engine = _Engine()
    monkeypatch.setattr(db_gateway, "db_circuit_breaker", breaker)
    monkeypatch.setattr(db_gateway, "DB_STATEMENT_TIMEOUT_MS", 30_000)
    monkeypatch.setattr(db_gateway, "REQUEST_CLEANUP_ALLOWANCE_MS", 3_000)

    with bind_request_execution_scope(deadline=_deadline(10_000), request_id="req-db-budget", actor_id="actor-db"):
        with db_gateway.database_connection(engine, operation="test") as connection:
            assert connection is engine.connection

    sql, params = engine.connection.calls[0]
    assert "set_config('statement_timeout'" in sql
    assert 6_000 <= int(params["timeout_ms"]) <= 7_000
    assert breaker.successes == 1


def test_db_read_only_is_applied_before_statement_timeout(monkeypatch):
    engine = _Engine()
    monkeypatch.setattr(db_gateway, "db_circuit_breaker", _Breaker())

    with bind_request_execution_scope(deadline=_deadline(), request_id="req-db-order", actor_id="actor-db"):
        with db_gateway.database_connection(engine, operation="test", read_only=True):
            pass

    assert str(engine.connection.calls[0][0]) == "SET TRANSACTION READ ONLY"
    assert "set_config('statement_timeout'" in engine.connection.calls[1][0]

def test_db_call_does_not_checkout_when_cleanup_budget_is_unavailable(monkeypatch):
    engine = _Engine()
    breaker = _Breaker()
    monkeypatch.setattr(db_gateway, "db_circuit_breaker", breaker)
    monkeypatch.setattr(db_gateway, "REQUEST_CLEANUP_ALLOWANCE_MS", 3_000)

    with bind_request_execution_scope(deadline=_deadline(2_000), request_id="req-db-expired", actor_id="actor-db"):
        with pytest.raises(RequestDeadlineExceeded):
            with db_gateway.database_connection(engine, operation="test"):
                pass

    assert engine.connect_count == 0
    assert breaker.successes == 0
    assert breaker.failures == 0


def test_provider_receives_bounded_timeout_and_duplicate_stage_is_blocked(monkeypatch):
    captured = []
    breaker = _Breaker()

    class _LLM:
        def invoke(self, _messages, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(content="ok")

    monkeypatch.setattr(llm_core, "get_llm_breaker", lambda _provider: breaker)
    with bind_request_execution_scope(deadline=_deadline(10_000), request_id="req-provider-once", actor_id="actor-1"):
        result = llm_core._invoke_with_resilience(_LLM(), [("user", "not recorded")], "gpt-4o", attempt_stage="router")
        assert result.content == "ok"
        with pytest.raises(ProviderExecutionError) as exc:
            llm_core._invoke_with_resilience(_LLM(), [("user", "not recorded")], "gpt-4o", attempt_stage="router")

    assert len(captured) == 1
    assert 0 < captured[0]["timeout"] <= 7.0
    assert exc.value.disposition == ProviderDeliveryDisposition.COMPLETED


def test_lost_provider_response_is_ambiguous_and_never_falls_back(monkeypatch):
    fallback_calls = []

    class _Primary:
        def invoke(self, _messages, **_kwargs):
            raise TimeoutError("response lost after send")

    monkeypatch.setattr(llm_core, "get_llm_breaker", lambda _provider: _Breaker())
    monkeypatch.setattr(llm_core, "_should_fallback_to_openai", lambda: True)
    monkeypatch.setattr(llm_core, "make_openai", lambda: fallback_calls.append(True))

    with bind_request_execution_scope(deadline=_deadline(), request_id="req-ambiguous", actor_id="actor-2"):
        with pytest.raises(ProviderExecutionError) as exc:
            llm_core._invoke_with_openai_fallback(
                lambda: _Primary(),
                "gemini-2.5-flash",
                [("user", "not recorded")],
                llm_start=time.time(),
                label="question analyzer",
            )

    assert exc.value.disposition == ProviderDeliveryDisposition.AMBIGUOUS
    assert fallback_calls == []


@pytest.mark.parametrize("status_code", [400, 401, 403, 422])
def test_permanent_provider_failures_are_never_retried(monkeypatch, status_code):
    fallback_calls = []

    class _PermanentError(RuntimeError):
        pass

    error = _PermanentError("rejected")
    error.status_code = status_code

    class _Primary:
        def invoke(self, _messages, **_kwargs):
            raise error

    monkeypatch.setattr(llm_core, "get_llm_breaker", lambda _provider: _Breaker())
    monkeypatch.setattr(llm_core, "_should_fallback_to_openai", lambda: True)
    monkeypatch.setattr(llm_core, "make_openai", lambda: fallback_calls.append(True))

    with bind_request_execution_scope(
        deadline=_deadline(), request_id=f"req-permanent-{status_code}", actor_id="actor-3"
    ):
        with pytest.raises(ProviderExecutionError) as exc:
            llm_core._invoke_with_openai_fallback(
                lambda: _Primary(),
                "gemini-2.5-flash",
                [("user", "not recorded")],
                llm_start=time.time(),
                label="planner",
            )

    assert exc.value.disposition == ProviderDeliveryDisposition.PERMANENT_FAILURE
    assert fallback_calls == []


def test_explicit_rate_limit_rejection_can_use_one_bounded_fallback(monkeypatch):
    class _Rejected(RuntimeError):
        status_code = 429

    class _Primary:
        def invoke(self, _messages, **_kwargs):
            raise _Rejected("not accepted")

    class _Fallback:
        def invoke(self, _messages, **kwargs):
            assert 0 < kwargs["timeout"] <= 7.0
            return SimpleNamespace(content="fallback", usage_metadata={})

    monkeypatch.setattr(llm_core, "get_llm_breaker", lambda _provider: _Breaker())
    monkeypatch.setattr(llm_core, "_should_fallback_to_openai", lambda: True)
    monkeypatch.setattr(llm_core, "make_openai", lambda: _Fallback())
    monkeypatch.setattr(llm_core.random, "uniform", lambda _low, _high: 0.0)

    with bind_request_execution_scope(deadline=_deadline(), request_id="req-safe-fallback", actor_id="actor-4"):
        result = llm_core._invoke_with_openai_fallback(
            lambda: _Primary(),
            "gemini-2.5-flash",
            [("user", "not recorded")],
            llm_start=time.time(),
            label="planner",
        )

    assert result.content == "fallback"


@pytest.mark.parametrize(
    ("provider", "configured_seconds", "expected_timeout"),
    [
        ("gemini", 5.0, 5_000),
        ("openai", 7.0, 7.0),
        ("nvidia", 9.0, 9.0),
    ],
)
def test_each_provider_has_an_independent_native_timeout(monkeypatch, provider, configured_seconds, expected_timeout):
    captured = {}

    class _LLM:
        def invoke(self, _messages, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content="ok")

    monkeypatch.setattr(llm_core, "_provider_from_model_name", lambda _model: provider)
    monkeypatch.setattr(llm_core, "get_llm_breaker", lambda _provider: _Breaker())
    monkeypatch.setattr(
        llm_core,
        {"gemini": "GEMINI_TIMEOUT_SECONDS", "openai": "OPENAI_TIMEOUT_SECONDS", "nvidia": "NVIDIA_TIMEOUT_SECONDS"}[
            provider
        ],
        configured_seconds,
    )
    with bind_request_execution_scope(
        deadline=_deadline(20_000),
        request_id=f"req-timeout-{provider}",
        actor_id="actor-timeouts",
    ):
        llm_core._invoke_with_resilience(_LLM(), [("user", "not recorded")], "test-model", attempt_stage="timeout_test")

    assert captured["timeout"] == expected_timeout
    if provider == "gemini":
        assert captured["max_retries"] == 1
