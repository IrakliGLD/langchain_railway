"""P7.A privacy and public-observability contract tests."""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.public_metadata import (  # noqa: E402
    build_public_response_metadata,
    project_public_charts,
)
from utils.privacy_logging import (  # noqa: E402
    PrivacyLogFilter,
    hash_private_identifier,
    redact_log_message,
    sanitize_observability_value,
)
from utils.trace_logging import trace_detail  # noqa: E402


def test_private_identifiers_are_keyed_and_request_correlation_stays_public():
    raw_actor = "actor-73d96e32-806c-44ac-a8c5-6ed79ef0d064"
    hashed = hash_private_identifier(raw_actor, namespace="actor")

    assert raw_actor not in hashed
    assert hashed.startswith("hmac-sha256:")
    assert sanitize_observability_value("actor_id", raw_actor) == hashed
    assert sanitize_observability_value("request_id", "req-safe.123:attempt_1") == "req-safe.123:attempt_1"


def test_final_log_filter_redacts_content_pii_tokens_uuids_and_exception_messages():
    raw_query = "show account alice@example.com with token eyJhbGciOiJIUzI1NiJ9.private.signature"
    record = logging.LogRecord(
        "Enai",
        logging.INFO,
        __file__,
        1,
        "query=%s actor=73d96e32-806c-44ac-a8c5-6ed79ef0d064 error=%s",
        (raw_query, RuntimeError("provider echoed the raw prompt")),
        None,
    )

    assert PrivacyLogFilter().filter(record) is True
    rendered = record.getMessage()
    assert raw_query not in rendered
    assert "alice@example.com" not in rendered
    assert "eyJhbGci" not in rendered
    assert "73d96e32-806c-44ac-a8c5-6ed79ef0d064" not in rendered
    assert "provider echoed the raw prompt" not in rendered

    error_record = logging.LogRecord(
        "Enai",
        logging.ERROR,
        __file__,
        1,
        "provider failure: %s",
        (RuntimeError("provider echoed the raw prompt"),),
        None,
    )
    assert PrivacyLogFilter().filter(error_record) is True
    assert error_record.getMessage() == "provider failure: RuntimeError"
    assert "redacted" in redact_log_message(f"answer: {raw_query}")


def test_trace_detail_never_emits_raw_query_answer_sql_email_token_or_session(caplog):
    ctx = SimpleNamespace(trace_id="span-safe-1", session_id="session-private-1")
    raw_values = {
        "query": "email alice@example.com and show the private tariff",
        "answer_preview": "The private answer is 42",
        "sql": "select * from private_relation where email='alice@example.com'",
        "token": "eyJhbGciOiJIUzI1NiJ9.private.signature",
    }

    with caplog.at_level(logging.INFO, logger="Enai"):
        trace_detail(logging.getLogger("Enai"), ctx, "stage_test", "privacy_probe", **raw_values)

    record = next(record for record in caplog.records if str(record.message).startswith("TRACE_DETAIL "))
    serialized = record.getMessage()
    payload = json.loads(serialized.split("TRACE_DETAIL ", 1)[1])

    for raw in [*raw_values.values(), ctx.session_id, "alice@example.com"]:
        assert raw not in serialized
    assert payload["trace_id"] == "span-safe-1"
    assert payload["session_id"].startswith("hmac-sha256:")
    assert payload["stage"] == "stage_test"
    assert payload["event"] == "privacy_probe"


def test_public_response_metadata_is_allow_listed_and_protected_telemetry_is_dropped():
    ctx = SimpleNamespace(
        chart_meta={
            "title": "Observed prices",
            "labels": ["Balancing price"],
            "axisMode": "single",
            "internal_debug_dump": "must not escape",
        },
        summary_citations=["data_preview"],
        summary_confidence=0.91,
        summary_provenance_coverage=1.0,
        summary_provenance_gate_passed=True,
        summary_provenance_gate_reason="passed",
        provenance_query_hash="query-hash",
        provenance_source="sql",
        provenance_refs=["sql:query-hash"],
    )

    metadata = build_public_response_metadata(
        ctx,
        request_id="req-1",
        trace_id="span-1",
        metric_unit_registry_version="1.0.0",
        request_deadline={"budget_ms": 1000, "remaining_ms": 500, "retry_owner": "backend"},
        answer_provenance={"answer_path": "deterministic_render"},
        protected_telemetry={
            "llm_telemetry": {"total_tokens": 999, "estimated_cost_usd": 3.14},
            "stage_timings_ms": {"stage_1": 12.0},
            "summary_claims": ["private answer preview"],
            "session_bound_history_turns": 3,
        },
    )

    assert metadata["title"] == "Observed prices"
    assert metadata["request_id"] == "req-1"
    assert metadata["trace_id"] == "span-1"
    assert metadata["metric_unit_registry_version"] == "1.0.0"
    assert "internal_debug_dump" not in metadata
    assert "llm_telemetry" not in metadata
    assert "stage_timings_ms" not in metadata
    assert "summary_claims" not in metadata
    assert "session_bound_history_turns" not in metadata


def test_every_chart_metadata_block_uses_the_same_allow_list():
    charts = project_public_charts(
        [
            {
                "data": [{"date": "2026-01", "value": 1.0}],
                "type": "line",
                "metadata": {"title": "Safe", "llm_prompt": "private prompt"},
                "unexpected": "drop me",
            }
        ]
    )

    assert charts == [
        {
            "data": [{"date": "2026-01", "value": 1.0}],
            "type": "line",
            "metadata": {"title": "Safe"},
        }
    ]
