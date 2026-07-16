"""Contract and projection tests for the additive chat-gateway-v2 surface."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from contracts.chat_gateway_contract import (
    V2_CONTRACT_ARTIFACT_PATH,
    build_contract_document,
    load_committed_contract,
    serialize_contract_document,
)
from contracts.chat_gateway_v2 import build_chat_gateway_v2_response
from models import CHAT_GATEWAY_V2_CONTRACT_VERSION, TerminalOutcome


def _provenance() -> dict:
    return {
        "answer_path": "deterministic_render",
        "summary_source": "generic_renderer",
        "data_source": "tool",
        "provenance_refs": ["dataset:prices"],
        "tool_name": "get_prices",
        "used_sql": False,
        "retrieval_tier": "skip",
        "analyzer": {
            "authoritative": True,
            "query_type": "trend",
            "answer_kind": "timeseries",
            "confidence": 0.9,
        },
        "grounding_gate": {
            "passed": True,
            "reason": "grounded",
            "coverage": 1.0,
        },
    }


def _context() -> SimpleNamespace:
    first = {
        "type": "line",
        "data": [
            {"date": "2024-01-01", "Price": 10.0},
            {"date": "2024-02-01", "Price": 11.0},
        ],
        "metadata": {
            "title": "Price",
            "timeGrain": "month",
            "evidenceFilterApplied": True,
            "evidenceSource": "canonical_frame_filtered",
            "evidenceUnit": "GEL/MWh",
            "provenanceRefs": ["dataset:prices"],
            "seriesConfig": {
                "Price": {"type": "line", "yAxis": "left"},
            },
            "stage_timings_ms": {"must": "not leak"},
        },
    }
    second = {
        "type": "bar",
        "data": [{"entity": "A", "Share": 50.0}],
        "metadata": {
            "title": "Share",
            "evidenceFilterApplied": False,
            "evidenceSource": "canonical_frame_aligned",
            "evidenceUnit": "%",
            "provenanceRefs": ["dataset:shares"],
        },
    }
    return SimpleNamespace(
        charts=[first, second],
        chart_data=first["data"],
        chart_type=first["type"],
        chart_meta=first["metadata"],
        terminal_outcome=TerminalOutcome.DATA_ANSWER.value,
        is_conceptual=False,
        summary_confidence=0.9,
        summary_provenance_coverage=1.0,
        summary_provenance_gate_passed=True,
        summary_provenance_gate_reason="grounded",
        summary_citations=["dataset:prices"],
    )


def test_v2_committed_artifact_has_no_drift():
    assert V2_CONTRACT_ARTIFACT_PATH.exists()
    assert serialize_contract_document(
        contract_version=CHAT_GATEWAY_V2_CONTRACT_VERSION
    ) == V2_CONTRACT_ARTIFACT_PATH.read_text(encoding="utf-8")
    assert (
        load_committed_contract(CHAT_GATEWAY_V2_CONTRACT_VERSION)["contract_version"]
        == CHAT_GATEWAY_V2_CONTRACT_VERSION
    )


def test_v2_response_schema_is_closed_and_explicit():
    document = build_contract_document(CHAT_GATEWAY_V2_CONTRACT_VERSION)
    schema = document["schemas"]["ChatGatewayV2Response"]
    error_definitions = document["schemas"]["APIErrorResponse"]["$defs"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {
        "contract_version",
        "answer",
        "terminal_outcome",
        "charts",
        "provenance",
        "trust",
        "request_id",
        "session",
        "execution_time",
    }
    definitions = schema["$defs"]
    assert definitions["ChartSpec"]["additionalProperties"] is False
    assert definitions["ChartMetadata"]["additionalProperties"] is False
    assert definitions["ChartIdentity"]["additionalProperties"] is False
    assert error_definitions["APIErrorDetail"]["additionalProperties"] is False


def test_v2_contract_excludes_private_and_browser_authority_fields():
    serialized = json.dumps(
        build_contract_document(CHAT_GATEWAY_V2_CONTRACT_VERSION),
        sort_keys=True,
    )
    for forbidden in (
        "service_tier",
        "session_id",
        "session_token",
        "stage_timings_ms",
        "llm_telemetry",
        "total_tokens",
        "estimated_cost",
        "safe_sql",
        "raw_sql",
    ):
        assert forbidden not in serialized


def test_v2_projection_preserves_multiple_charts_and_public_identity():
    payload = build_chat_gateway_v2_response(
        _context(),
        answer="Grounded answer",
        request_id="req-v2-projection",
        execution_time=1.25,
        answer_provenance=_provenance(),
        session_continuity_available=True,
    ).model_dump(mode="json", by_alias=True, exclude_none=True)

    assert payload["contract_version"] == CHAT_GATEWAY_V2_CONTRACT_VERSION
    assert payload["terminal_outcome"] == "data_answer"
    assert len(payload["charts"]) == 2
    assert payload["charts"][0]["identity"] == {
        "filter_applied": True,
        "unit": "GEL/MWh",
        "period": {
            "grain": "month",
            "start": "2024-01-01",
            "end": "2024-02-01",
        },
        "evidence": {
            "source": "canonical_frame_filtered",
            "refs": ["dataset:prices"],
        },
    }
    assert "stage_timings_ms" not in payload["charts"][0]["metadata"]
    assert payload["provenance"]["provenance_refs"] == ["dataset:prices"]
    assert payload["trust"]["grounding_passed"] is True
    assert payload["session"] == {"continuity_available": True}


def test_v2_projection_rejects_an_invalid_chart_type():
    ctx = _context()
    ctx.charts[0]["type"] = "arbitrary-client-chart"

    with pytest.raises(ValidationError):
        build_chat_gateway_v2_response(
            ctx,
            answer="answer",
            request_id="req-invalid-chart",
            execution_time=0.1,
            answer_provenance=_provenance(),
            session_continuity_available=False,
        )
