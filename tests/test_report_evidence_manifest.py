"""Report evidence-manifest construction and integrity tests."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from agent import summarizer
from agent.report_evidence import build_report_evidence_manifest
from contracts.report_evidence import (
    REPORT_EVIDENCE_MANIFEST_VERSION,
    ReportEvidenceItem,
    ReportEvidenceKind,
    ReportEvidenceManifest,
)
from core.llm import SummaryEnvelope
from models import QueryContext


def _context() -> QueryContext:
    ctx = QueryContext(query="Explain the electricity price trend.")
    ctx.provenance_cols = ["period", "price_gel_mwh", "note"]
    ctx.provenance_rows = [
        ("2026-01", 120.5, "observed"),
        ("2026-02", float("nan"), "missing"),
        ("2026-03", 131.25, "observed"),
    ]
    ctx.provenance_source = "tool"
    ctx.provenance_refs = ["query:tool:abc123", "source:rows:def456"]
    ctx.stats_hint = "Average observed price: 125.88 GEL/MWh."
    ctx.summary_domain_knowledge = (
        "Balancing prices reflect the cost of resolving real-time system imbalances."
    )
    ctx.evidence_collected = {
        "correlation_driver": {
            "tool": "get_generation",
            "cols": ["period", "hydro_gwh"],
            "rows": [("2026-01", 410.0), ("2026-02", 390.0)],
            "provenance_refs": ["query:tool:driver123"],
        }
    }
    return ctx


def test_manifest_is_deterministic_bounded_and_carries_exact_verified_evidence():
    first = build_report_evidence_manifest(_context())
    second = build_report_evidence_manifest(_context())

    assert first == second
    assert first.contract_version == REPORT_EVIDENCE_MANIFEST_VERSION
    assert first.manifest_id == second.manifest_id
    assert first.query_digest == second.query_digest
    assert len(first.items) >= 4

    primary = next(item for item in first.items if item.title == "Primary tabular evidence")
    assert primary.kind is ReportEvidenceKind.TABLE
    assert primary.columns == ["period", "price_gel_mwh", "note"]
    assert primary.rows[0]["price_gel_mwh"] == 120.5
    assert primary.rows[1]["price_gel_mwh"] is None
    assert primary.provenance_refs == [
        "query:tool:abc123",
        "source:rows:def456",
    ]

    assert any(item.kind is ReportEvidenceKind.STATISTICS for item in first.items)
    assert any(item.kind is ReportEvidenceKind.KNOWLEDGE for item in first.items)
    assert any(item.kind is ReportEvidenceKind.LIMITATION for item in first.items)


def test_manifest_records_truncation_as_a_limitation():
    ctx = _context()
    ctx.provenance_rows = [(f"2026-{index:03d}", float(index), "row") for index in range(150)]

    manifest = build_report_evidence_manifest(ctx, max_rows_per_table=20)
    table = next(item for item in manifest.items if item.title == "Primary tabular evidence")

    assert table.total_row_count == 150
    assert len(table.rows) == 20
    assert table.truncated is True
    assert any(
        "150" in item.content and "20" in item.content
        for item in manifest.items
        if item.kind is ReportEvidenceKind.LIMITATION
    )


def test_manifest_normalizes_runtime_source_labels_and_dict_rows():
    ctx = _context()
    ctx.provenance_source = "SQL Fallback / Primary"
    ctx.provenance_rows = [
        {"period": "2026-01", "price_gel_mwh": 120.5, "note": "observed"}
    ]

    manifest = build_report_evidence_manifest(ctx)
    primary = next(item for item in manifest.items if item.title == "Primary tabular evidence")

    assert primary.source == "sql_fallback_primary"
    assert primary.rows[0]["price_gel_mwh"] == 120.5


def test_evidence_item_shape_is_closed_and_kind_consistent():
    table = {
        "evidence_ref": "evidence:table:" + "a" * 16,
        "kind": "table",
        "title": "Observed data",
        "source": "tool",
        "provenance_refs": ["query:tool:abc"],
        "columns": ["period", "value"],
        "rows": [{"period": "2026-01", "value": 1.0}],
        "content": "",
        "unit_by_column": {"value": "MWh"},
        "total_row_count": 1,
        "truncated": False,
    }
    item = ReportEvidenceItem.model_validate(table)
    assert item.kind is ReportEvidenceKind.TABLE
    assert ReportEvidenceItem.model_json_schema()["additionalProperties"] is False

    invalid = dict(table)
    invalid["content"] = "table items cannot mix narrative evidence"
    with pytest.raises(ValidationError, match="Table evidence"):
        ReportEvidenceItem.model_validate(invalid)

    invalid = dict(table)
    invalid["rows"] = [{"period": "2026-01", "value": math.inf}]
    with pytest.raises(ValidationError):
        ReportEvidenceItem.model_validate(invalid)


def test_manifest_rejects_duplicate_refs_and_oversized_serialized_content():
    item = {
        "evidence_ref": "evidence:knowledge:" + "b" * 16,
        "kind": "knowledge",
        "title": "Knowledge",
        "source": "vector",
        "provenance_refs": [],
        "columns": [],
        "rows": [],
        "content": "Evidence text.",
        "unit_by_column": {},
        "total_row_count": 0,
        "truncated": False,
    }
    payload = {
        "contract_version": REPORT_EVIDENCE_MANIFEST_VERSION,
        "manifest_id": "manifest:" + "c" * 32,
        "query_digest": "d" * 64,
        "items": [item, dict(item)],
    }
    with pytest.raises(ValidationError, match="unique evidence_ref"):
        ReportEvidenceManifest.model_validate(payload)


def test_conceptual_answer_exposes_curated_knowledge_to_report_manifest(
    monkeypatch,
):
    curated_knowledge = (
        '{"market_structure":"GENEX operates Georgian day-ahead and '
        'intraday electricity markets."}'
    )
    monkeypatch.setattr(
        summarizer,
        "get_relevant_domain_knowledge",
        lambda *_args, **_kwargs: curated_knowledge,
    )
    monkeypatch.setattr(
        summarizer,
        "llm_summarize_structured",
        lambda *_args, **_kwargs: SummaryEnvelope(
            answer="GENEX is part of the Georgian electricity market model.",
            claims=["GENEX is part of the Georgian electricity market model."],
            citations=["domain_knowledge"],
            confidence=0.9,
        ),
    )
    ctx = QueryContext(
        query="Explain the Georgian electricity market model.",
        lang_instruction="Respond in English.",
    )

    summarizer.answer_conceptual(ctx)
    manifest = build_report_evidence_manifest(ctx)

    knowledge_item = next(
        item
        for item in manifest.items
        if item.kind is ReportEvidenceKind.KNOWLEDGE
    )
    assert ctx.summary_domain_knowledge == curated_knowledge
    assert knowledge_item.content == curated_knowledge
