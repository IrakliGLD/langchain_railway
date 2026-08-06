"""Report evidence-manifest construction and integrity tests."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agent import summarizer
from agent.report_evidence import (
    build_report_evidence_manifest,
    build_report_manifest_from_items,
    make_report_narrative_evidence_item,
)
from contracts.report_evidence import (
    REPORT_EVIDENCE_CONTENT_MAX_CHARS,
    REPORT_EVIDENCE_MANIFEST_MAX_BYTES,
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
    assert primary.unit_by_column["price_gel_mwh"] == "GEL/MWh"
    assert primary.provenance_refs == [
        "query:tool:abc123",
        "source:rows:def456",
    ]
    supporting = next(
        item
        for item in first.items
        if item.title == "Supporting evidence: correlation driver"
    )
    assert supporting.unit_by_column["hydro_gwh"] == "GWh"

    assert any(item.kind is ReportEvidenceKind.STATISTICS for item in first.items)
    assert any(item.kind is ReportEvidenceKind.KNOWLEDGE for item in first.items)
    assert any(item.kind is ReportEvidenceKind.LIMITATION for item in first.items)


def test_manifest_keeps_table_truncation_metadata_without_exposing_packaging_details():
    ctx = _context()
    ctx.provenance_rows = [(f"2026-{index:03d}", float(index), "row") for index in range(150)]

    manifest = build_report_evidence_manifest(ctx, max_rows_per_table=20)
    table = next(item for item in manifest.items if item.title == "Primary tabular evidence")

    assert table.total_row_count == 150
    assert len(table.rows) == 20
    assert table.truncated is True
    assert table.rows[0]["period"] == "2026-000"
    assert table.rows[-1]["period"] == "2026-149"
    assert [row["period"] for row in table.rows] != [
        f"2026-{index:03d}" for index in range(20)
    ]
    limitations = [
        item for item in manifest.items if item.kind is ReportEvidenceKind.LIMITATION
    ]
    assert [item.title for item in limitations] == ["Evidence boundary"]
    assert all("characters" not in item.content for item in limitations)
    assert all("manifest includes" not in item.content for item in limitations)


def test_manifest_prioritizes_periods_explicitly_requested_by_the_user():
    ctx = _context()
    ctx.query = "Explain the electricity price in 2020-07."
    ctx.provenance_rows = [
        (
            "2020-07" if index == 73 else f"period-{index:03d}",
            float(index),
            "row",
        )
        for index in range(150)
    ]

    manifest = build_report_evidence_manifest(ctx, max_rows_per_table=10)
    table = next(
        item
        for item in manifest.items
        if item.title == "Primary tabular evidence"
    )

    assert "2020-07" in {row["period"] for row in table.rows}
    assert table.rows[0]["period"] == "period-000"
    assert table.rows[-1]["period"] == "period-149"


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


def test_manifest_uses_registered_storage_units_for_raw_table_columns():
    ctx = QueryContext(query="Report generation.")
    ctx.provenance_cols = ["period", "quantity_tech"]
    ctx.provenance_rows = [("2026-01", 120.5)]
    ctx.provenance_source = "tool"

    manifest = build_report_evidence_manifest(ctx)
    primary = next(item for item in manifest.items if item.kind is ReportEvidenceKind.TABLE)

    assert primary.unit_by_column == {"quantity_tech": "thousand MWh"}


def test_long_table_sampling_preserves_extrema_and_largest_change():
    rows = [
        (f"2025-{index + 1:03d}", float(index))
        for index in range(300)
    ]
    rows[137] = (rows[137][0], -500.0)
    rows[138] = (rows[138][0], 900.0)
    ctx = QueryContext(query="Analyze the price trend.")
    ctx.provenance_cols = ["period", "p_bal_gel"]
    ctx.provenance_rows = rows
    ctx.provenance_source = "tool"

    manifest = build_report_evidence_manifest(ctx)
    primary = next(item for item in manifest.items if item.kind is ReportEvidenceKind.TABLE)
    sampled = {row["p_bal_gel"] for row in primary.rows}

    assert -500.0 in sampled
    assert 900.0 in sampled


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


def test_manifest_builder_skips_items_that_exceed_the_persistence_budget():
    large_items = [
        make_report_narrative_evidence_item(
            kind=ReportEvidenceKind.STATISTICS,
            title=f"Track statistics {index}",
            source="derived",
            content=f"{index:02d}" + "x" * REPORT_EVIDENCE_CONTENT_MAX_CHARS,
        )
        for index in range(20)
    ]
    limitation = make_report_narrative_evidence_item(
        kind=ReportEvidenceKind.LIMITATION,
        title="Evidence boundary",
        source="system",
        content="Only evidence retained in this manifest may support claims.",
    )

    manifest = build_report_manifest_from_items(
        "Explain the electricity price trend.",
        [*large_items, limitation],
    )

    assert len(manifest.model_dump_json().encode("utf-8")) <= (
        REPORT_EVIDENCE_MANIFEST_MAX_BYTES
    )
    assert len(manifest.items) < len(large_items) + 1
    assert limitation in manifest.items


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


def test_pipeline_narrative_items_carry_statistics_and_curated_knowledge():
    """The adaptive path must still receive what the standard pipeline computes.

    ``stats_hint`` is the verified computed-statistics channel and
    ``summary_domain_knowledge`` is the curated knowledge file content. The v1
    manifest carried both; the v2 collector path carried neither, so reports
    had no computed analysis and never saw the knowledge files at all.
    """

    from agent.report_evidence import build_report_narrative_items

    ctx = SimpleNamespace(
        stats_hint="Observed mean balancing price was 141.0 GEL/MWh.",
        summary_domain_knowledge="The balancing market settles hourly.",
        provenance_refs=["query:prices"],
    )

    items = build_report_narrative_items(ctx)
    by_kind = {item.kind: item for item in items}

    assert ReportEvidenceKind.STATISTICS in by_kind
    assert ReportEvidenceKind.KNOWLEDGE in by_kind
    statistics = by_kind[ReportEvidenceKind.STATISTICS]
    assert "141.0" in statistics.content
    assert statistics.provenance_refs == ["query:prices"]
    assert by_kind[ReportEvidenceKind.KNOWLEDGE].source == "curated_knowledge"


def test_pipeline_narrative_items_skip_empty_sources():
    from agent.report_evidence import build_report_narrative_items

    assert build_report_narrative_items(
        SimpleNamespace(
            stats_hint="   ",
            summary_domain_knowledge="",
            provenance_refs=[],
        )
    ) == []
    assert build_report_narrative_items(None) == []


def test_pipeline_narrative_items_reject_blocked_or_incomplete_contexts():
    from agent.report_evidence import build_report_narrative_items

    blocked = SimpleNamespace(
        terminal_outcome="clarification_required",
        missing_evidence_for_metrics=["mom_percent_change"],
        stats_hint="Observed mean balancing price was 141.0 GEL/MWh.",
        summary_domain_knowledge="The balancing market settles hourly.",
        provenance_refs=["query:prices"],
    )

    assert build_report_narrative_items(blocked) == []


def test_count_like_columns_receive_a_dimensionless_unit():
    from agent.report_evidence import _inferred_unit_by_column

    units = _inferred_unit_by_column(
        ["plant_count", "n_units", "unit_rank", "price_gel"]
    )

    assert units["plant_count"] == "count"
    assert units["n_units"] == "count"
    assert units["unit_rank"] == "rank"
    assert units["price_gel"] == "GEL/MWh"


def test_emitted_share_columns_carry_the_unit_they_actually_hold():
    """The two share aliases the SQL layer emits hold different scales.

    ``share`` is the 0-1 ratio from sql_executor's share expression;
    ``share_percent`` is ROUND(x / y * 100, 2) from the aggregation examples.
    A claim on a column with no unit, or with the wrong scale, cannot be
    verified at all -- which is how report job 1779c440 failed.
    """

    from agent.report_evidence import _inferred_unit_by_column

    units = _inferred_unit_by_column(
        ["period", "segment", "share", "share_percent", "number_of_months"]
    )

    assert units["share"] == "ratio"
    assert units["share_percent"] == "%"
    assert units["number_of_months"] == "count"
    # Label columns stay unit-less: they carry no magnitude to claim.
    assert "segment" not in units
    assert "period" not in units


def test_statistics_narrative_keeps_far_more_than_a_passage_sized_clip():
    """A 59 KB stats_hint must not be truncated to a passage's worth.

    Report job 22237205 produced 59,131 characters of computed statistics --
    21 column aggregates, 4 correlation targets and why-context -- of which a
    6,000-character cap kept 10%. The correlations sit late in the text, so the
    clip removed exactly the analytical content reports were missing.
    """

    from agent.report_evidence import build_report_narrative_items
    from contracts.report_evidence import REPORT_EVIDENCE_CONTENT_MAX_CHARS

    # Sized like the 59,131-character stats_hint from job 22237205.
    long_stats = "Observed mean was 141.0 GEL/MWh. " * 1800
    assert len(long_stats) > 4 * 6_000, "must dwarf the old passage-sized cap"

    items = build_report_narrative_items(
        SimpleNamespace(
            stats_hint=long_stats,
            summary_domain_knowledge="",
            provenance_refs=[],
            query="",
        )
    )

    statistics = next(
        item for item in items if item.kind is ReportEvidenceKind.STATISTICS
    )
    # A real stats_hint must survive whole, not be clipped to a passage.
    assert len(statistics.content) == len(long_stats.strip())
    assert REPORT_EVIDENCE_CONTENT_MAX_CHARS >= 59_131


def test_report_mode_selects_curated_knowledge_without_the_summarizer(
    monkeypatch,
):
    """Report mode bypasses the summarizer, where knowledge is normally set."""

    from agent import report_evidence

    captured = {}

    def fake_selector(query, *, use_cache, preferred_topics):
        captured.update(
            query=query, use_cache=use_cache, preferred_topics=preferred_topics
        )
        return "The balancing market settles hourly."

    monkeypatch.setattr(
        "core.llm.get_relevant_domain_knowledge", fake_selector
    )

    items = report_evidence.build_report_narrative_items(
        SimpleNamespace(
            stats_hint="",
            summary_domain_knowledge="",
            provenance_refs=[],
            query="Explain the generation mix and prices.",
            question_analysis=None,
            question_analysis_source="",
        )
    )

    knowledge = next(
        item for item in items if item.kind is ReportEvidenceKind.KNOWLEDGE
    )
    assert knowledge.source == "curated_knowledge"
    assert captured["query"] == "Explain the generation mix and prices."
    assert captured["use_cache"] is False


def test_curated_knowledge_failure_never_fails_a_report(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("knowledge unavailable")

    monkeypatch.setattr("core.llm.get_relevant_domain_knowledge", boom)

    from agent import report_evidence

    assert report_evidence.build_report_narrative_items(
        SimpleNamespace(
            stats_hint="",
            summary_domain_knowledge="",
            provenance_refs=[],
            query="Explain prices.",
            question_analysis=None,
            question_analysis_source="",
        )
    ) == []
