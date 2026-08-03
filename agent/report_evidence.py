"""Build a deterministic report evidence manifest from verified pipeline state."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable, Sequence

from agent.metric_units import METRIC_UNITS
from config_metrics.metric_units import metric_value_unit
from contracts.report_evidence import (
    REPORT_EVIDENCE_CONTENT_MAX_CHARS,
    REPORT_EVIDENCE_MANIFEST_VERSION,
    ReportEvidenceItem,
    ReportEvidenceKind,
    ReportEvidenceManifest,
    ReportKnowledgeEvidenceRole,
)
from models import QueryContext
from utils.coverage_sampling import coverage_priority_indices

_QUERY_PERIOD_PATTERN = re.compile(
    r"(?<!\w)\d{4}(?:[-/](?:[Qq][1-4]|\d{1,2})(?:[-/]\d{1,2})?)?(?!\w)"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: Any, *, length: int = 16) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:length]


def _json_scalar(value: Any) -> str | int | float | bool | None:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:
            value = str(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)[:1000]


def _normalize_source(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9_:]+", "_", str(value or "").strip().lower())
    normalized = normalized.strip("_:")[:64]
    if not normalized or not normalized[0].isalpha():
        return "pipeline"
    return normalized


def _normalize_table(
    columns: Sequence[Any],
    rows: Iterable[Iterable[Any]],
    *,
    max_rows: int,
    max_columns: int,
    relevant_tokens: set[str],
) -> tuple[list[str], list[dict[str, Any]], int, bool]:
    normalized_columns = [str(column)[:128] for column in list(columns)[:max_columns]]
    materialized_rows = list(rows)
    preferred_indices = [
        index
        for index, row in enumerate(materialized_rows)
        if relevant_tokens
        and any(
            token in str(value).casefold().replace("/", "-")
            for token in relevant_tokens
            for value in (
                row.values()
                if isinstance(row, dict)
                else row
            )
        )
    ]
    if len(preferred_indices) > max_rows and max_rows > 1:
        preferred_indices = [
            preferred_indices[
                round(index * (len(preferred_indices) - 1) / (max_rows - 1))
            ]
            for index in range(max_rows)
        ]
    selected_indices = sorted(
        index
        for _, index in zip(
            range(max_rows),
            coverage_priority_indices(
                len(materialized_rows),
                preferred_indices=preferred_indices,
            ),
            strict=False,
        )
    )
    normalized_rows = []
    for row_index in selected_indices:
        row = materialized_rows[row_index]
        if isinstance(row, dict):
            normalized_rows.append(
                {
                    column: _json_scalar(row.get(column))
                    for column in normalized_columns
                }
            )
        else:
            normalized_rows.append(
                {
                    column: _json_scalar(row[index] if index < len(row) else None)
                    for index, column in enumerate(normalized_columns)
                }
            )
    total = len(materialized_rows)
    return normalized_columns, normalized_rows, total, total > len(normalized_rows)


def _inferred_unit_by_column(columns: Sequence[str]) -> dict[str, str]:
    """Expose deterministic units already encoded by canonical column names."""

    units: dict[str, str] = {}
    for column in columns:
        normalized = str(column or "").strip().lower()
        if not normalized:
            continue
        # A percent-scaled share must be resolved before the registry, whose
        # ``share_*`` pattern would label it "ratio" (0-1). The aggregation
        # examples emit ROUND(x / y * 100, 2) AS share_percent, so that label
        # would make every correct percentage claim fail verification.
        if normalized.endswith("_percent") or normalized.endswith("_pct"):
            units[column] = "%"
            continue
        registered = METRIC_UNITS.find_for_source_metric(normalized)
        if registered is not None:
            units[column] = registered.storage_unit
            continue
        # sql_executor emits a bare "share" for its 0-1 share expression. Both
        # the registry pattern (share_*) and metric_value_unit's
        # startswith("share_") need a suffix, so the bare name resolved to
        # nothing and no claim on it could be verified.
        if normalized == "share":
            units[column] = "ratio"
            continue
        if (
            "price" in normalized
            or normalized.startswith("p_")
            or "tariff" in normalized
        ) and ("_gel" in normalized or normalized.endswith("gel")):
            units[column] = "GEL/MWh"
            continue
        if (
            "price" in normalized
            or normalized.startswith("p_")
            or "tariff" in normalized
        ) and ("_usd" in normalized or normalized.endswith("usd")):
            units[column] = "USD/MWh"
            continue
        suffix_units = {
            "_gwh": "GWh",
            "_mwh": "MWh",
            "_mw": "MW",
            "_kw": "kW",
            "_kwh": "kWh",
        }
        suffix_unit = next(
            (
                unit
                for suffix, unit in suffix_units.items()
                if normalized.endswith(suffix)
            ),
            "",
        )
        if suffix_unit:
            units[column] = suffix_unit
            continue
        # Dimensionless columns still need a declared unit: a claim on a column
        # without one cannot be verified at all.
        if normalized.endswith("_rank") or normalized == "rank":
            units[column] = "rank"
            continue
        if normalized.endswith("_index") or normalized == "index":
            units[column] = "index"
            continue
        if (
            normalized.endswith("_count")
            or normalized.startswith("count_")
            or normalized.startswith("n_")
            or normalized.endswith("_units")
            # COUNT(*) AS number_of_months in the aggregation examples.
            or normalized.startswith("number_of_")
        ):
            units[column] = "count"
            continue
        canonical_unit = metric_value_unit(normalized)
        if canonical_unit != "value":
            units[column] = canonical_unit
    return units


def _make_item(
    *,
    kind: ReportEvidenceKind,
    title: str,
    source: str,
    knowledge_role: ReportKnowledgeEvidenceRole | None = None,
    provenance_refs: list[str] | None = None,
    columns: list[str] | None = None,
    rows: list[dict[str, Any]] | None = None,
    content: str = "",
    unit_by_column: dict[str, str] | None = None,
    total_row_count: int = 0,
    truncated: bool = False,
) -> ReportEvidenceItem:
    body = {
        "kind": kind.value,
        "title": title,
        "source": _normalize_source(source),
        "provenance_refs": list(dict.fromkeys(provenance_refs or [])),
        "columns": columns or [],
        "rows": rows or [],
        "content": content,
        "unit_by_column": unit_by_column or {},
        "total_row_count": total_row_count,
        "truncated": truncated,
    }
    if knowledge_role is not None:
        body["knowledge_role"] = knowledge_role.value
    body["evidence_ref"] = f"evidence:{kind.value}:{_digest(body)}"
    return ReportEvidenceItem.model_validate(body)


def make_report_table_evidence_item(
    *,
    query: str,
    title: str,
    source: str,
    columns: Sequence[Any],
    rows: Iterable[Iterable[Any]],
    provenance_refs: list[str] | None = None,
    max_rows: int = 100,
    max_columns: int = 24,
) -> ReportEvidenceItem | None:
    """Create one bounded deterministic table item for report evidence."""

    relevant_tokens = {
        match.group(0).casefold().replace("/", "-")
        for match in _QUERY_PERIOD_PATTERN.finditer(str(query or ""))
    }
    normalized_columns, normalized_rows, total_rows, truncated = (
        _normalize_table(
            columns,
            rows,
            max_rows=max_rows,
            max_columns=max_columns,
            relevant_tokens=relevant_tokens,
        )
    )
    if not normalized_columns or not normalized_rows:
        return None
    return _make_item(
        kind=ReportEvidenceKind.TABLE,
        title=title,
        source=source,
        provenance_refs=provenance_refs,
        columns=normalized_columns,
        rows=normalized_rows,
        unit_by_column=_inferred_unit_by_column(normalized_columns),
        total_row_count=total_rows,
        truncated=truncated,
    )


def _curated_knowledge_for_context(ctx: Any) -> str:
    """Select curated knowledge-file content for one report context.

    Mirrors the standard summarizer: the analyzer's ranked topics steer the
    selection when it ran, otherwise the query's own topic match is used.
    Returns "" on any failure — knowledge enriches a report, never fails one.
    """

    query = str(getattr(ctx, "query", "") or "").strip()
    if not query:
        return ""
    preferred_topics: list[str] | None = None
    analysis = getattr(ctx, "question_analysis", None)
    if (
        analysis is not None
        and getattr(ctx, "question_analysis_source", "") == "llm_active"
    ):
        try:
            ranked = sorted(
                analysis.knowledge.candidate_topics,
                key=lambda candidate: candidate.score,
                reverse=True,
            )
            preferred_topics = [
                candidate.name.value
                if hasattr(candidate.name, "value")
                else str(candidate.name)
                for candidate in ranked
                if candidate.score >= 0.25
            ][:3] or None
        except Exception:
            preferred_topics = None
    try:
        from core.llm import get_relevant_domain_knowledge

        return str(
            get_relevant_domain_knowledge(
                query,
                use_cache=False,
                preferred_topics=preferred_topics,
            )
            or ""
        ).strip()
    except Exception:
        return ""


def build_report_narrative_items(ctx: Any) -> list[ReportEvidenceItem]:
    """Derive the narrative evidence the standard query pipeline computes.

    ``stats_hint`` is the verified computed-statistics channel and
    ``summary_domain_knowledge`` holds the curated knowledge-file content. Both
    reach the report as narrative items, whose facts ground every sentence of a
    paragraph that cites them — which is how a report states an average or a
    year-on-year change without binding it to a single table coordinate.

    Returns an empty list rather than raising: this evidence enriches a report
    and must never be able to fail one.
    """

    if report_pipeline_context_block_reason(ctx):
        return []
    items: list[ReportEvidenceItem] = []
    statistics = str(getattr(ctx, "stats_hint", "") or "").strip()
    if statistics:
        items.append(
            make_report_narrative_evidence_item(
                kind=ReportEvidenceKind.STATISTICS,
                title="Verified statistics",
                source="derived",
                provenance_refs=list(
                    getattr(ctx, "provenance_refs", None) or []
                ),
                content=statistics,
            )
        )
    knowledge = str(
        getattr(ctx, "summary_domain_knowledge", "") or ""
    ).strip()
    if not knowledge:
        # summary_domain_knowledge is populated inside the summarizer, and
        # report mode bypasses that stage ("Stage 4 bypassed | report evidence
        # ready"), so the curated knowledge files never reached a report.
        # Select them here with the same broad selector standard mode uses,
        # steered by the analyzer's topics when it ran.
        knowledge = _curated_knowledge_for_context(ctx)
    if knowledge:
        items.append(
            make_report_narrative_evidence_item(
                kind=ReportEvidenceKind.KNOWLEDGE,
                title="Curated domain knowledge",
                source="curated_knowledge",
                content=knowledge,
            )
        )
    return items


def report_pipeline_context_block_reason(ctx: Any) -> str:
    """Return a stable reason when pipeline output is unsafe for reports."""
    if ctx is None:
        return "missing_context"
    if list(getattr(ctx, "missing_evidence_for_metrics", None) or []):
        return "missing_derived_evidence"
    terminal_outcome = str(
        getattr(ctx, "terminal_outcome", "") or ""
    ).strip().lower()
    if terminal_outcome and terminal_outcome not in {
        "data_answer",
        "conceptual_answer",
    }:
        return f"terminal_{re.sub(r'[^a-z0-9_]+', '_', terminal_outcome)[:48]}"
    return ""


def make_report_narrative_evidence_item(
    *,
    kind: ReportEvidenceKind,
    title: str,
    source: str,
    content: str,
    provenance_refs: list[str] | None = None,
    knowledge_role: ReportKnowledgeEvidenceRole | None = None,
) -> ReportEvidenceItem:
    """Create one bounded narrative evidence item."""

    return _make_item(
        kind=kind,
        title=title,
        source=source,
        knowledge_role=knowledge_role,
        provenance_refs=provenance_refs,
        content=str(content or "").strip()[:REPORT_EVIDENCE_CONTENT_MAX_CHARS],
    )


def build_report_manifest_from_items(
    query: str,
    items: Sequence[ReportEvidenceItem],
) -> ReportEvidenceManifest:
    """Bind a de-duplicated evidence sequence to one deterministic manifest."""

    unique_items: list[ReportEvidenceItem] = []
    seen_refs: set[str] = set()
    for item in items:
        if item.evidence_ref in seen_refs:
            continue
        seen_refs.add(item.evidence_ref)
        unique_items.append(item)
        if len(unique_items) == 32:
            break
    if not unique_items:
        raise ValueError("A report evidence manifest requires evidence items.")
    query_digest = hashlib.sha256(str(query or "").encode("utf-8")).hexdigest()
    manifest_material = {
        "contract_version": REPORT_EVIDENCE_MANIFEST_VERSION,
        "query_digest": query_digest,
        "items": [item.model_dump(mode="json") for item in unique_items],
    }
    manifest_id = f"manifest:{_digest(manifest_material, length=32)}"
    return ReportEvidenceManifest.model_validate(
        {
            **manifest_material,
            "manifest_id": manifest_id,
        }
    )


def build_report_evidence_manifest(
    ctx: QueryContext,
    *,
    max_rows_per_table: int = 100,
    max_columns_per_table: int = 24,
) -> ReportEvidenceManifest:
    """Project exact pipeline evidence into a closed, checkpoint-safe manifest."""

    if not 1 <= max_rows_per_table <= 200:
        raise ValueError("max_rows_per_table must be between 1 and 200.")
    if not 1 <= max_columns_per_table <= 32:
        raise ValueError("max_columns_per_table must be between 1 and 32.")

    items: list[ReportEvidenceItem] = []
    seen_table_material: set[str] = set()
    relevant_tokens = {
        match.group(0).casefold().replace("/", "-")
        for match in _QUERY_PERIOD_PATTERN.finditer(str(ctx.query or ""))
    }

    def add_narrative(
        *,
        kind: ReportEvidenceKind,
        title: str,
        source: str,
        content: str,
        provenance_refs: list[str] | None = None,
    ) -> None:
        normalized_content = str(content or "").strip()
        if not normalized_content:
            return
        clipped = normalized_content[:REPORT_EVIDENCE_CONTENT_MAX_CHARS]
        items.append(
            _make_item(
                kind=kind,
                title=title,
                source=source,
                provenance_refs=provenance_refs,
                content=clipped,
            )
        )

    def add_table(
        *,
        title: str,
        source: str,
        columns: Sequence[Any],
        rows: Iterable[Iterable[Any]],
        provenance_refs: list[str],
    ) -> None:
        normalized_columns, normalized_rows, total_rows, truncated = _normalize_table(
            columns,
            rows,
            max_rows=max_rows_per_table,
            max_columns=max_columns_per_table,
            relevant_tokens=relevant_tokens,
        )
        if not normalized_columns or not normalized_rows:
            return
        material_key = _digest(
            {
                "columns": normalized_columns,
                "rows": normalized_rows,
                "total": total_rows,
            },
            length=32,
        )
        if material_key in seen_table_material:
            return
        seen_table_material.add(material_key)
        items.append(
            _make_item(
                kind=ReportEvidenceKind.TABLE,
                title=title,
                source=source,
                provenance_refs=provenance_refs,
                columns=normalized_columns,
                rows=normalized_rows,
                unit_by_column=_inferred_unit_by_column(normalized_columns),
                total_row_count=total_rows,
                truncated=truncated,
            )
        )

    add_table(
        title="Primary tabular evidence",
        source=str(ctx.provenance_source or "pipeline"),
        columns=list(ctx.provenance_cols or ctx.cols or []),
        rows=list(ctx.provenance_rows or ctx.rows or []),
        provenance_refs=list(ctx.provenance_refs or []),
    )

    for role, evidence in sorted((ctx.evidence_collected or {}).items()):
        add_table(
            title=f"Supporting evidence: {str(role).replace('_', ' ')}",
            source=str(evidence.get("tool") or "tool").lower().replace("-", "_"),
            columns=list(evidence.get("cols") or []),
            rows=list(evidence.get("rows") or []),
            provenance_refs=list(evidence.get("provenance_refs") or []),
        )

    if (ctx.stats_hint or "").strip():
        add_narrative(
            kind=ReportEvidenceKind.STATISTICS,
            title="Verified statistics",
            source="derived",
            provenance_refs=list(ctx.provenance_refs or []),
            content=ctx.stats_hint,
        )

    if (ctx.summary_domain_knowledge or "").strip():
        add_narrative(
            kind=ReportEvidenceKind.KNOWLEDGE,
            title="Curated domain knowledge",
            source="curated_knowledge",
            content=ctx.summary_domain_knowledge,
        )

    if (
        ctx.vector_knowledge is not None
        and ctx.vector_knowledge_source == "vector_active"
    ):
        for chunk in list(ctx.vector_knowledge.chunks or [])[:6]:
            title = chunk.document_title or chunk.section_title or "Retrieved source passage"
            add_narrative(
                kind=ReportEvidenceKind.KNOWLEDGE,
                title=title[:200],
                source="vector",
                provenance_refs=[
                    f"vector:{chunk.source_key or chunk.document_id}:{chunk.id}"
                ],
                content=chunk.text_content,
            )

    items.append(
        _make_item(
            kind=ReportEvidenceKind.LIMITATION,
            title="Evidence boundary",
            source="system",
            content=(
                "The report may make claims only from the evidence items in this "
                "manifest. Missing periods, external drivers, and causal mechanisms "
                "must be stated as limitations rather than inferred."
            ),
        )
    )

    query_digest = hashlib.sha256(str(ctx.query or "").encode("utf-8")).hexdigest()
    manifest_material = {
        "contract_version": REPORT_EVIDENCE_MANIFEST_VERSION,
        "query_digest": query_digest,
        "items": [item.model_dump(mode="json") for item in items],
    }
    manifest_id = f"manifest:{_digest(manifest_material, length=32)}"
    return ReportEvidenceManifest.model_validate(
        {
            **manifest_material,
            "manifest_id": manifest_id,
        }
    )
