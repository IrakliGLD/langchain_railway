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
    REPORT_EVIDENCE_MANIFEST_VERSION,
    ReportEvidenceItem,
    ReportEvidenceKind,
    ReportEvidenceManifest,
)
from models import QueryContext

_QUANTITATIVE_REPORT_PATTERN = re.compile(
    r"\b(?:"
    r"prices?|pricing|tariffs?|generation|demand|consumption|"
    r"imports?|exports?|capacity|quantit(?:y|ies|ative)|volumes?|"
    r"market\s+shares?|trends?|historical"
    r")\b",
    re.IGNORECASE,
)


def report_request_requires_table(query: str) -> bool:
    """Return whether a report request explicitly asks for measurable evidence."""

    return bool(_QUANTITATIVE_REPORT_PATTERN.search(str(query or "")))


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
) -> tuple[list[str], list[dict[str, Any]], int, bool]:
    normalized_columns = [str(column)[:128] for column in list(columns)[:max_columns]]
    materialized_rows = list(rows)
    normalized_rows = []
    for row in materialized_rows[:max_rows]:
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
        registered = METRIC_UNITS.find_for_source_metric(normalized)
        if registered is not None:
            units[column] = registered.storage_unit
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
        canonical_unit = metric_value_unit(normalized)
        if canonical_unit != "value":
            units[column] = canonical_unit
    return units


def _make_item(
    *,
    kind: ReportEvidenceKind,
    title: str,
    source: str,
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
    body["evidence_ref"] = f"evidence:{kind.value}:{_digest(body)}"
    return ReportEvidenceItem.model_validate(body)


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
        clipped = normalized_content[:6000]
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
