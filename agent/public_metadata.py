"""One allow-listed projection for caller-visible response metadata."""

from __future__ import annotations

from typing import Any, Iterable

# Chart rendering fields are intentionally explicit.  Internal diagnostics,
# model/token/cost telemetry, stage timings, prompts, SQL, claims, and session
# state are absent by construction.
PUBLIC_CHART_METADATA_FIELDS = frozenset(
    {
        "aggregation",
        "axisMode",
        "companionTable",
        "evidenceFilterApplied",
        "evidenceSource",
        "evidenceUnit",
        "groupIndex",
        "groupSource",
        "has_projection",
        "labels",
        "longFrame",
        "longFrameColumns",
        "measureTransform",
        "projection_to",
        "provenanceRefs",
        "role",
        "seasonLabels",
        "seriesConfig",
        "sourceMetrics",
        "timeGrain",
        "title",
        "trendlines",
        "visualGoal",
        "xAxisTitle",
        "yAxisLeft",
        "yAxisRight",
        "yAxisTitle",
    }
)


def project_public_chart_metadata(metadata: Any) -> dict[str, Any]:
    """Drop every chart metadata field that is not part of the public DTO."""

    if not isinstance(metadata, dict):
        return {}
    return {
        key: metadata[key]
        for key in PUBLIC_CHART_METADATA_FIELDS
        if key in metadata
    }


def project_public_charts(charts: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Apply the same metadata policy to every multi-chart response entry."""

    projected: list[dict[str, Any]] = []
    for chart in charts or []:
        if not isinstance(chart, dict):
            continue
        projected.append(
            {
                "data": chart.get("data"),
                "type": chart.get("type"),
                "metadata": project_public_chart_metadata(chart.get("metadata")),
            }
        )
    return projected


def build_public_response_metadata(
    ctx,
    *,
    request_id: str,
    trace_id: str,
    metric_unit_registry_version: str,
    request_deadline: dict[str, Any],
    answer_provenance: dict[str, Any],
    protected_telemetry: dict[str, Any] | None = None,
    guardrail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the only caller-visible metadata DTO for ``/ask``.

    ``protected_telemetry`` is accepted only to make the boundary explicit to
    callers: it is deliberately never copied into the returned DTO.
    """

    del protected_telemetry
    metadata = project_public_chart_metadata(getattr(ctx, "chart_meta", None))
    metadata.update(
        {
            "trace_id": trace_id,
            "request_id": request_id,
            "summary_citations": list(getattr(ctx, "summary_citations", []) or []),
            "summary_confidence": float(getattr(ctx, "summary_confidence", 0.0) or 0.0),
            "summary_provenance_coverage": float(
                getattr(ctx, "summary_provenance_coverage", 0.0) or 0.0
            ),
            "summary_provenance_gate_passed": bool(
                getattr(ctx, "summary_provenance_gate_passed", False)
            ),
            "summary_provenance_gate_reason": str(
                getattr(ctx, "summary_provenance_gate_reason", "") or ""
            ),
            "provenance_query_hash": str(
                getattr(ctx, "provenance_query_hash", "") or ""
            ),
            "provenance_source": str(getattr(ctx, "provenance_source", "") or ""),
            "provenance_refs": list(getattr(ctx, "provenance_refs", []) or []),
            "metric_unit_registry_version": metric_unit_registry_version,
            "answer_provenance": answer_provenance,
            "request_deadline": dict(request_deadline),
        }
    )
    if guardrail:
        metadata["guardrail_action"] = str(guardrail.get("action") or "")
        metadata["guardrail_reason"] = str(guardrail.get("reason") or "")
        metadata["guardrail_risk_score"] = float(guardrail.get("risk_score") or 0.0)
    return metadata
