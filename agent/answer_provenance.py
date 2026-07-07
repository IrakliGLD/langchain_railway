"""Answer-provenance surface: how this answer was produced, for the client.

Pure read-only projection of QueryContext — no pipeline behavior. Rides in
``chart_metadata.answer_provenance`` so UIs can express calibrated trust
(architecture §3.9). Additive and backward compatible.

Deliberately tolerant of partial context objects (``getattr`` throughout):
this block must never be able to fail a response, and test fakes exercise
``/ask`` with duck-typed stand-ins for QueryContext.
"""
from __future__ import annotations

_PATH_BY_SUMMARY_SOURCE = {
    "generic_renderer": "deterministic_render",
    "deterministic_share_summary": "specialized_formatter",
    "deterministic_regulated_tariff_list_direct": "specialized_formatter",
    "deterministic_residual_weighted_price_direct": "specialized_formatter",
    "structured_conceptual_summary": "conceptual",
    "legacy_conceptual_text_fallback": "conceptual",
    "clarification_request": "clarify",
}


def _answer_path(summary_source: str) -> str:
    if not summary_source:
        return "unknown"
    return _PATH_BY_SUMMARY_SOURCE.get(summary_source, "narrative_llm")


def build_answer_provenance(ctx) -> dict:
    """Build the response-facing provenance block from pipeline state."""
    qa = getattr(ctx, "question_analysis", None)
    authoritative = bool(getattr(ctx, "has_authoritative_question_analysis", False))
    used_tool = bool(getattr(ctx, "used_tool", False))

    query_type = ""
    answer_kind = ""
    confidence = 0.0
    if qa is not None and authoritative:
        classification = getattr(qa, "classification", None)
        query_type = getattr(getattr(classification, "query_type", None), "value", "") or ""
        answer_kind = getattr(getattr(qa, "answer_kind", None), "value", "") or ""
        confidence = float(getattr(classification, "confidence", 0.0) or 0.0)

    return {
        "answer_path": _answer_path(str(getattr(ctx, "summary_source", "") or "")),
        "summary_source": str(getattr(ctx, "summary_source", "") or ""),
        "data_source": str(getattr(ctx, "provenance_source", "") or ""),
        "tool_name": str(getattr(ctx, "tool_name", "") or "") if used_tool else "",
        "used_sql": bool(getattr(ctx, "safe_sql", "")) and not used_tool,
        "retrieval_tier": str(getattr(ctx, "vector_retrieval_tier", "") or ""),
        "analyzer": {
            "authoritative": authoritative,
            "query_type": query_type,
            "answer_kind": answer_kind,
            "confidence": confidence,
        },
        "grounding_gate": {
            "passed": bool(getattr(ctx, "summary_provenance_gate_passed", False)),
            "reason": str(getattr(ctx, "summary_provenance_gate_reason", "") or ""),
            "coverage": float(getattr(ctx, "summary_provenance_coverage", 0.0) or 0.0),
        },
    }
