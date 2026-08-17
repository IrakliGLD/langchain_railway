"""Contract-continuity snapshot (architecture §3.2, design item 2 slice 1).

Builds the compact JSON of a turn's routed contract that the NEXT turn's
analyzer prompt receives as a TRUSTED block (``TRUSTED_PREVIOUS_CONTRACT``).
The snapshot is structured, session-scoped state — cheaper and more stable
than re-deriving intent from raw history text, which the truncation policy
drops first under budget pressure.
"""
from __future__ import annotations

import json

# Partial JSON is worse than none: an over-cap snapshot is dropped entirely.
_MAX_SNAPSHOT_CHARS = 2000


def continuity_snapshot_json(ctx) -> str:
    """Compact routed-contract JSON for the next turn, or "" when unavailable.

    Never raises — this runs on the /ask hot path after every successful
    request, and a continuity failure must not be able to fail the response.
    """
    try:
        return _build_snapshot(ctx)
    except Exception:
        return ""


def _build_snapshot(ctx) -> str:
    if not getattr(ctx, "has_authoritative_question_analysis", False):
        return ""
    qa = getattr(ctx, "question_analysis", None)
    if qa is None:
        return ""

    tools = qa.tooling.candidate_tools or []
    top = tools[0] if tools else None
    hint = getattr(top, "params_hint", None) if top else None
    hint_dict: dict = {}
    if hint is not None:
        # Deliberately no start_date / end_date. Those describe WHEN the last
        # question was about, and the next one may be about a different period --
        # while metric, currency and granularity describe WHAT is being asked and
        # are safe to carry.
        #
        # 2026-08-17: turn 1's analyzer invented "August 2026" for a dateless
        # question, and inheriting that window is how an answer's coverage came to
        # depend on which question was asked first. Scope is worth inheriting; a
        # period nobody asked for should not survive one turn, let alone
        # propagate through a session.
        for field in ("metric", "currency", "granularity"):
            value = getattr(hint, field, None)
            if value:
                hint_dict[field] = str(value)
        entities = list(getattr(hint, "entities", []) or [])
        if entities:
            hint_dict["entities"] = entities

    snapshot = {
        "query_type": qa.classification.query_type.value,
        "answer_kind": qa.answer_kind.value if qa.answer_kind else None,
        "render_style": qa.render_style.value if qa.render_style else None,
        "preferred_path": qa.routing.preferred_path.value,
        "top_tool": top.name if top else None,  # ToolName is a str-enum
        "params_hint": hint_dict or None,
        "entity_scope": getattr(qa, "entity_scope", None),
        "canonical_query_en": qa.canonical_query_en,
    }
    compact = {key: value for key, value in snapshot.items() if value}
    rendered = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) > _MAX_SNAPSHOT_CHARS:
        return ""
    return rendered
