"""Fixture-candidate emission for the routing golden set (architecture §5.3).

When production detects a routing-quality signal (answer-kind cross-check
disagreement, provenance-gate failure), we emit ONE parseable log line
proposing the request as a golden-set fixture. The offline harvester
(``evaluation/harvest_fixture_candidates.py``) converts pasted logs into
candidate cases — expected fields intentionally blank, a human labels before
adoption. This turns the §5.3 failure taxonomy into a self-growing eval.

Import-light on purpose (stdlib only): the keyless harvester imports
``MARKER`` and ``routed_fields_snapshot`` from here.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger("Enai")

MARKER = "ROUTING_FIXTURE_CANDIDATE"


def routed_fields_snapshot(qa) -> dict:
    """Post-finalize routed contract fields, tolerant of a missing analysis."""
    if qa is None:
        return {}
    tools = qa.tooling.candidate_tools or []
    return {
        "query_type": qa.classification.query_type.value,
        "answer_kind": qa.answer_kind.value if qa.answer_kind else None,
        "render_style": qa.render_style.value if qa.render_style else None,
        "preferred_path": qa.routing.preferred_path.value,
        "top_tool": tools[0].name if tools else None,
    }


def log_fixture_candidate(trigger: str, ctx) -> None:
    """Emit one ``MARKER {json}`` line proposing this request as a fixture.

    Never raises — observability must not be able to break the pipeline.
    Newlines are collapsed so the line stays single-line parseable.
    """
    try:
        payload = {
            "trigger": str(trigger),
            "query": " ".join(str(getattr(ctx, "query", "") or "").split()),
            "routed": routed_fields_snapshot(getattr(ctx, "question_analysis", None)),
            "trace_id": str(getattr(ctx, "trace_id", "") or ""),
        }
        log.info(
            "%s %s", MARKER,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
    except Exception:
        log.debug("fixture-candidate emission failed", exc_info=True)
