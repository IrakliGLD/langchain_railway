"""Catalogs and normalization helpers for vector-backed knowledge ingestion."""

from __future__ import annotations

from typing import Dict


DOCUMENT_TYPE_ALIASES: Dict[str, str] = {
    "regulation": "regulation",
    "rule": "regulation",
    "market_rule": "regulation",
    "law": "law",
    "report": "report",
    "study": "report",
    "analysis": "report",
    "methodology": "methodology",
    "order": "order",
    "tariff_order": "order",
    "presentation": "presentation",
    "consultation": "consultation",
    "notice": "notice",
    "other": "other",
}


def normalize_document_type(value: str) -> str:
    """Map loose document-type labels onto a compact stable catalog."""

    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return "other"
    return DOCUMENT_TYPE_ALIASES.get(raw, raw)

