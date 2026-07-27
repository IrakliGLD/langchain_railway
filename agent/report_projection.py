"""Single authority for which evidence rows a report section may see."""

from __future__ import annotations

import json
from typing import Any

from contracts.report_evidence import ReportEvidenceItem, ReportEvidenceKind
from utils.coverage_sampling import coverage_priority_indices


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def projected_row_indices(
    item: ReportEvidenceItem,
    *,
    budget_chars: int,
) -> list[int]:
    """Return the manifest row indices that fit one section's prompt budget.

    Boundary rows come first, then interior rows by widest gap, so a truncated
    projection still spans the item's full range. Selection is deterministic:
    the same item and budget always yield the same rows.
    """

    if item.kind is not ReportEvidenceKind.TABLE or not item.rows:
        return []
    selected: set[int] = set()
    serialized_size = 0
    for row_index in coverage_priority_indices(len(item.rows)):
        indexed_row = {
            "row_index": row_index,
            "values": item.rows[row_index],
        }
        row_cost = len(_compact_json(indexed_row)) + (1 if selected else 0)
        if serialized_size + row_cost > budget_chars:
            continue
        selected.add(row_index)
        serialized_size += row_cost
    return sorted(selected)
