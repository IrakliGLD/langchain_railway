"""Canonical evidence frames for the query pipeline.

Tool-specific DataFrames are normalized into these frames during evidence
collection.  The generic renderer and Stage 3 enrichment operate on frames,
never on raw tool columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ObservationFrame:
    """Period-indexed observations — the default frame for prices, generation, shares.

    Each row is one (period, entity, metric, value) tuple.  Multi-currency or
    multi-metric results become multiple rows with different ``metric`` values.
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)
    # Expected keys per row:
    #   period: str          — ISO date or "YYYY" for yearly
    #   entity_id: str       — canonical entity key (may be None for single-entity)
    #   entity_label: str    — human-readable label
    #   metric: str          — canonical metric name
    #   value: float | None
    #   unit: str            — e.g. "tetri/kWh", "GEL/MWh", "USD/MWh", "%"
    provenance_refs: List[str] = field(default_factory=list)

    @property
    def periods(self) -> List[str]:
        seen: dict[str, None] = {}
        for r in self.rows:
            p = r.get("period")
            if p and p not in seen:
                seen[p] = None
        return list(seen)

    @property
    def entities(self) -> List[str]:
        seen: dict[str, None] = {}
        for r in self.rows:
            e = r.get("entity_id")
            if e and e not in seen:
                seen[e] = None
        return list(seen)

    @property
    def metrics(self) -> List[str]:
        seen: dict[str, None] = {}
        for r in self.rows:
            m = r.get("metric")
            if m and m not in seen:
                seen[m] = None
        return list(seen)

    def is_empty(self) -> bool:
        return len(self.rows) == 0


@dataclass
class EntitySetFrame:
    """Entity enumeration — used for LIST answer_kind (e.g. "which plants are regulated?").

    Each row is one entity with an optional membership reason and attributes.
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)
    # Expected keys per row:
    #   entity_id: str
    #   entity_label: str
    #   membership_reason: str   — e.g. "regulated", "hydro", "thermal"
    #   attributes: dict         — optional extra k/v pairs for display
    provenance_refs: List[str] = field(default_factory=list)

    @property
    def entities(self) -> List[str]:
        return [r.get("entity_id", "") for r in self.rows]

    def is_empty(self) -> bool:
        return len(self.rows) == 0


@dataclass
class ComparisonFrame:
    """Side-by-side comparison — used for COMPARISON answer_kind.

    Each row compares one metric across two subjects (periods or entities).
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)
    # Expected keys per row:
    #   metric: str
    #   subject_label: str        — e.g. "January 2024"
    #   subject_value: float
    #   baseline_label: str       — e.g. "December 2023"
    #   baseline_value: float
    #   delta: float              — subject_value - baseline_value
    #   delta_percent: float|None — percentage change
    #   unit: str
    provenance_refs: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.rows) == 0


# Type alias for any canonical frame
CanonicalFrame = ObservationFrame | EntitySetFrame | ComparisonFrame
