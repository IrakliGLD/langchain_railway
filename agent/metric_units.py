"""Versioned dimensional contract for canonical analytics frames.

Storage values are converted exactly once while constructing a canonical
frame. Legacy raw DataFrames intentionally remain in storage units until the
canonical-frame rollout gate is enabled in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "contracts" / "metric_units_v1.json"


class UnitCompatibilityError(ValueError):
    """Raised when a value's unit is incompatible with a metric dimension."""


def normalize_unit(unit: str) -> str:
    """Normalize harmless spelling and typography differences in unit input."""
    normalized = str(unit or "").strip().lower()
    normalized = normalized.replace("¢", " cents ").replace("\u00a0", " ")
    normalized = " ".join(normalized.split())
    return normalized.replace(" / ", "/").replace("/ ", "/").replace(" /", "/")


@dataclass(frozen=True)
class MetricUnitDefinition:
    metric_id: str
    dimension: str
    storage_unit: str
    canonical_unit: str
    display_unit: str
    to_canonical_factor: float
    compatible_aggregations: tuple[str, ...]
    precision: int
    filter_unit_aliases: Mapping[str, float]

    def storage_to_canonical(self, value: float) -> float:
        return float(value) * self.to_canonical_factor

    def canonical_to_storage(self, value: float) -> float:
        return float(value) / self.to_canonical_factor

    def input_to_canonical(self, value: float, unit: str | None = None) -> float:
        """Convert user/filter input to this definition's canonical unit."""
        if unit is None or not str(unit).strip():
            return float(value)
        key = normalize_unit(unit)
        factor = self.filter_unit_aliases.get(key)
        if factor is None:
            raise UnitCompatibilityError(
                f"Unit {unit!r} is incompatible with {self.metric_id} "
                f"({self.dimension}; expected {self.canonical_unit})"
            )
        return float(value) * float(factor)


class MetricUnitRegistry:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported metric-unit schema_version")
        self.version = str(payload.get("registry_version", ""))
        raw_metrics = payload.get("metrics")
        if not isinstance(raw_metrics, Mapping) or not raw_metrics:
            raise ValueError("Metric-unit registry must define metrics")
        definitions: dict[str, MetricUnitDefinition] = {}
        for metric_id, raw in raw_metrics.items():
            aliases = {
                normalize_unit(key): float(value)
                for key, value in dict(raw["filter_unit_aliases"]).items()
            }
            definition = MetricUnitDefinition(
                metric_id=str(metric_id),
                dimension=str(raw["dimension"]),
                storage_unit=str(raw["storage_unit"]),
                canonical_unit=str(raw["canonical_unit"]),
                display_unit=str(raw["display_unit"]),
                to_canonical_factor=float(raw["to_canonical_factor"]),
                compatible_aggregations=tuple(str(item) for item in raw["compatible_aggregations"]),
                precision=int(raw["precision"]),
                filter_unit_aliases=aliases,
            )
            if definition.to_canonical_factor <= 0:
                raise ValueError(f"{metric_id}: to_canonical_factor must be positive")
            if normalize_unit(definition.canonical_unit) not in aliases:
                raise ValueError(f"{metric_id}: canonical unit must be an accepted filter unit")
            definitions[definition.metric_id] = definition
        self._definitions = definitions

    def get(self, metric_id: str) -> MetricUnitDefinition:
        try:
            return self._definitions[metric_id]
        except KeyError as exc:
            raise KeyError(f"Unknown metric-unit contract: {metric_id}") from exc

    def are_compatible(self, left_metric_id: str, right_metric_id: str) -> bool:
        return self.get(left_metric_id).dimension == self.get(right_metric_id).dimension

    def require_compatible(self, left_metric_id: str, right_metric_id: str) -> None:
        if not self.are_compatible(left_metric_id, right_metric_id):
            raise UnitCompatibilityError(
                f"Cannot compare {left_metric_id} ({self.get(left_metric_id).dimension}) "
                f"with {right_metric_id} ({self.get(right_metric_id).dimension})"
            )

    def as_public_contract(self) -> dict[str, Any]:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def load_metric_unit_registry(path: Path = REGISTRY_PATH) -> MetricUnitRegistry:
    return MetricUnitRegistry(json.loads(path.read_text(encoding="utf-8")))


METRIC_UNITS = load_metric_unit_registry()
