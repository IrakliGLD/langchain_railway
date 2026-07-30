"""Closed, size-bounded evidence manifest for multi-call report generation."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPORT_EVIDENCE_CONTENT_MAX_CHARS = 60_000
REPORT_EVIDENCE_MANIFEST_VERSION = "report-evidence-manifest-v1"
REPORT_EVIDENCE_MANIFEST_MAX_BYTES = 786_432

JsonScalar = Union[str, int, float, bool, None]
EvidenceRow = Dict[str, JsonScalar]


class ReportEvidenceKind(str, Enum):
    TABLE = "table"
    STATISTICS = "statistics"
    KNOWLEDGE = "knowledge"
    LIMITATION = "limitation"


class ReportKnowledgeEvidenceRole(str, Enum):
    """How a retrieved passage may support a report claim."""

    primary = "primary"
    supporting_reference = "supporting_reference"
    provenance_context = "provenance_context"


class _StrictEvidenceModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class ReportEvidenceItem(_StrictEvidenceModel):
    evidence_ref: str = Field(
        pattern=r"^evidence:(?:table|statistics|knowledge|limitation):[0-9a-f]{16}$"
    )
    kind: ReportEvidenceKind
    title: str = Field(min_length=1, max_length=200)
    source: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_:-]*$",
    )
    knowledge_role: Optional[ReportKnowledgeEvidenceRole] = None
    provenance_refs: List[str] = Field(default_factory=list, max_length=32)
    columns: List[str] = Field(default_factory=list, max_length=32)
    rows: List[EvidenceRow] = Field(default_factory=list, max_length=200)
    # 6000 was sized for a retrieved passage. The computed-statistics item
    # carries the whole analytical layer the standard pipeline produces —
    # column aggregates, correlations and why-context — which runs to tens of
    # thousands of characters — 59,131 on report job 22237205 — so a
    # passage-sized cap discarded ~90% of it and left reports restating raw
    # cells. Sized to hold a full stats_hint, still far under the 768 KB
    # manifest and 1 MB checkpoint ceilings; the prompt share is allocated
    # separately at projection time.
    content: str = Field(
        default="",
        max_length=REPORT_EVIDENCE_CONTENT_MAX_CHARS,
    )
    unit_by_column: Dict[str, str] = Field(default_factory=dict)
    total_row_count: int = Field(ge=0, le=10_000_000)
    truncated: bool

    @field_validator("provenance_refs")
    @classmethod
    def _validate_provenance_refs(cls, refs: List[str]) -> List[str]:
        if len(refs) != len(set(refs)):
            raise ValueError("Evidence provenance_refs must be unique.")
        if any(not ref or len(ref) > 256 for ref in refs):
            raise ValueError("Evidence provenance_refs must be bounded non-empty strings.")
        return refs

    @field_validator("columns")
    @classmethod
    def _validate_columns(cls, columns: List[str]) -> List[str]:
        if len(columns) != len(set(columns)):
            raise ValueError("Evidence columns must be unique.")
        if any(not column or len(column) > 128 for column in columns):
            raise ValueError("Evidence columns must be bounded non-empty strings.")
        return columns

    @field_validator("rows")
    @classmethod
    def _validate_row_strings(cls, rows: List[EvidenceRow]) -> List[EvidenceRow]:
        for row in rows:
            for value in row.values():
                if isinstance(value, str) and len(value) > 1000:
                    raise ValueError("Evidence row strings must not exceed 1000 characters.")
        return rows

    @model_validator(mode="after")
    def _validate_kind_shape(self) -> "ReportEvidenceItem":
        if self.kind is ReportEvidenceKind.KNOWLEDGE:
            if self.knowledge_role is None:
                # Backward compatibility for persisted v1 knowledge items.
                self.knowledge_role = ReportKnowledgeEvidenceRole.primary
        elif self.knowledge_role is not None:
            raise ValueError(
                "knowledge_role is allowed only for knowledge evidence"
            )
        if self.kind is ReportEvidenceKind.TABLE:
            if not self.columns or not self.rows:
                raise ValueError("Table evidence requires columns and rows.")
            if self.content:
                raise ValueError("Table evidence cannot mix narrative content.")
            if self.total_row_count < len(self.rows):
                raise ValueError("Table evidence total_row_count cannot be smaller than rows.")
            if self.truncated != (self.total_row_count > len(self.rows)):
                raise ValueError("Table evidence truncated must match its row counts.")
            expected_columns = set(self.columns)
            for row in self.rows:
                if set(row) != expected_columns:
                    raise ValueError("Every table evidence row must contain exactly the declared columns.")
            if not set(self.unit_by_column).issubset(expected_columns):
                raise ValueError("Evidence units may reference only declared columns.")
        else:
            if not self.content:
                raise ValueError("Narrative evidence requires content.")
            if self.columns or self.rows or self.unit_by_column:
                raise ValueError("Narrative evidence cannot contain table fields.")
            if self.total_row_count != 0 or self.truncated:
                raise ValueError("Narrative evidence cannot contain row counts.")
        return self


class ReportEvidenceManifest(_StrictEvidenceModel):
    contract_version: str = Field(pattern=r"^report-evidence-manifest-v1$")
    manifest_id: str = Field(pattern=r"^manifest:[0-9a-f]{32}$")
    query_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    items: List[ReportEvidenceItem] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _validate_manifest(self) -> "ReportEvidenceManifest":
        refs = [item.evidence_ref for item in self.items]
        if len(refs) != len(set(refs)):
            raise ValueError("Report manifest items must use a unique evidence_ref.")
        serialized_size = len(self.model_dump_json().encode("utf-8"))
        if serialized_size > REPORT_EVIDENCE_MANIFEST_MAX_BYTES:
            raise ValueError("Report evidence manifest exceeds its persistence size limit.")
        return self

    def item_by_ref(self) -> Dict[str, ReportEvidenceItem]:
        return {item.evidence_ref: item for item in self.items}
