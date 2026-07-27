"""Strict model output and validation result contracts for report sections."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictSectionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ReportDerivedOperand(_StrictSectionModel):
    evidence_ref: str = Field(
        pattern=r"^evidence:table:[0-9a-f]{16}$",
    )
    row_index: int = Field(ge=0, le=199)
    column: str = Field(min_length=1, max_length=128)


class ReportDirectClaim(_StrictSectionModel):
    evidence_ref: str = Field(
        pattern=r"^evidence:table:[0-9a-f]{16}$",
    )
    row_index: int = Field(ge=0, le=199)
    column: str = Field(min_length=1, max_length=128)
    display_value: str = Field(
        min_length=1,
        max_length=32,
        pattern=(
            r"^[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
            r"(?:[eE][-+]?\d+)?%?$"
        ),
    )
    unit: str = Field(min_length=1, max_length=64)


class ReportDerivedClaim(_StrictSectionModel):
    operation: Literal[
        "sum",
        "mean",
        "difference",
        "percent_change",
        "ratio",
        "percentage_point_change",
    ]
    operands: List[ReportDerivedOperand] = Field(min_length=1, max_length=32)
    display_value: str = Field(
        min_length=1,
        max_length=32,
        pattern=r"^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,6})?%?$",
    )
    unit: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _validate_operation_shape(self) -> "ReportDerivedClaim":
        coordinates = [
            (operand.evidence_ref, operand.row_index, operand.column)
            for operand in self.operands
        ]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("Derived claim operands must be unique.")

        if self.operation in {"sum", "mean"} and len(self.operands) < 2:
            raise ValueError(
                f"Derived claim operation {self.operation} requires at least two operands."
            )
        exact_two = {
            "difference",
            "percent_change",
            "ratio",
            "percentage_point_change",
        }
        if self.operation in exact_two and len(self.operands) != 2:
            raise ValueError(
                f"Derived claim operation {self.operation} requires exactly two operands."
            )

        displays_percent = self.display_value.endswith("%")
        normalized_unit = " ".join(self.unit.lower().split())
        if self.operation in {"percent_change", "ratio"}:
            if not displays_percent or normalized_unit != "%":
                raise ValueError(
                    "Percent change and ratio claims require a percent display and unit."
                )
        elif self.operation == "percentage_point_change":
            if displays_percent or normalized_unit not in {
                "percentage point",
                "percentage points",
                "pp",
            }:
                raise ValueError(
                    "Percentage-point claims require a percentage-points unit."
                )
        elif displays_percent or normalized_unit in {
            "%",
            "percentage point",
            "percentage points",
            "pp",
        }:
            raise ValueError(
                "Absolute derived claims require a non-percentage display and unit."
            )
        return self


class ReportSectionParagraph(_StrictSectionModel):
    text: str = Field(min_length=20, max_length=6000)
    evidence_refs: List[str] = Field(min_length=1, max_length=32)
    direct_claims: List[ReportDirectClaim] = Field(
        default_factory=list,
        max_length=32,
    )
    derived_claims: List[ReportDerivedClaim] = Field(
        default_factory=list,
        max_length=16,
    )

    @field_validator("evidence_refs")
    @classmethod
    def _validate_refs(cls, refs: List[str]) -> List[str]:
        if len(refs) != len(set(refs)):
            raise ValueError("Section paragraph evidence_refs must be unique.")
        if any(not ref or len(ref) > 256 for ref in refs):
            raise ValueError("Section paragraph evidence_refs must be bounded.")
        return refs

    @field_validator("text")
    @classmethod
    def _reject_section_headings(cls, text: str) -> str:
        if any(line.lstrip().startswith("#") for line in text.splitlines()):
            raise ValueError("Section paragraphs cannot create Markdown headings.")
        return text

    @field_validator("direct_claims")
    @classmethod
    def _validate_direct_claims(
        cls,
        claims: List[ReportDirectClaim],
    ) -> List[ReportDirectClaim]:
        identities = [
            (
                claim.evidence_ref,
                claim.row_index,
                claim.column,
                claim.display_value,
                claim.unit,
            )
            for claim in claims
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Paragraph direct_claims must be unique.")
        return claims

    @field_validator("derived_claims")
    @classmethod
    def _validate_derived_claims(
        cls,
        claims: List[ReportDerivedClaim],
    ) -> List[ReportDerivedClaim]:
        identities = [
            (
                claim.operation,
                tuple(
                    (
                        operand.evidence_ref,
                        operand.row_index,
                        operand.column,
                    )
                    for operand in claim.operands
                ),
                claim.display_value,
                claim.unit,
            )
            for claim in claims
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Paragraph derived_claims must be unique.")
        return claims


class ReportSectionDraft(_StrictSectionModel):
    contract_version: Literal["report-section-v1"]
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    title: str = Field(min_length=1, max_length=160)
    paragraphs: List[ReportSectionParagraph] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def _validate_paragraph_uniqueness(self) -> "ReportSectionDraft":
        texts = [paragraph.text for paragraph in self.paragraphs]
        if len(texts) != len(set(texts)):
            raise ValueError("Report section paragraphs must be unique.")
        return self

    @property
    def content_markdown(self) -> str:
        return "\n\n".join(paragraph.text for paragraph in self.paragraphs)


class ReportSectionValidation(_StrictSectionModel):
    valid: bool
    error_codes: List[str] = Field(default_factory=list, max_length=16)
    word_count: int = Field(ge=0, le=5000)

    @model_validator(mode="after")
    def _validate_result(self) -> "ReportSectionValidation":
        if self.valid == bool(self.error_codes):
            raise ValueError("Section validation valid flag must match error_codes.")
        return self
