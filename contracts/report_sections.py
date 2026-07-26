"""Strict model output and validation result contracts for report sections."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictSectionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ReportSectionParagraph(_StrictSectionModel):
    text: str = Field(min_length=20, max_length=6000)
    evidence_refs: List[str] = Field(min_length=1, max_length=32)

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
