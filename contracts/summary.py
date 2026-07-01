"""Summary-generation result contract.

Extracted from ``core/llm.py`` (P0-1, architecture-audit 2026-06-30) so that
guardrails, the summarizer, and tests can depend on the summary schema without
importing the LLM hub. ``core.llm`` re-exports ``SummaryEnvelope`` for backward
compatibility.
"""
from typing import List

from pydantic import BaseModel, Field


class SummaryEnvelope(BaseModel):
    """Strict schema for guarded summarizer output."""

    answer: str = Field(min_length=1)
    claims: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
