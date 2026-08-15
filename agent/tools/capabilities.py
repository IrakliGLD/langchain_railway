"""Deterministic result-shape capabilities for typed retrieval tools."""

from __future__ import annotations

from contracts.question_analysis import ToolName

# With no date filters, each current time-aware retrieval tool sorts newest
# first and applies a row limit. The result is therefore a bounded historical
# series rather than a scalar lookup. Keep this list explicit so a future
# single-value tool is not accidentally treated as multi-period merely because
# it was added to the execution registry.
_DEFAULT_MULTI_PERIOD_TOOLS = frozenset(
    {
        ToolName.GET_PRICES.value,
        ToolName.GET_TARIFFS.value,
        ToolName.GET_END_USER_PRICES.value,
        ToolName.GET_GENERATION_MIX.value,
        ToolName.GET_BALANCING_COMPOSITION.value,
    }
)


def defaults_to_multi_period(tool_name: str) -> bool:
    """Return whether an unbounded invocation supplies historical rows."""

    return str(tool_name or "").strip() in _DEFAULT_MULTI_PERIOD_TOOLS
