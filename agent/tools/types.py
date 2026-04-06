"""
Typed tool contracts for Phase 3.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd


# Shared tuple contract returned by every deterministic tool implementation.
ToolResult = Tuple[pd.DataFrame, List[str], List[tuple]]


# Router/executor payload used to carry a validated tool call end-to-end.
@dataclass
class ToolInvocation:
    """Deterministic invocation emitted by the fast router."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
