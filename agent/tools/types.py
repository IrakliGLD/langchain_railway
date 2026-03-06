"""
Typed tool contracts for Phase 3.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd


ToolResult = Tuple[pd.DataFrame, List[str], List[tuple]]


@dataclass
class ToolInvocation:
    """Deterministic invocation emitted by the fast router."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""

