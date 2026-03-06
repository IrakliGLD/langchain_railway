"""
Registry for typed retrieval tools.
"""
from typing import Callable, Dict, List

from .composition_tools import get_balancing_composition
from .generation_tools import get_generation_mix
from .price_tools import get_prices
from .tariff_tools import get_tariffs
from .types import ToolInvocation, ToolResult


TOOL_REGISTRY: Dict[str, Callable[..., ToolResult]] = {
    "get_prices": get_prices,
    "get_balancing_composition": get_balancing_composition,
    "get_tariffs": get_tariffs,
    "get_generation_mix": get_generation_mix,
}


def list_tools() -> List[str]:
    return sorted(TOOL_REGISTRY.keys())


def execute_tool(invocation: ToolInvocation) -> ToolResult:
    """Execute the invocation against a strict registry."""
    if invocation.name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {invocation.name}")
    func = TOOL_REGISTRY[invocation.name]
    return func(**invocation.params)

