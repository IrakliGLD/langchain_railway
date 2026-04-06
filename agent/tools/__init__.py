"""
Typed retrieval tools package.
"""
from .types import ToolInvocation, ToolResult


def execute_tool(invocation: ToolInvocation) -> ToolResult:
    # Lazy import keeps router/unit tests lightweight and avoids DB/config side effects.
    from .registry import execute_tool as _execute_tool
    return _execute_tool(invocation)


# Mirror the registry helper without importing the heavier modules at package import time.
def list_tools():
    from .registry import list_tools as _list_tools
    return _list_tools()


__all__ = [
    "ToolInvocation",
    "ToolResult",
    "execute_tool",
    "list_tools",
]
