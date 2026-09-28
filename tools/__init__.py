"""
Tools Module
=============
Built-in tools for the AI Agent.
"""

from .base import BaseAgentTool
from .search import SearchTool, create_search_tool
from .calculator import CalculatorTool, create_calculator_tool
from .code_executor import CodeExecutorTool, create_code_executor_tool
from .file_ops import FileOpsTool, create_file_ops_tool

__all__ = [
    "BaseAgentTool",
    "SearchTool", "create_search_tool",
    "CalculatorTool", "create_calculator_tool",
    "CodeExecutorTool", "create_code_executor_tool",
    "FileOpsTool", "create_file_ops_tool",
]


def get_all_tools() -> list:
    """Get all built-in tools."""
    tools = [
        create_search_tool(),
        create_calculator_tool(),
        create_code_executor_tool(),
    ]
    # create_file_ops_tool() returns a *list* of three tools (read/write/list)
    tools.extend(create_file_ops_tool())
    return tools
