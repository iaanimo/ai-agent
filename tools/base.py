"""
Base Tool
==========
All custom tools inherit from this base class.
"""

from typing import Any

from langchain_core.tools import BaseTool


class BaseAgentTool(BaseTool):
    """
    Base class for all agent tools.

    Subclasses implement the async ``_arun`` (preferred) and/or the sync
    ``_run`` method. The default implementations never call each other, so
    no infinite recursion is possible.
    """
    verbose_name: str = ""
    requires_confirmation: bool = False

    async def _arun(self, **kwargs) -> Any:
        """Async execution. Subclasses should override this."""
        raise NotImplementedError(
            "BaseAgentTool subclasses must implement `_arun` (or override `_run`)."
        )

    def _run(self, **kwargs) -> Any:
        """Synchronous execution (delegates to the async implementation)."""
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — safe to drive a fresh one.
            return asyncio.run(self._arun(**kwargs))
        raise RuntimeError("BaseAgentTool is async; call ainvoke() from an async context.")
