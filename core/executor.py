"""
Tool Executor & Function Caller
=================================
负责工具调度、Function Calling 协议和执行结果处理。
"""

import json
import time
import traceback
from typing import Any, Callable, Optional
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool



# ─── 执行结果 ─────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0

    def to_display(self) -> str:
        if self.success:
            output_str = str(self.output)[:500]
            return f"✅ [{self.tool_name}] ({self.execution_time:.2f}s)\n{output_str}"
        else:
            return f"❌ [{self.tool_name}] Error: {self.error}"


# ─── Function Calling 协议 ─────────────────────────────────────

FUNCTION_CALL_PROMPT = """You have access to the following tools:

{tool_descriptions}

To use a tool, respond with a JSON object in this exact format:
{{
    "tool": "tool_name",
    "args": {{"arg1": "value1", "arg2": "value2"}}
}}

To provide a final answer without using a tool, respond with:
{{
    "tool": "final_answer",
    "args": {{"answer": "Your final answer here"}}
}}

To also save the answer as a file, add "file_path" to final_answer args,
e.g. {{"answer": "...", "file_path": "report.md"}} → saved under data/report.md.

Think carefully about which tool to use and what arguments to pass.
Current context:
{context}
"""


class ToolExecutor:
    """
    Executes tools and manages tool registry.
    """

    def __init__(self):
        self._tools: dict[str, BaseTool | Callable] = {}

    def register_tool(self, tool: BaseTool | Callable, name: str = None) -> None:
        """Register a tool for use."""
        tool_name = name or getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        self._tools[tool_name] = tool

    def register_tools(self, tools: list[BaseTool | Callable]) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register_tool(tool)

    def get_tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for prompt injection."""
        lines = []
        for name, tool in self._tools.items():
            if hasattr(tool, "description"):
                desc = tool.description
            elif hasattr(tool, "__doc__") and tool.__doc__:
                desc = tool.__doc__.strip().split("\n")[0]
            else:
                desc = "No description"
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    async def execute(self, tool_name: str, args: dict) -> ToolResult:
        """Execute a tool by name with arguments."""
        if tool_name not in self._tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found. Available: {list(self._tools.keys())}",
            )

        tool = self._tools[tool_name]
        start_time = time.time()

        try:
            if isinstance(tool, BaseTool):
                result = await tool.ainvoke(args)
            elif callable(tool):
                # Support both sync and async callables
                import asyncio
                if asyncio.iscoroutinefunction(tool):
                    result = await tool(**args)
                else:
                    result = tool(**args)
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")

            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=result,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                execution_time=time.time() - start_time,
            )


class FunctionCaller:
    """
    LLM-driven function calling.
    Decides which tool to use and extracts arguments.
    """

    def __init__(self, llm: BaseChatModel, executor: ToolExecutor, max_retries: int = 2):
        self.llm = llm
        self.executor = executor
        self.max_retries = max_retries

    async def call(self, query: str, context: str = "", history: str = "") -> ToolResult | str:
        """
        Let the LLM decide which tool to call and execute it.

        Returns:
            ToolResult if a tool was called, or str if it's a final answer.
        """
        prompt = FUNCTION_CALL_PROMPT.format(
            tool_descriptions=self.executor.get_tool_descriptions(),
            context=context,
        )

        messages = [
            SystemMessage(content=prompt),
        ]
        if history:
            messages.append(HumanMessage(content=f"Conversation history:\n{history}"))
        messages.append(HumanMessage(content=query))

        for attempt in range(self.max_retries + 1):
            response = await self.llm.ainvoke(messages)
            parsed = self._parse_function_call(response.content)

            if parsed is None:
                # Couldn't parse as function call, return as text
                return response.content

            tool_name = parsed.get("tool", "")
            args = parsed.get("args", {})

            # Final answer shortcut
            if tool_name == "final_answer":
                return args.get("answer", response.content)

            # Execute the tool
            result = await self.executor.execute(tool_name, args)

            if result.success:
                return result

            # Retry with error context
            if attempt < self.max_retries:
                messages.append(AIMessage(content=response.content))
                messages.append(HumanMessage(
                    content=f"Tool execution failed: {result.error}\nPlease try again with correct arguments."
                ))
            else:
                return result

        return "Failed to execute tool after retries."

    def _parse_function_call(self, content: str) -> Optional[dict]:
        """Parse function call from LLM response."""
        content = content.strip()

        # Remove markdown code fences
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(content)
            if isinstance(data, dict) and "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the text
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
            if isinstance(data, dict) and "tool" in data:
                return data
        except (ValueError, json.JSONDecodeError):
            pass

        return None
