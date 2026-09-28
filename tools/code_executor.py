"""
Code Executor Tool
===================
Execute Python code in an isolated subprocess with a hard timeout.
"""

import subprocess
import sys
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CodeExecutorInput(BaseModel):
    code: str = Field(description="Python code to execute")
    timeout: int = Field(default=30, description="Execution timeout in seconds (1-120)")


class CodeExecutorTool:
    """Python code execution tool: isolated subprocess + enforced timeout."""

    name = "code_executor"
    description = "Execute Python code and return the output. Use this for calculations, data processing, or any task that requires code. The code runs in an isolated subprocess with a hard timeout and access to common libraries."

    # Modules pre-imported into the subprocess so snippets keep working.
    PRELUDE = (
        "import math, random, datetime, json, re, collections, itertools, "
        "functools, string, textwrap, pprint, statistics, decimal, fractions, csv, io\n"
        "for _mod in ('numpy', 'pandas', 'matplotlib'):\n"
        "    try:\n"
        "        globals()[_mod] = __import__(_mod)\n"
        "    except ImportError:\n"
        "        pass\n"
    )

    @staticmethod
    def execute(code: str, timeout: int = 30) -> str:
        """Execute Python code in an isolated subprocess and capture output."""
        # Validate and clamp the timeout (1-120s).
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 30
        timeout = min(max(timeout, 1), 120)

        # `-I` (isolated mode): ignore PYTHON* env vars and user site-packages,
        # so the subprocess cannot reach the parent's environment by accident.
        cmd = [sys.executable, "-I", "-c", CodeExecutorTool.PRELUDE + code]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Execution timed out after {timeout}s. The subprocess was terminated."
        except OSError as e:
            return f"Failed to launch subprocess: {e}"

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()

        if proc.returncode != 0:
            detail = err or out or "unknown error"
            return f"Execution error (exit code {proc.returncode}):\n{detail}"

        result = out
        if err:
            result += f"\n[stderr]\n{err}" if result else f"[stderr]\n{err}"
        return result.strip() or "Code executed successfully (no output)."


def create_code_executor_tool() -> StructuredTool:
    """Create a code executor tool instance."""
    return StructuredTool(
        name=CodeExecutorTool.name,
        description=CodeExecutorTool.description,
        func=CodeExecutorTool.execute,
        args_schema=CodeExecutorInput,
    )
