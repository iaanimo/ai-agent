"""
File Operations Tool
=====================
Read, write, and list files.
"""

from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config.settings import get_settings


def resolve_data_path(path: str) -> Path:
    """Resolve a path inside the project data directory.

    Rejects absolute paths and `..` escapes so LLM-facing tools stay
    contained inside data/. Shared by the file tools and the agent itself
    (e.g. writing reports from `final_answer`).
    """
    settings = get_settings()
    base_dir = settings.project_root / "data"
    base_dir.mkdir(parents=True, exist_ok=True)
    p = Path(path)
    if p.is_absolute():
        raise ValueError(f"Absolute paths are not allowed. Use a path relative to {base_dir}")
    resolved = (base_dir / p).resolve()
    base = base_dir.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"Path escapes the data directory: {path}")
    return resolved


class FileReadInput(BaseModel):
    path: str = Field(description="File path to read")
    max_lines: int = Field(default=100, description="Maximum lines to read")


class FileWriteInput(BaseModel):
    path: str = Field(description="File path to write")
    content: str = Field(description="Content to write")
    mode: str = Field(default="w", description="Write mode: 'w' for overwrite, 'a' for append")


class FileListInput(BaseModel):
    directory: str = Field(default=".", description="Directory to list")
    pattern: str = Field(default="*", description="Glob pattern to match files")


class FileOpsTool:
    """File operations tool with safety boundaries."""

    name = "file_ops"
    description = "Read, write, or list files. Use 'read' to read a file, 'write' to create/modify files, and 'list' to see directory contents."

    def __init__(self):
        settings = get_settings()
        self.base_dir = settings.project_root / "data"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def read_file(self, path: str, max_lines: int = 100) -> str:
        """Read a file and return its contents."""
        file_path = self._resolve_path(path)
        if not file_path.exists():
            return f"File not found: {path}"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[:max_lines]
            content = "".join(lines)
            if len(lines) == max_lines:
                content += f"\n... (truncated at {max_lines} lines)"
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_file(self, path: str, content: str, mode: str = "w") -> str:
        """Write content to a file."""
        file_path = self._resolve_path(path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def list_files(self, directory: str = ".", pattern: str = "*") -> str:
        """List files in a directory."""
        dir_path = self._resolve_path(directory)
        if not dir_path.exists():
            return f"Directory not found: {directory}"
        try:
            files = sorted(dir_path.glob(pattern))
            if not files:
                return f"No files matching '{pattern}' in {directory}"
            lines = []
            for f in files[:50]:  # Limit to 50 entries
                size = f.stat().st_size if f.is_file() else 0
                type_icon = "[DIR]" if f.is_dir() else "[FILE]"
                lines.append(f"  {type_icon} {f.name} ({size} bytes)" if f.is_file() else f"  {type_icon} {f.name}/")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the base directory."""
        return resolve_data_path(path)


def create_file_ops_tool() -> list:
    """Create file operation tools."""
    ops = FileOpsTool()
    return [
        StructuredTool(
            name="file_read",
            description="Read a file's contents. Provide the file path (relative to the data/ directory) and optional max lines.",
            func=ops.read_file,
            args_schema=FileReadInput,
        ),
        StructuredTool(
            name="file_write",
            description="Write content to a file. Path is relative to the data/ directory (do NOT prefix with 'data/'). Creates the file and parent folders automatically.",
            func=ops.write_file,
            args_schema=FileWriteInput,
        ),
        StructuredTool(
            name="file_list",
            description="List files in a directory with optional glob pattern.",
            func=ops.list_files,
            args_schema=FileListInput,
        ),
    ]
