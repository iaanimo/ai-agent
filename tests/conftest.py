"""Shared fixtures for the test suite.

Tests never touch the network, the real DeepSeek API, or the real data/
directory. All LLM calls go through FakeLLM; all file writes go to tmp dirs.

The `_sandbox_real_paths` fixture below is what actually enforces that last
claim — see its docstring.
"""

import sys
from pathlib import Path

# Make the project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


@pytest.fixture(autouse=True)
def _sandbox_real_paths(tmp_path, monkeypatch):
    """Keep every test off the real data/ tree and out of the project directory.

    Two leaks this closes:

    1. Modules do `from config.settings import get_settings`, and get_settings
       is lru_cached — so one shared Settings instance decides where data/
       lives for every module at once. Redirecting that instance covers all of
       them. Without this, constructing a Memory() inside a test reads and
       writes the real data/: since long-term memory moved to SQLite, merely
       constructing it creates the real store and migrates the real JSON, so
       running the test suite mutated live user memory.

    2. tempfile falls back to the current directory when TMPDIR holds a POSIX
       path on Windows, so every `tempfile.mkdtemp()` in the suite dropped a
       tmp*/ dir into the project root (~8 per run, never cleaned).
    """
    import tempfile

    import config.settings as settings_mod

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    settings = settings_mod.get_settings()
    monkeypatch.setattr(settings, "project_root", tmp_path)
    monkeypatch.setattr(settings, "data_dir", data_dir)
    monkeypatch.setattr(settings, "knowledge_base_dir", data_dir / "knowledge_base")
    monkeypatch.setattr(settings.vector_store, "persist_dir", str(data_dir / "vector_store"))

    tmp_home = tmp_path / "tmp"
    tmp_home.mkdir(exist_ok=True)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_home))


class _Message:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Deterministic chat model: returns scripted responses per call.

    `responses` is a list of strings; the last one repeats on overflow.
    """

    def __init__(self, responses=None):
        self.responses = list(responses or ["ok"])
        self.calls = 0
        self.last_prompt = ""

    async def ainvoke(self, messages):
        self.calls += 1
        self.last_prompt = messages[-1].content if messages else ""
        content = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        return _Message(content)


@pytest.fixture
def make_llm():
    """Factory: make_llm("resp1", "resp2", ...) -> FakeLLM."""
    return lambda *responses: FakeLLM(list(responses))


@pytest.fixture
def temp_project_root(tmp_path):
    """Project root for tests that assert on paths written under data/.

    `_sandbox_real_paths` (autouse) already redirects the settings singleton to
    this same tmp_path, so all this fixture has to do is hand the path over.
    """
    return tmp_path
