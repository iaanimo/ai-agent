"""Tests for the tool layer: calculator safety, file containment,
code executor timeout, and search result filtering."""

import pytest

from tools.calculator import CalculatorTool
from tools.file_ops import resolve_data_path
from tools.code_executor import CodeExecutorTool
from tools.search import SearchTool


# ─── Calculator (AST safety) ───────────────────────────────────

def test_calculator_basic_math():
    assert CalculatorTool.calculate("2 + 2 * 3").startswith("Result: 8")


def test_calculator_math_functions():
    assert "Result: 9.0" in CalculatorTool.calculate("sqrt(16) + max(3, 5)")
    assert "Result: 1.0" in CalculatorTool.calculate("sin(pi / 2)")


def test_calculator_blocks_code_execution():
    # Classic eval-sandbox escapes must be rejected.
    for exploit in (
        "().__class__.__base__.__subclasses__()",
        "__import__('os').system('echo pwned')",
        "[].__class__.__mro__",
    ):
        result = CalculatorTool.calculate(exploit)
        assert result.startswith("Calculation error"), exploit


def test_calculator_blocks_unknown_names():
    assert CalculatorTool.calculate("open('x')").startswith("Calculation error")


# ─── File ops (path containment) ───────────────────────────────

def test_resolve_data_path_allows_relative(temp_project_root):
    p = resolve_data_path("reports/a.md")
    assert p == temp_project_root / "data" / "reports" / "a.md"


def test_resolve_data_path_rejects_absolute(temp_project_root):
    with pytest.raises(ValueError):
        resolve_data_path("C:/Windows/win.ini")


def test_resolve_data_path_rejects_escape(temp_project_root):
    with pytest.raises(ValueError):
        resolve_data_path("../evil.md")
    with pytest.raises(ValueError):
        resolve_data_path("../../.env")


# ─── Code executor (subprocess + timeout) ──────────────────────

def test_code_executor_basic():
    assert CodeExecutorTool.execute("print(6 * 7)").strip() == "42"


def test_code_executor_reports_errors():
    out = CodeExecutorTool.execute("raise ValueError('boom')")
    assert "boom" in out


def test_code_executor_enforces_timeout():
    out = CodeExecutorTool.execute("while True: pass", timeout=1)
    assert "timed out" in out


# ─── Search (parsing + junk filtering, no network) ─────────────

SAMPLE_HTML = """
<ol id="b_results">
  <li class="b_algo"><h2><a href="https://news.example.com/a">某公司发布 AI 新品</a></h2><p>今日新闻摘要</p></li>
  <li class="b_algo"><h2><a href="https://tools.example.com">AI工具集导航</a></h2><p>收录海量AI工具</p></li>
  <li class="b_algo"><h2><a href="https://blog.example.com/b">深度解析大模型</a></h2><p>技术分析文章</p></li>
</ol>
"""


def test_parse_bing_filters_junk():
    out = SearchTool._parse_bing(SAMPLE_HTML, 5)
    assert "AI工具集导航" not in out          # navigation site filtered
    assert "某公司发布 AI 新品" in out
    assert "深度解析大模型" in out


def test_parse_bing_respects_max_results():
    out = SearchTool._parse_bing(SAMPLE_HTML, 1)
    assert out.count("https://") == 1


def test_parse_bing_empty():
    assert SearchTool._parse_bing("<html></html>", 5) == "No results found."


def test_is_junk_markers():
    assert SearchTool._is_junk("AI工具集导航", "https://x.com")
    assert SearchTool._is_junk("网址大全", "https://x.com")
    assert not SearchTool._is_junk("某公司发布新品", "https://news.example.com")
