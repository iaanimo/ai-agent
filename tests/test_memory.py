"""Tests for the memory layer: dedup, CJK search, delete, context assembly."""

import tempfile

from core.memory import LongTermMemory, Memory


def _ltm():
    return LongTermMemory(persist_dir=tempfile.mkdtemp())


# ─── LongTermMemory ────────────────────────────────────────────

def test_store_dedup():
    mem = _ltm()
    assert mem.store("用户偏好简洁代码") is True
    assert mem.store("用户偏好简洁代码") is False        # near-duplicate skipped
    assert len(mem.get_all()) == 1


def test_store_returns_true_for_distinct():
    mem = _ltm()
    assert mem.store("用户偏好简洁代码") is True
    assert mem.store("用户喜欢喝咖啡") is True
    assert len(mem.get_all()) == 2


def test_cjk_search():
    mem = _ltm()
    mem.store("张明是后端开发，擅长 Python 和 Go")
    hits = mem.search("擅长什么语言")
    assert hits and "后端开发" in hits[0]["content"]


def test_delete_by_id_and_keyword():
    mem = _ltm()
    mem.store("用户喜欢喝咖啡")
    mem.store("项目目标是做 AI 助手")
    eid = mem.get_all()[0]["id"]
    assert mem.delete(eid) is True          # removes the coffee entry by id
    assert mem.delete_matching("AI") == 1   # removes the remaining entry by keyword
    assert mem.get_all() == []


def test_load_backfills_ids():
    mem = _ltm()
    mem.store("old entry")
    assert all("id" in e for e in mem.get_all())


# ─── Memory (context assembly) ─────────────────────────────────

def test_get_full_context_includes_history():
    mem = Memory(enable_long_term=False)
    mem.add_user_message("我的名字是张三")
    mem.add_ai_message("你好张三")
    mem.add_user_message("我住北京")          # current query
    ctx = mem.get_full_context("我住北京")
    assert "=== Conversation History ===" in ctx
    assert "我的名字是张三" in ctx
    assert ctx.count("我住北京") == 0          # current query not echoed


def test_get_full_context_empty():
    mem = Memory(enable_long_term=False)
    assert mem.get_full_context("") == ""


def test_parse_facts():
    from core.agent import Agent
    a = Agent.__new__(Agent)
    assert a._parse_facts('{"facts": [{"content": "用户喜欢咖啡", "keywords": ["咖啡"]}]}') \
        == [{"content": "用户喜欢咖啡", "keywords": ["咖啡"]}]
    # old plain-string format still accepted
    assert a._parse_facts('{"facts": ["用户喜欢咖啡"]}') \
        == [{"content": "用户喜欢咖啡", "keywords": []}]
    assert a._parse_facts("not json") == []


def test_keyword_recall():
    mem = _ltm()
    mem.store("用户姓张，做后端开发", keywords=["职业", "工作", "做什么"])
    # a vague query that pure keyword search on content would miss
    hits = mem.search("我的职业是什么")
    assert hits and "后端开发" in hits[0]["content"]
