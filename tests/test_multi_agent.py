"""Tests for the multi-agent layer: tool loop and orchestration wiring."""

import asyncio

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from multi_agent import Orchestrator, ResearcherAgent, CoderAgent, ReviewerAgent
from multi_agent.orchestrator import CollaborationMode


class _In(BaseModel):
    query: str = Field(description="q")


def _fake_search_tool():
    calls = []

    def fake_search(query):
        calls.append(query)
        return f"RESULTS:{query}"

    tool = StructuredTool(
        name="search",
        description="web search",
        func=fake_search,
        args_schema=_In,
    )
    return tool, calls


# ─── SpecializedAgent tool loop ────────────────────────────────

def test_agent_tool_loop(make_llm):
    tool, calls = _fake_search_tool()
    fake = make_llm(
        '{"tool": "search", "args": {"query": "春节"}}',
        "FINAL",
    )
    agent = ResearcherAgent(llm=fake, tools=[tool])
    result = asyncio.run(agent.execute("查一下春节"))
    assert result == "FINAL"
    assert calls == ["春节"]
    assert fake.calls == 2


def test_agent_without_tools_single_call(make_llm):
    fake = make_llm("PLAIN")
    agent = ResearcherAgent(llm=fake, tools=[])
    assert asyncio.run(agent.execute("hi")) == "PLAIN"
    assert fake.calls == 1


# ─── Default tool wiring ───────────────────────────────────────

def test_default_tool_wiring(make_llm):
    c = CoderAgent(llm=make_llm())
    assert c.executor.get_tool_names() == ["code_executor"]

    rs = ResearcherAgent(llm=make_llm())
    assert rs.executor.get_tool_names() == ["search"]

    rv = ReviewerAgent(llm=make_llm())
    names = set(rv.executor.get_tool_names())
    assert {"code_executor", "file_read", "file_write", "file_list"} <= names


# ─── Orchestrator modes ────────────────────────────────────────

def test_orchestrator_sequential(make_llm):
    o = Orchestrator(llm=make_llm("SYNTH"), verbose=False)
    o.add_agent(ResearcherAgent(llm=make_llm("R")))
    o.add_agent(CoderAgent(llm=make_llm("C")))
    o.add_agent(ReviewerAgent(llm=make_llm("RV")))
    res = asyncio.run(o.run("task", mode=CollaborationMode.SEQUENTIAL))
    assert res.final_answer == "SYNTH"
    assert set(res.agent_results) == {"Researcher", "Coder", "Reviewer"}


def test_orchestrator_debate(make_llm):
    o = Orchestrator(llm=make_llm("SYNTH2"), verbose=False)
    o.add_agent(ResearcherAgent(llm=make_llm("A1", "A1R")))
    o.add_agent(CoderAgent(llm=make_llm("A2", "A2R")))
    res = asyncio.run(o.run("task", mode=CollaborationMode.DEBATE))
    assert res.final_answer == "SYNTH2"
    assert set(res.agent_results) == {"Researcher", "Coder"}
