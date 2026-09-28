"""Tests for the agent: action parsing, report writing, replan merge, RAG fallback."""

import asyncio

from core.agent import Agent
from core.planner import Plan, PlanStep, StepStatus


# ─── Action parsing ────────────────────────────────────────────

def test_parse_action_valid():
    a = Agent.__new__(Agent)
    d = a._parse_action('{"action": {"tool": "search", "args": {"query": "x"}}}')
    assert d and d["action"]["tool"] == "search"


def test_parse_action_garbage():
    a = Agent.__new__(Agent)
    assert a._parse_action("no json here") is None
    assert a._parse_action("```json\n{'bad': 1}\n```") is None


def test_parse_action_from_fenced_json():
    a = Agent.__new__(Agent)
    d = a._parse_action('```json\n{"action": {"tool": "final_answer", "args": {}}}\n```')
    assert d and d["action"]["tool"] == "final_answer"


# ─── ReAct: report writing + parse retry ───────────────────────

def test_react_final_answer_writes_report(temp_project_root, make_llm):
    fake = make_llm('{"action": {"tool": "final_answer", "args": {"answer": "REPORT", "file_path": "r.md"}}}')
    agent = Agent(llm=fake, tools=[], enable_cot=False, verbose=False,
                  auto_memory=False, enable_memory=False)
    result = asyncio.run(agent.run("write a report"))
    assert "r.md" in result
    p = temp_project_root / "data" / "r.md"
    assert p.exists()
    assert p.read_text(encoding="utf-8") == "REPORT"


def test_react_retries_on_bad_json(make_llm):
    fake = make_llm(
        "not json at all",                                    # unparseable → retry
        '{"action": {"tool": "final_answer", "args": {"answer": "OK"}}}',  # parsed
    )
    agent = Agent(llm=fake, tools=[], enable_cot=False, verbose=False,
                  auto_memory=False, enable_memory=False)
    result = asyncio.run(agent.run("hi"))
    assert result == "OK"
    assert fake.calls >= 2


# ─── Plan-and-Execute: replan preserves results ────────────────

def test_merge_replanned_plan_preserves_results():
    a = Agent.__new__(Agent)

    old = Plan(goal="g")
    old.add_step("s0", "reason")
    old.steps[0].status = StepStatus.COMPLETED
    old.steps[0].result = "RESULT_0"
    old.add_step("s1", "search")
    old.steps[1].status = StepStatus.FAILED

    new = Plan(goal="g2")
    new.revision = 1
    new.add_step("n0", "reason", dependencies=[])
    new.add_step("n1", "final_answer", dependencies=[0])

    m = a._merge_replanned_plan(old, new)
    assert m.steps[0].result == "RESULT_0"
    assert m.steps[0].status == StepStatus.COMPLETED
    assert [s.id for s in m.steps] == [0, 1, 2]
    assert m.steps[2].dependencies == [1]
    assert m.goal == "g2" and m.revision == 1


# ─── RAG step ─────────────────────────────────────────────────

def test_rag_step_falls_back_without_rag(make_llm):
    fake = make_llm("LLM_FALLBACK")
    agent = Agent(llm=fake, tools=[], enable_cot=False, verbose=False, auto_memory=False)
    step = PlanStep(id=0, description="rag", action="rag_search", args={})
    plan = Plan(goal="g")
    plan.steps.append(step)
    result = asyncio.run(agent._execute_rag_step(step, "question"))
    assert result == "LLM_FALLBACK"
