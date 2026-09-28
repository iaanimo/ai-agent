"""
Planner Module
==============
任务规划与分解引擎。
将复杂任务拆解为可执行的步骤序列，支持动态重规划。
"""

import json
import time
from datetime import datetime
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel as PydanticBaseModel, Field


# ─── 数据结构 ─────────────────────────────────────────────────

class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in a plan."""
    id: int
    description: str
    action: str  # tool name or "reason" or "final_answer"
    args: dict = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    dependencies: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "action": self.action,
            "args": self.args,
            "status": self.status.value,
            "result": self.result,
            "dependencies": self.dependencies,
        }


class Plan:
    """A complete execution plan."""

    def __init__(self, goal: str, steps: Optional[list[PlanStep]] = None):
        self.goal = goal
        self.steps: list[PlanStep] = steps or []
        self.created_at = time.time()
        self.revision = 0

    def add_step(self, description: str, action: str, args: dict = None, dependencies: list[int] = None) -> PlanStep:
        step = PlanStep(
            id=len(self.steps),
            description=description,
            action=action,
            args=args or {},
            dependencies=dependencies or [],
        )
        self.steps.append(step)
        return step

    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next executable step (pending with all dependencies met)."""
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            # Check dependencies
            deps_met = all(
                self.steps[dep_id].status == StepStatus.COMPLETED
                for dep_id in step.dependencies
                if dep_id < len(self.steps)
            )
            if deps_met:
                return step
        return None

    def mark_step(self, step_id: int, status: StepStatus, result: str = None, error: str = None) -> None:
        if 0 <= step_id < len(self.steps):
            self.steps[step_id].status = status
            if result:
                self.steps[step_id].result = result
            if error:
                self.steps[step_id].error = error

    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for s in self.steps)

    def has_failed(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def get_progress(self) -> str:
        total = len(self.steps)
        done = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return f"[{done}/{total}]"

    def to_display(self) -> str:
        """Human-readable plan display."""
        lines = [f"📋 Plan for: {self.goal} {self.get_progress()}"]
        lines.append("=" * 50)
        for step in self.steps:
            status_icons = {
                StepStatus.PENDING: "⏳",
                StepStatus.IN_PROGRESS: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️",
            }
            icon = status_icons.get(step.status, "?")
            lines.append(f"  {icon} Step {step.id}: {step.description}")
            if step.result:
                lines.append(f"     → Result: {step.result[:100]}...")
            if step.error:
                lines.append(f"     → Error: {step.error[:100]}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "revision": self.revision,
        }


# ─── 规划器 ───────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """You are a task planning expert. Your job is to decompose complex tasks into clear, executable steps.

Today's date is {today}. Prefer current information where relevant.

For each step, specify:
1. "description": What this step does (human-readable)
2. "action": The action to take - one of:
   - A tool name (e.g., "search", "calculator", "code_executor", "file_read")
   - "reason" for pure reasoning/analysis steps
   - "rag_search" for knowledge retrieval
   - "final_answer" for the concluding step
3. "args": Arguments for the action (dict)
4. "dependencies": List of step IDs this step depends on (usually the previous step)

Available tools: {available_tools}

Respond in JSON format:
{{
    "goal": "The main objective",
    "steps": [
        {{"description": "...", "action": "...", "args": {{}}, "dependencies": []}},
        ...
    ]
}}
"""

REPLANNER_PROMPT = """The original plan has encountered an issue. Please revise the plan.

Today's date is {today}.

Original goal: {goal}
Current progress:
{progress}

Failed step: {failed_step}
Error: {error}

Available context:
{context}

Available tools: {available_tools}

Create a revised plan to achieve the goal. Respond in the same JSON format.
"""


class Planner:
    """
    LLM-powered task planner.
    Decomposes goals into executable step sequences.
    """

    def __init__(self, llm: BaseChatModel, available_tools: list[str] = None):
        self.llm = llm
        self.available_tools = available_tools or ["search", "calculator", "code_executor", "file_read", "reason"]

    async def create_plan(self, goal: str, context: str = "") -> Plan:
        """Create an execution plan for the given goal."""
        tool_list = ", ".join(self.available_tools)
        system_msg = PLANNER_SYSTEM_PROMPT.format(available_tools=tool_list, today=datetime.now().strftime("%Y-%m-%d %H:%M"))

        user_content = f"Goal: {goal}"
        if context:
            user_content += f"\n\nContext:\n{context}"

        response = await self.llm.ainvoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_content),
        ])

        plan_data = self._parse_plan(response.content)
        plan = Plan(goal=plan_data.get("goal", goal))
        for step_data in plan_data.get("steps", []):
            plan.add_step(
                description=step_data.get("description", ""),
                action=step_data.get("action", "reason"),
                args=step_data.get("args", {}),
                dependencies=step_data.get("dependencies", []),
            )
        return plan

    async def replan(self, plan: Plan, failed_step: PlanStep, context: str = "") -> Plan:
        """Replan after a step failure."""
        tool_list = ", ".join(self.available_tools)

        progress_lines = []
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED:
                progress_lines.append(f"✅ Step {step.id}: {step.description} → {step.result[:200] if step.result else 'done'}")
            elif step.status == StepStatus.FAILED:
                progress_lines.append(f"❌ Step {step.id}: {step.description} → FAILED")
            else:
                progress_lines.append(f"⏳ Step {step.id}: {step.description}")

        prompt = REPLANNER_PROMPT.format(
            goal=plan.goal,
            today=datetime.now().strftime("%Y-%m-%d %H:%M"),
            progress="\n".join(progress_lines),
            failed_step=f"Step {failed_step.id}: {failed_step.description}",
            error=failed_step.error or "Unknown error",
            context=context,
            available_tools=tool_list,
        )

        response = await self.llm.ainvoke([
            SystemMessage(content="You are a replanning expert. Create a revised plan."),
            HumanMessage(content=prompt),
        ])

        plan_data = self._parse_plan(response.content)
        new_plan = Plan(goal=plan_data.get("goal", plan.goal))
        new_plan.revision = plan.revision + 1
        for step_data in plan_data.get("steps", []):
            new_plan.add_step(
                description=step_data.get("description", ""),
                action=step_data.get("action", "reason"),
                args=step_data.get("args", {}),
                dependencies=step_data.get("dependencies", []),
            )
        return new_plan

    def _parse_plan(self, content: str) -> dict:
        """Parse plan JSON from LLM response."""
        # Try to extract JSON from the response
        content = content.strip()
        # Remove markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback: create a simple single-step plan
            return {
                "goal": "Answer the question",
                "steps": [
                    {"description": "Analyze and answer", "action": "reason", "args": {"query": content}, "dependencies": []},
                    {"description": "Provide final answer", "action": "final_answer", "args": {}, "dependencies": [0]},
                ]
            }
