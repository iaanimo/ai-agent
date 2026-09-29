"""
Multi-Agent Orchestrator
=========================
协调多个专业 Agent 完成复杂任务。
支持三种协作模式：
  - Sequential（串行）：Agent 按顺序处理
  - Parallel（并行）：Agent 同时处理不同子任务
  - Debate（辩论）：多个 Agent 互相评审，达成共识
"""

import json
import time
import asyncio
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rich.console import Console

from core.llm_factory import create_llm
from core.executor import ToolExecutor

console = Console()


class CollaborationMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DEBATE = "debate"


@dataclass
class AgentMessage:
    """A message passed between agents."""
    sender: str
    receiver: str
    content: str
    task_id: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskResult:
    """Result from the orchestrator."""
    task: str
    mode: CollaborationMode
    agent_results: dict[str, str] = field(default_factory=dict)
    final_answer: str = ""
    elapsed_time: float = 0.0

    def to_display(self) -> str:
        lines = [f"🎯 Task: {self.task}", f"Mode: {self.mode.value}", f"Time: {self.elapsed_time:.2f}s", "=" * 50]
        for agent, result in self.agent_results.items():
            lines.append(f"\n👤 {agent}:\n{result[:300]}...")
        lines.append(f"\n{'=' * 50}")
        lines.append(f"\n📋 Final Answer:\n{self.final_answer}")
        return "\n".join(lines)


# ─── Base Specialized Agent ───────────────────────────────────

class SpecializedAgent:
    """Base class for specialized agents in the multi-agent system.

    An agent may hold tools (e.g. Researcher → web search, Coder → code
    executor). With tools, `execute()` runs a short reason→act→observe loop;
    without tools it is a single LLM call (the original behavior).
    """

    def __init__(self, name: str, role: str, llm: BaseChatModel, system_prompt: str,
                 tools: Optional[list] = None, max_tool_calls: int = 5):
        self.name = name
        self.role = role
        self.llm = llm
        self.system_prompt = system_prompt
        self.message_history: list[AgentMessage] = []
        self.tools = list(tools or [])
        self.max_tool_calls = max_tool_calls
        self.executor = ToolExecutor()
        if self.tools:
            self.executor.register_tools(self.tools)

    async def execute(self, task: str, context: str = "") -> str:
        """Execute a task and return the result (using tools if available)."""
        if not self.tools:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self._build_prompt(task, context)),
            ]
            response = await self.llm.ainvoke(messages)
            return response.content
        return await self._execute_with_tools(task, context)

    async def _execute_with_tools(self, task: str, context: str) -> str:
        """Reason→act→observe loop with the agent's tools."""
        system = self.system_prompt + f"""

You have access to these tools:
{self.executor.get_tool_descriptions()}

When you need to use a tool, respond with ONLY a JSON object:
{{"tool": "tool_name", "args": {{...}}}}

When you have the answer, reply with your final answer as plain text (no JSON)."""

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=self._build_prompt(task, context)),
        ]

        for _ in range(self.max_tool_calls):
            response = await self.llm.ainvoke(messages)
            parsed = self._parse_tool_call(response.content)
            if parsed is None:
                return response.content  # final plain-text answer
            tool_name, tool_args = parsed
            if tool_name == "final_answer":
                return tool_args.get("answer", response.content)
            result = await self.executor.execute(tool_name, tool_args)
            observation = (f"Tool '{tool_name}' result: {result.output}"
                           if result.success else f"Tool '{tool_name}' error: {result.error}")
            messages.append(AIMessage(content=response.content))
            messages.append(HumanMessage(
                content=f"Observation: {observation}\n\nContinue. When done, reply with your final answer as plain text."
            ))

        return "Reached the tool-call limit."

    def _parse_tool_call(self, content: str) -> Optional[tuple[str, dict]]:
        """Parse a JSON tool call from the LLM response. Returns (tool, args) or None."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "tool" in data:
                return data["tool"], data.get("args", {})
        except json.JSONDecodeError:
            pass
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            data = json.loads(text[start:end])
            if isinstance(data, dict) and "tool" in data:
                return data["tool"], data.get("args", {})
        except (ValueError, json.JSONDecodeError):
            pass
        return None

    async def review(self, content: str, original_task: str) -> str:
        """Review content and provide feedback (tools available if any)."""
        prompt = f"Review the following work for the task: {original_task}\n\nWork:\n{content}\n\nProvide constructive feedback."
        return await self.execute(prompt)

    def receive_message(self, message: AgentMessage) -> None:
        self.message_history.append(message)

    def _build_prompt(self, task: str, context: str) -> str:
        prompt = f"Task: {task}"
        if context:
            prompt += f"\n\nContext from other agents:\n{context}"
        if self.message_history:
            recent = self.message_history[-5:]
            msg_context = "\n".join(f"[{m.sender}]: {m.content[:200]}" for m in recent)
            prompt += f"\n\nRecent messages:\n{msg_context}"
        return prompt


# ─── Orchestrator ─────────────────────────────────────────────

class Orchestrator:
    """
    Multi-Agent Orchestrator.
    Coordinates multiple specialized agents to solve complex tasks.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None, verbose: bool = True):
        self.llm = llm or create_llm()
        self.verbose = verbose
        self.agents: dict[str, SpecializedAgent] = {}

    def add_agent(self, agent: SpecializedAgent) -> None:
        """Register an agent."""
        self.agents[agent.name] = agent

    async def run(self, task: str, mode: CollaborationMode = CollaborationMode.SEQUENTIAL) -> TaskResult:
        """
        Run the orchestrator on a task.

        Args:
            task: The task to accomplish.
            mode: Collaboration mode.

        Returns:
            TaskResult with all agent outputs and final answer.
        """
        start = time.time()
        result = TaskResult(task=task, mode=mode)

        self._log(f"🎯 Orchestrator started | Mode: {mode.value} | Agents: {list(self.agents.keys())}")

        if mode == CollaborationMode.SEQUENTIAL:
            result = await self._run_sequential(task, result)
        elif mode == CollaborationMode.PARALLEL:
            result = await self._run_parallel(task, result)
        elif mode == CollaborationMode.DEBATE:
            result = await self._run_debate(task, result)

        result.elapsed_time = time.time() - start
        self._log(f"✅ Orchestrator completed in {result.elapsed_time:.2f}s")

        return result

    async def _run_sequential(self, task: str, result: TaskResult) -> TaskResult:
        """Sequential: each agent processes the task, passing results forward."""
        context = ""
        for name, agent in self.agents.items():
            self._log(f"\n👤 {name} ({agent.role}) working...")
            output = await agent.execute(task, context)
            result.agent_results[name] = output
            context += f"\n\n[{name}]: {output}"
            self._log(f"  ✅ {name} completed")

        # Final synthesis
        result.final_answer = await self._synthesize(task, result.agent_results)
        return result

    async def _run_parallel(self, task: str, result: TaskResult) -> TaskResult:
        """Parallel: all agents work simultaneously on the same task."""
        self._log("⚡ Running agents in parallel...")

        async def run_agent(name: str, agent: SpecializedAgent):
            output = await agent.execute(task)
            return name, output

        tasks = [run_agent(name, agent) for name, agent in self.agents.items()]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        for item in outputs:
            if isinstance(item, Exception):
                self._log(f"  ❌ Agent failed: {item}", style="red")
            else:
                name, output = item
                result.agent_results[name] = output
                self._log(f"  ✅ {name} completed")

        result.final_answer = await self._synthesize(task, result.agent_results)
        return result

    async def _run_debate(self, task: str, result: TaskResult, rounds: int = 2) -> TaskResult:
        """Debate: agents review each other's work and refine."""
        agent_names = list(self.agents.keys())
        if len(agent_names) < 2:
            return await self._run_sequential(task, result)

        # Round 1: Initial answers
        self._log("\n📢 Debate Round 1: Initial answers")
        answers = {}
        for name, agent in self.agents.items():
            output = await agent.execute(task)
            answers[name] = output
            result.agent_results[name] = output
            self._log(f"  👤 {name}: {output[:100]}...")

        # Round 2+: Cross-review
        for round_num in range(rounds):
            self._log(f"\n📢 Debate Round {round_num + 2}: Cross-review")
            reviews = {}
            for name, agent in self.agents.items():
                # Each agent reviews others' answers
                other_answers = {k: v for k, v in answers.items() if k != name}
                review_prompt = f"Task: {task}\n\nOther agents' answers:\n"
                for other_name, other_answer in other_answers.items():
                    review_prompt += f"\n[{other_name}]: {other_answer[:500]}\n"

                review_prompt += f"\nYour previous answer:\n{answers[name][:500]}\n"
                review_prompt += "\nConsidering the other perspectives, refine your answer. If you agree with a better answer from another agent, adopt it."

                refined = await agent.execute(review_prompt)
                reviews[name] = refined
                self._log(f"  👤 {name} refined answer")

            answers = reviews

        # Store final answers
        for name, answer in answers.items():
            result.agent_results[name] = answer

        result.final_answer = await self._synthesize(task, answers)
        return result

    async def _synthesize(self, task: str, agent_results: dict[str, str]) -> str:
        """Synthesize all agent results into a final answer."""
        results_text = ""
        for name, result in agent_results.items():
            agent = self.agents.get(name)
            role = agent.role if agent else name
            results_text += f"\n[{name} - {role}]:\n{result}\n"

        prompt = f"""Synthesize the following expert contributions into a single, comprehensive answer.

Task: {task}

Expert contributions:
{results_text}

Create a unified, well-structured final answer that incorporates the best insights from each expert. Resolve any contradictions by favoring the most well-reasoned position."""

        response = await self.llm.ainvoke([
            SystemMessage(content="You are a synthesis expert. Combine multiple perspectives into a coherent, comprehensive answer."),
            HumanMessage(content=prompt),
        ])

        return response.content

    def _log(self, msg: str, style: str = "") -> None:
        if self.verbose:
            if style:
                console.print(msg, style=style)
            else:
                console.print(msg)
