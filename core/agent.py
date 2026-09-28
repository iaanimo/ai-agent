"""
Agent — The Brain
==================
核心智能体，整合规划、记忆、推理、工具调用的完整循环。
支持两种运行模式：
  - ReAct 模式：Reason → Act → Observe 循环
  - Plan-and-Execute 模式：先规划，再逐步执行
"""

import json
import time
from datetime import datetime
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from .llm_factory import create_llm
from .memory import Memory
from .planner import Planner, Plan, PlanStep, StepStatus
from .reasoning import ChainOfThought, CoTStrategy, ReasoningChain
from .executor import ToolExecutor, FunctionCaller, ToolResult
from tools.file_ops import resolve_data_path

console = Console()


def current_datetime_str() -> str:
    """Current date and time, injected into prompts so the LLM knows what
    'now' really is (models otherwise treat their training cutoff as 'now',
    and without the clock time they can't reason about time-sensitive topics
    like whether the stock market is open yet)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M (%A)")


# ─── 运行模式 ─────────────────────────────────────────────────

class AgentMode(str, Enum):
    REACT = "react"              # Reason-Act-Observe 循环
    PLAN_AND_EXECUTE = "plan"    # 先规划再执行
    DIRECT = "direct"            # 直接回答（无工具）


# ─── 系统提示 ─────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are an intelligent AI assistant. You solve problems using a Reason-Act-Observe cycle.

Today's date and time is {today}. Your training data has a cutoff, so for anything
recent or time-sensitive you MUST use the search tool instead of relying on memory.

Before searching time-sensitive topics (stock market, weather, live events), reason about
what the current time implies — has it happened, is it ongoing, or not yet started? Search
for what is meaningful right now (e.g. before the A-share market opens at 09:30, that is
the latest close / pre-market news, not "today's intraday action").

For each step:
1. **Reason**: Think about what to do next
2. **Act**: Use a tool or provide a final answer
3. **Observe**: Review the result and decide next step

Available tools:
{tool_descriptions}

Respond in JSON format for each step:
{{
    "thought": "Your reasoning about what to do",
    "action": {{
        "tool": "tool_name",
        "args": {{"key": "value"}}
    }}
}}

When you have enough information to answer, use:
{{
    "thought": "I now have enough information",
    "action": {{
        "tool": "final_answer",
        "args": {{"answer": "Your comprehensive answer"}}
    }}
}}

To also save the answer as a file, add a "file_path" key to final_answer args,
e.g. {{"answer": "...", "file_path": "report.md"}} → saved under data/report.md.

Important: Always think before acting. Use tools when needed, but don't overuse them.
File paths for tools are relative to the data/ directory (do not prefix with "data/").

Search discipline: perform at most 2-3 searches total, then synthesize what you have
and give the final answer. Search results are often noisy or imperfect — do NOT repeat
similar searches hoping for better results. Once you have reasonable material (even if
not perfect), call 'final_answer'.

Recency awareness: for "today / latest / recent" tasks, evaluate the DATES shown in search
results. Prefer information from the last few days, and clearly distinguish "today's news"
from "recent weeks' news" by labeling items with their actual dates. If you cannot find
genuinely fresh information (e.g. it is early morning and nothing new is published yet),
say so honestly instead of presenting older items as today's news.
"""

DIRECT_SYSTEM_PROMPT = """You are an intelligent AI assistant. Answer the user's question directly and thoroughly.

Today's date and time is {today}. Your training data has a cutoff; if the question needs
recent or time-sensitive information, say so honestly instead of guessing. For
time-sensitive topics, reason about what the current time implies before answering.

{extra_context}
"""


# ─── Agent 状态 ────────────────────────────────────────────────

@dataclass
class AgentState:
    """Tracks the agent's execution state."""
    mode: AgentMode = AgentMode.REACT
    current_plan: Optional[Plan] = None
    step_count: int = 0
    total_tool_calls: int = 0
    start_time: float = 0.0
    errors: list[str] = field(default_factory=list)

    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time else 0


# ─── 主 Agent 类 ──────────────────────────────────────────────

class Agent:
    """
    The main AI Agent.
    Integrates planning, memory, reasoning, and tool execution.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: list = None,
        mode: AgentMode = AgentMode.REACT,
        enable_memory: bool = True,
        enable_cot: bool = True,
        cot_strategy: CoTStrategy = CoTStrategy.ZERO_SHOT,
        max_iterations: int = 10,
        verbose: bool = True,
        rag: Optional["RAGRetriever"] = None,
        auto_memory: bool = True,
    ):
        # LLM
        self.llm = llm or create_llm()

        # Tools
        self.executor = ToolExecutor()
        if tools:
            self.executor.register_tools(tools)
        self.function_caller = FunctionCaller(self.llm, self.executor)

        # Memory
        self.memory = Memory() if enable_memory else None

        # Planner
        self.planner = Planner(self.llm, self.executor.get_tool_names() or ["reason"])

        # Chain of Thought
        self.cot = ChainOfThought(self.llm, strategy=cot_strategy) if enable_cot else None

        # Config
        self.mode = mode
        self.max_iterations = max_iterations
        self.verbose = verbose

        # RAG engine (optional; wired up by the caller, e.g. main.py)
        self.rag = rag

        # Auto memory: after each turn, quietly extract durable user facts
        # into long-term memory (the "JARVIS remembers you" layer).
        self.auto_memory = auto_memory

    def register_tools(self, tools: list) -> None:
        """Register tools after initialization."""
        self.executor.register_tools(tools)
        self.planner.available_tools = self.executor.get_tool_names()

    # ─── 主入口 ────────────────────────────────────────────────

    async def run(self, query: str, mode: Optional[AgentMode] = None) -> str:
        """
        Main entry point. Process a user query.

        Args:
            query: The user's question or request.
            mode: Override the default mode for this run.

        Returns:
            The agent's final answer as a string.
        """
        mode = mode or self.mode
        state = AgentState(mode=mode, start_time=time.time())

        if self.memory:
            self.memory.add_user_message(query)

        self._log(f"🚀 Agent started in [bold]{mode.value}[/bold] mode")
        self._log(f"📝 Query: {query}")

        try:
            if mode == AgentMode.REACT:
                result = await self._run_react(query, state)
            elif mode == AgentMode.PLAN_AND_EXECUTE:
                result = await self._run_plan_and_execute(query, state)
            else:
                result = await self._run_direct(query, state)
        except Exception as e:
            result = f"Agent error: {type(e).__name__}: {str(e)}"
            state.errors.append(str(e))
            self._log(f"❌ Error: {e}", style="red")

        if self.memory:
            self.memory.add_ai_message(result)

        # JARVIS-style: after each turn, quietly remember durable facts about the user
        if self.memory and self.auto_memory:
            await self._extract_memories(query, result)

        self._log(f"✅ Completed in {state.elapsed():.2f}s | Steps: {state.step_count} | Tools: {state.total_tool_calls}")

        return result

    # ─── ReAct 模式 ────────────────────────────────────────────

    async def _run_react(self, query: str, state: AgentState) -> str:
        """
        ReAct loop: Reason → Act → Observe, repeat until final answer.
        """
        system_prompt = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=self.executor.get_tool_descriptions(),
            today=current_datetime_str(),
        )

        messages = [SystemMessage(content=system_prompt)]

        # Add memory context
        if self.memory:
            ctx = self.memory.get_full_context(query)
            if ctx:
                messages.append(SystemMessage(content=f"Context from memory:\n{ctx}"))

        messages.append(HumanMessage(content=query))

        for iteration in range(self.max_iterations):
            state.step_count += 1
            self._log(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # Get LLM response
            response = await self.llm.ainvoke(messages)
            parsed = self._parse_action(response.content)

            if parsed is None:
                # Couldn't parse as an action. Ask once for the required JSON
                # format before falling back to treating it as the answer.
                if iteration < self.max_iterations - 1:
                    self._log("⚠️ Unparseable response — requesting JSON action format", style="yellow")
                    messages.append(AIMessage(content=response.content))
                    messages.append(HumanMessage(
                        content="Your last response was not in the required format. "
                                "Respond with ONLY a JSON object: "
                                '{"thought": "...", "action": {"tool": "...", "args": {...}}}. '
                                "If you already have the answer, use the 'final_answer' tool."
                    ))
                    continue
                self._log(f"💬 Direct response (no tool call)")
                return response.content

            thought = parsed.get("thought", "")
            action = parsed.get("action", {})
            tool_name = action.get("tool", "")
            tool_args = action.get("args", {})

            self._log(f"🧠 Thought: {thought}")

            # Check for final answer
            if tool_name == "final_answer":
                answer = tool_args.get("answer", thought)
                # Optional: write the answer to a file (relative to data/)
                file_path = tool_args.get("file_path")
                if file_path:
                    try:
                        target = resolve_data_path(file_path)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(answer, encoding="utf-8")
                        self._log(f"📄 Report written to {target}", style="green")
                        answer = f"{answer}\n\n📄 报告已保存到 {target}"
                    except Exception as e:
                        self._log(f"⚠️ Failed to write report: {e}", style="yellow")
                        answer = f"{answer}\n\n(⚠️ 保存文件失败: {e})"
                self._log(f"💬 Final answer provided")
                return answer

            # Execute tool
            self._log(f"🔧 Calling tool: [cyan]{tool_name}[/cyan]({json.dumps(tool_args, ensure_ascii=False)[:100]})")
            result = await self.executor.execute(tool_name, tool_args)
            state.total_tool_calls += 1

            if result.success:
                self._log(f"✅ Tool result: {str(result.output)[:200]}")
            else:
                self._log(f"❌ Tool error: {result.error}", style="red")

            # Add to conversation for next iteration
            messages.append(AIMessage(content=response.content))
            observation = f"Tool '{tool_name}' result: {result.output if result.success else f'ERROR: {result.error}'}"
            messages.append(HumanMessage(content=f"Observation: {observation}\n\nContinue reasoning. If you have enough information, use 'final_answer' tool."))

            # Update working memory
            if self.memory:
                self.memory.working.add_observation(f"Used {tool_name}: {str(result.output)[:200]}")

        return "Reached maximum iterations. Here's what I found so far:\n" + self._summarize_findings(messages)

    # ─── Plan-and-Execute 模式 ─────────────────────────────────

    async def _run_plan_and_execute(self, query: str, state: AgentState) -> str:
        """
        Plan first, then execute each step.
        Supports dynamic replanning on failure.
        """
        # Phase 1: Create plan
        self._log("\n📋 Phase 1: Planning...")
        context = ""
        if self.memory:
            context = self.memory.get_full_context(query)

        plan = await self.planner.create_plan(query, context)
        state.current_plan = plan

        if self.verbose:
            console.print(Panel(plan.to_display(), title="Execution Plan", border_style="blue"))

        if not plan.steps:
            return await self._run_direct(query, state)

        # Phase 2: Execute steps
        self._log("\n⚡ Phase 2: Executing...")
        max_replans = 3

        for replan_count in range(max_replans + 1):
            while not plan.is_complete() and not plan.has_failed():
                step = plan.get_next_step()
                if step is None:
                    break

                self._log(f"\n  ▶ Step {step.id}: {step.description}")
                step.status = StepStatus.IN_PROGRESS
                state.step_count += 1

                # Execute step
                if step.action == "reason":
                    result = await self._execute_reason_step(step, query, plan)
                elif step.action == "final_answer":
                    result = await self._execute_final_step(step, query, plan)
                    plan.mark_step(step.id, StepStatus.COMPLETED, result)
                    return result
                elif step.action == "rag_search":
                    result = await self._execute_rag_step(step, query)
                else:
                    result = await self._execute_tool_step(step, state)

                if result is not None and not isinstance(result, str):
                    result = str(result)

                if step.status != StepStatus.FAILED:
                    plan.mark_step(step.id, StepStatus.COMPLETED, result)
                    self._log(f"  ✅ Step {step.id} completed: {str(result)[:150]}")
                else:
                    self._log(f"  ❌ Step {step.id} failed: {step.error}", style="red")

                    # Replan
                    if replan_count < max_replans:
                        self._log(f"\n🔄 Replanning (attempt {replan_count + 1}/{max_replans})...")
                        new_plan = await self.planner.replan(plan, step, context)
                        # Preserve completed steps' results into the revised plan
                        plan = self._merge_replanned_plan(plan, new_plan)
                        state.current_plan = plan
                        if self.verbose:
                            console.print(Panel(plan.to_display(), title="Revised Plan", border_style="yellow"))
                        break  # Break inner while, continue outer for
                    else:
                        return f"Plan execution failed after {max_replans} replans. Last error: {step.error}"

            if plan.is_complete():
                break

        # Summarize results
        results = []
        for step in plan.steps:
            if step.result:
                results.append(f"- {step.description}: {step.result}")
        return "\n".join(results) if results else "Plan completed but no results generated."

    async def _execute_reason_step(self, step: PlanStep, query: str, plan: Plan) -> str:
        """Execute a reasoning step using CoT."""
        if self.cot:
            context_parts = [f"Goal: {plan.goal}"]
            for s in plan.steps:
                if s.result and s.id < step.id:
                    context_parts.append(f"Step {s.id} result: {s.result[:300]}")

            chain = await self.cot.reason(
                step.description,
                context="\n".join(context_parts),
            )
            if self.verbose:
                console.print(Panel(chain.to_display(), title="Chain of Thought", border_style="green"))
            return chain.conclusion
        else:
            # Simple reasoning without CoT
            response = await self.llm.ainvoke([
                SystemMessage(content="Think carefully and provide a clear analysis."),
                HumanMessage(content=step.description),
            ])
            return response.content

    async def _execute_tool_step(self, step: PlanStep, state: AgentState) -> str:
        """Execute a tool step."""
        tool_name = step.action
        args = step.args

        # If args contain a "query" key that references previous steps, resolve it
        if "query" in args and isinstance(args["query"], str) and args["query"].startswith("$step"):
            ref_id = int(args["query"].replace("$step", ""))
            if 0 <= ref_id < len(state.current_plan.steps):
                args["query"] = state.current_plan.steps[ref_id].result or args["query"]

        result = await self.executor.execute(tool_name, args)
        state.total_tool_calls += 1

        if result.success:
            return str(result.output)
        else:
            step.status = StepStatus.FAILED
            step.error = result.error
            return None

    async def _execute_final_step(self, step: PlanStep, query: str, plan: Plan) -> str:
        """Generate final answer based on all step results."""
        results_summary = []
        for s in plan.steps:
            if s.result and s.id != step.id:
                results_summary.append(f"- {s.description}: {s.result[:500]}")

        prompt = f"""Based on the following research results, provide a comprehensive answer.

Today's date and time is {current_datetime_str()}. If the question needs recent information that
the research results do not cover, say so honestly.

Original question: {query}

Research results:
{chr(10).join(results_summary)}

Provide a clear, complete answer."""

        response = await self.llm.ainvoke([
            SystemMessage(content="You are a helpful assistant. Synthesize the research into a clear answer."),
            HumanMessage(content=prompt),
        ])
        answer = response.content

        # Optional: write the synthesized answer to a file (relative to data/)
        file_path = step.args.get("file_path")
        if file_path:
            try:
                target = resolve_data_path(file_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(answer, encoding="utf-8")
                self._log(f"📄 Report written to {target}", style="green")
                answer = f"{answer}\n\n📄 报告已保存到 {target}"
            except Exception as e:
                self._log(f"⚠️ Failed to write report: {e}", style="yellow")
        return answer

    async def _execute_rag_step(self, step: PlanStep, query: str) -> str:
        """Execute a RAG retrieval step using the wired-up RAG engine."""
        rag_query = step.args.get("query", step.description)
        if self.rag is None:
            # No RAG engine wired up — fall back to LLM reasoning.
            response = await self.llm.ainvoke([
                HumanMessage(content=f"Please research and answer: {step.description}")
            ])
            return response.content
        try:
            return await self.rag.query(rag_query)
        except Exception as e:
            return f"RAG search failed: {type(e).__name__}: {str(e)}"

    def _merge_replanned_plan(self, old_plan: Plan, new_plan: Plan) -> Plan:
        """Merge a replanned plan, preserving completed steps' results.

        The replanner generates a brand-new plan with no results, which would
        otherwise throw away everything already done. This keeps completed /
        skipped steps (with their results) at the front, then appends the new
        plan's steps with renumbered ids and remapped dependencies.
        """
        prefix = [s for s in old_plan.steps if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)]
        offset = len(prefix)

        merged = Plan(goal=new_plan.goal)
        merged.revision = new_plan.revision
        for i, s in enumerate(prefix):
            merged.steps.append(PlanStep(
                id=i,
                description=s.description,
                action=s.action,
                args=dict(s.args),
                status=s.status,
                result=s.result,
                error=s.error,
                dependencies=[],
            ))
        for j, s in enumerate(new_plan.steps):
            merged.steps.append(PlanStep(
                id=offset + j,
                description=s.description,
                action=s.action,
                args=dict(s.args),
                status=StepStatus.PENDING,
                dependencies=[d + offset for d in s.dependencies],
            ))
        return merged

    # ─── Direct 模式 ───────────────────────────────────────────

    async def _run_direct(self, query: str, state: AgentState) -> str:
        """Direct mode: answer without tools."""
        extra = ""
        if self.memory:
            extra = self.memory.get_full_context(query)

        system = DIRECT_SYSTEM_PROMPT.format(extra_context=extra or "No additional context.", today=current_datetime_str())
        messages = [SystemMessage(content=system), HumanMessage(content=query)]

        # Optional CoT
        if self.cot:
            chain = await self.cot.reason(query, extra)
            if self.verbose:
                console.print(Panel(chain.to_display(), title="Chain of Thought", border_style="green"))
            return chain.conclusion

        response = await self.llm.ainvoke(messages)
        state.step_count = 1
        return response.content

    # ─── 辅助方法 ──────────────────────────────────────────────

    def _parse_action(self, content: str) -> Optional[dict]:
        """Parse action JSON from LLM response."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(content)
            if isinstance(data, dict) and "action" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Try to find JSON in text
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
            if isinstance(data, dict) and "action" in data:
                return data
        except (ValueError, json.JSONDecodeError):
            pass

        return None

    def _summarize_findings(self, messages: list) -> str:
        """Summarize findings from conversation history."""
        findings = []
        for msg in messages:
            if isinstance(msg, HumanMessage) and "Observation:" in msg.content:
                findings.append(msg.content.replace("Observation: ", ""))
        return "\n".join(findings[-5:]) if findings else "No findings recorded."

    def _log(self, msg: str, style: str = "") -> None:
        """Log a message if verbose."""
        if self.verbose:
            if style:
                console.print(msg, style=style)
            else:
                console.print(msg)

    # ─── 便捷方法 ──────────────────────────────────────────────

    async def chat(self, message: str) -> str:
        """Simple chat interface (direct mode)."""
        return await self.run(message, mode=AgentMode.DIRECT)

    async def think(self, problem: str) -> ReasoningChain:
        """Pure reasoning without tools."""
        if not self.cot:
            self.cot = ChainOfThought(self.llm)
        return await self.cot.reason(problem)

    def reset(self) -> None:
        """Reset agent state."""
        if self.memory:
            self.memory.clear_all()

    # ─── 自动记忆 ──────────────────────────────────────────────

    async def _extract_memories(self, user_query: str, ai_result: str) -> None:
        """Quietly extract durable facts about the user into long-term memory."""
        if self.memory is None or self.memory.long_term is None:
            return

        # 语义去重：把已记住的事实注入抽取 prompt，让 LLM 判断"意思相同"就不要再存。
        # 比纯文本相似度（memory.store 的 0.8 阈值）更能拦措辞变化带来的重复。
        existing = self.memory.long_term.get_all()[-40:]
        memory_context = "\n".join(f"- {e['content']}" for e in existing) if existing else "(暂无)"

        prompt = f"""Based on this exchange, list durable facts worth remembering about the user —
personal details, preferences, ongoing projects, goals, or constraints.

User: {user_query[:500]}

Assistant: {ai_result[:500]}

You already remember these facts about the user. If something in the exchange
means the same thing as one of these (even if worded differently), DO NOT
include it again.
Already remembered:
{memory_context}

Respond in JSON format:
{{"facts": [
  {{"content": "fact 1", "keywords": ["trigger phrase", "trigger phrase"]}},
  ...
]}}

"keywords" are short phrases the user might say when recalling this fact
(e.g. content "做后端开发" → keywords ["职业", "工作", "做什么"]). Empty list
if nothing is worth remembering. Be conservative: only include facts likely
to be useful in future conversations. Do not echo the assistant's reply."""
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You extract durable facts about a user from a conversation."),
                HumanMessage(content=prompt),
            ])
            for fact in self._parse_facts(response.content):
                content = fact["content"].strip()
                if content and len(content) >= 6:
                    stored = self.memory.store_fact(content, keywords=fact["keywords"])
                    if stored and self.verbose:
                        self._log(f"🧠 Remembered: {content[:80]}", style="cyan")
        except Exception as e:
            self._log(f"⚠️ Memory extraction skipped: {e}", style="yellow")

    def _parse_facts(self, content: str) -> list[dict]:
        """Parse facts from an LLM JSON response.

        Each fact is {"content": str, "keywords": [str]}. Plain-string facts
        (old format) are accepted too.
        """
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            start, end = text.index("{"), text.rindex("}")
            data = json.loads(text[start:end + 1])
            facts = data.get("facts", [])
            result = []
            for f in facts:
                if isinstance(f, str):
                    result.append({"content": f, "keywords": []})
                elif isinstance(f, dict) and f.get("content"):
                    kws = f.get("keywords", [])
                    result.append({
                        "content": f["content"],
                        "keywords": [k for k in kws if isinstance(k, str)],
                    })
            return result
        except (ValueError, json.JSONDecodeError):
            return []
