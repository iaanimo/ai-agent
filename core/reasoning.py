"""
Chain of Thought (CoT) Reasoning
==================================
思维链推理引擎，支持：
  - Zero-shot CoT（"Let's think step by step"）
  - Few-shot CoT（带示例的推理）
  - Self-consistency（多路径投票）
  - Tree-of-Thought（树形推理探索）
"""

import json
from enum import Enum
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


# ─── CoT 策略 ─────────────────────────────────────────────────

class CoTStrategy(str, Enum):
    ZERO_SHOT = "zero_shot"         # "Let's think step by step"
    FEW_SHOT = "few_shot"           # 带示例推理
    SELF_CONSISTENCY = "self_consistency"  # 多路径投票
    TREE_OF_THOUGHT = "tree_of_thought"    # 树形探索


# ─── 推理链数据结构 ────────────────────────────────────────────

@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    thought: str
    evidence: str = ""
    confidence: float = 0.8

    def to_dict(self) -> dict:
        return {"thought": self.thought, "evidence": self.evidence, "confidence": self.confidence}


@dataclass
class ReasoningChain:
    """A complete reasoning chain."""
    question: str
    steps: list[ReasoningStep] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0

    def add_step(self, thought: str, evidence: str = "", confidence: float = 0.8) -> None:
        self.steps.append(ReasoningStep(thought=thought, evidence=evidence, confidence=confidence))

    def to_display(self) -> str:
        """Human-readable reasoning chain."""
        lines = [f"🧠 Reasoning Chain for: {self.question}", "=" * 50]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"\n  Step {i}: {step.thought}")
            if step.evidence:
                lines.append(f"  Evidence: {step.evidence}")
            lines.append(f"  Confidence: {step.confidence:.0%}")
        if self.conclusion:
            lines.append(f"\n{'=' * 50}")
            lines.append(f"Conclusion: {self.conclusion}")
            lines.append(f"Overall Confidence: {self.confidence:.0%}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "steps": [s.to_dict() for s in self.steps],
            "conclusion": self.conclusion,
            "confidence": self.confidence,
        }


# ─── 提示模板 ─────────────────────────────────────────────────

ZERO_SHOT_COT_PROMPT = """You are a careful, logical reasoner. Think through the problem step by step.

For each reasoning step, output:
- "thought": Your reasoning for this step
- "evidence": Supporting evidence or facts
- "confidence": Your confidence level (0.0 to 1.0)

After all steps, provide your final "conclusion" and overall "confidence".

Respond in JSON format:
{{
    "steps": [
        {{"thought": "...", "evidence": "...", "confidence": 0.9}},
        ...
    ],
    "conclusion": "...",
    "confidence": 0.85
}}

Problem: {problem}
"""

FEW_SHOT_COT_PROMPT = """You are a careful, logical reasoner. Follow the example reasoning style.

Example:
Problem: "If a train travels 60 mph for 2.5 hours, how far does it go?"
Reasoning:
{{"steps": [
    {{"thought": "The formula for distance is: distance = speed × time", "evidence": "Basic physics formula", "confidence": 1.0}},
    {{"thought": "Speed = 60 mph, Time = 2.5 hours", "evidence": "Given in the problem", "confidence": 1.0}},
    {{"thought": "Distance = 60 × 2.5 = 150 miles", "evidence": "Calculation", "confidence": 1.0}}
], "conclusion": "The train travels 150 miles", "confidence": 1.0}}

Now solve:
Problem: {problem}
"""

SELF_CONSISTENCY_PROMPT = """You are a logical reasoner. Solve this problem through independent reasoning paths.

For each path, think independently and reach a conclusion.
After all paths, select the most common conclusion.

Respond in JSON format:
{{
    "paths": [
        {{"steps": [...], "conclusion": "...", "confidence": 0.8}},
        {{"steps": [...], "conclusion": "...", "confidence": 0.9}},
        {{"steps": [...], "conclusion": "...", "confidence": 0.85}}
    ],
    "final_conclusion": "The conclusion agreed upon by majority of paths",
    "agreement_ratio": 0.67
}}

Problem: {problem}
"""


# ─── CoT 引擎 ─────────────────────────────────────────────────

class ChainOfThought:
    """
    Chain of Thought reasoning engine.
    Supports multiple CoT strategies.
    """

    def __init__(self, llm: BaseChatModel, strategy: CoTStrategy = CoTStrategy.ZERO_SHOT, num_paths: int = 3):
        self.llm = llm
        self.strategy = strategy
        self.num_paths = num_paths  # For self-consistency

    async def reason(self, problem: str, context: str = "") -> ReasoningChain:
        """Execute CoT reasoning on a problem."""
        if self.strategy == CoTStrategy.ZERO_SHOT:
            return await self._zero_shot_cot(problem, context)
        elif self.strategy == CoTStrategy.FEW_SHOT:
            return await self._few_shot_cot(problem, context)
        elif self.strategy == CoTStrategy.SELF_CONSISTENCY:
            return await self._self_consistency(problem, context)
        elif self.strategy == CoTStrategy.TREE_OF_THOUGHT:
            return await self._tree_of_thought(problem, context)
        else:
            return await self._zero_shot_cot(problem, context)

    async def _zero_shot_cot(self, problem: str, context: str = "") -> ReasoningChain:
        """Zero-shot Chain of Thought."""
        full_problem = problem
        if context:
            full_problem = f"Context:\n{context}\n\nProblem: {problem}"

        prompt = ZERO_SHOT_COT_PROMPT.format(problem=full_problem)
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a logical reasoning engine."),
            HumanMessage(content=prompt),
        ])

        return self._parse_reasoning(problem, response.content)

    async def _few_shot_cot(self, problem: str, context: str = "") -> ReasoningChain:
        """Few-shot Chain of Thought with examples."""
        full_problem = problem
        if context:
            full_problem = f"Context:\n{context}\n\nProblem: {problem}"

        prompt = FEW_SHOT_COT_PROMPT.format(problem=full_problem)
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a logical reasoning engine following example patterns."),
            HumanMessage(content=prompt),
        ])

        return self._parse_reasoning(problem, response.content)

    async def _self_consistency(self, problem: str, context: str = "") -> ReasoningChain:
        """Self-consistency: multiple independent paths, majority vote."""
        full_problem = problem
        if context:
            full_problem = f"Context:\n{context}\n\nProblem: {problem}"

        prompt = SELF_CONSISTENCY_PROMPT.format(problem=full_problem)
        response = await self.llm.ainvoke([
            SystemMessage(content=f"Generate exactly {self.num_paths} independent reasoning paths."),
            HumanMessage(content=prompt),
        ])

        return self._parse_self_consistency(problem, response.content)

    async def _tree_of_thought(self, problem: str, context: str = "") -> ReasoningChain:
        """
        Tree of Thought: explore multiple branches, evaluate, and select the best.
        Simplified version: generate 3 approaches, evaluate each, pick best.
        """
        # Phase 1: Generate multiple approaches
        approach_prompt = f"""Generate 3 different approaches to solve this problem.
For each approach, outline the key steps.

Problem: {problem}
{f'Context: {context}' if context else ''}

Respond in JSON format:
{{
    "approaches": [
        {{"name": "Approach 1", "outline": "Step-by-step outline...", "estimated_confidence": 0.8}},
        {{"name": "Approach 2", "outline": "...", "estimated_confidence": 0.7}},
        {{"name": "Approach 3", "outline": "...", "estimated_confidence": 0.9}}
    ]
}}"""

        resp1 = await self.llm.ainvoke([
            SystemMessage(content="You are a strategic thinker. Generate diverse approaches."),
            HumanMessage(content=approach_prompt),
        ])

        # Phase 2: Evaluate and select best approach
        eval_prompt = f"""Evaluate these approaches and solve the problem using the best one.

Approaches:
{resp1.content}

Problem: {problem}

Select the best approach, execute it step by step, and provide your conclusion.

Respond in JSON format:
{{
    "selected_approach": "Approach name",
    "reason": "Why this approach is best",
    "steps": [
        {{"thought": "...", "evidence": "...", "confidence": 0.9}},
        ...
    ],
    "conclusion": "...",
    "confidence": 0.85
}}"""

        resp2 = await self.llm.ainvoke([
            SystemMessage(content="You are a careful evaluator and problem solver."),
            HumanMessage(content=eval_prompt),
        ])

        return self._parse_reasoning(problem, resp2.content)

    def _parse_reasoning(self, problem: str, content: str) -> ReasoningChain:
        """Parse reasoning steps from LLM response."""
        chain = ReasoningChain(question=problem)

        try:
            # Try JSON parsing
            content_clean = content.strip()
            if content_clean.startswith("```"):
                lines = content_clean.split("\n")
                content_clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            data = json.loads(content_clean)

            for step_data in data.get("steps", []):
                chain.add_step(
                    thought=step_data.get("thought", ""),
                    evidence=step_data.get("evidence", ""),
                    confidence=step_data.get("confidence", 0.8),
                )
            chain.conclusion = data.get("conclusion", "")
            chain.confidence = data.get("confidence", 0.8)
        except (json.JSONDecodeError, KeyError):
            # Fallback: treat the whole response as a single reasoning step
            chain.add_step(thought=content, confidence=0.7)
            chain.conclusion = content
            chain.confidence = 0.7

        return chain

    def _parse_self_consistency(self, problem: str, content: str) -> ReasoningChain:
        """Parse self-consistency results with majority voting."""
        chain = ReasoningChain(question=problem)

        try:
            content_clean = content.strip()
            if content_clean.startswith("```"):
                lines = content_clean.split("\n")
                content_clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            data = json.loads(content_clean)

            # Parse all paths
            paths = data.get("paths", [])
            for i, path in enumerate(paths):
                steps = path.get("steps", [])
                for step_data in steps:
                    # Handle both dict and string step formats
                    if isinstance(step_data, dict):
                        chain.add_step(
                            thought=f"[Path {i+1}] {step_data.get('thought', '')}",
                            evidence=step_data.get("evidence", ""),
                            confidence=step_data.get("confidence", 0.8),
                        )
                    elif isinstance(step_data, str):
                        chain.add_step(
                            thought=f"[Path {i+1}] {step_data}",
                            confidence=0.8,
                        )
                # Also capture path conclusion
                path_conclusion = path.get("conclusion", "")
                if path_conclusion:
                    chain.add_step(
                        thought=f"[Path {i+1} Conclusion] {path_conclusion}",
                        confidence=path.get("confidence", 0.8),
                    )

            chain.conclusion = data.get("final_conclusion", data.get("conclusion", ""))
            chain.confidence = data.get("agreement_ratio", 0.8)
        except (json.JSONDecodeError, KeyError):
            chain.add_step(thought=content, confidence=0.7)
            chain.conclusion = content
            chain.confidence = 0.7

        return chain
