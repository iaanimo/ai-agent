"""
Reviewer Agent
===============
Specialized agent for code review, quality assurance, and feedback.
"""

from ..orchestrator import SpecializedAgent
from core.llm_factory import create_llm
from tools.code_executor import create_code_executor_tool
from tools.file_ops import create_file_ops_tool


REVIEWER_PROMPT = """You are an expert Reviewer Agent. Your role is to:

1. **Review**: Thoroughly examine code, plans, and solutions
2. **Evaluate**: Assess quality, correctness, and completeness
3. **Suggest**: Provide actionable improvement recommendations
4. **Approve**: Make clear accept/reject/revise decisions

Your review criteria:
- **Correctness**: Does it solve the problem correctly?
- **Completeness**: Are all requirements addressed?
- **Quality**: Is the code clean, maintainable, and well-structured?
- **Performance**: Are there obvious performance issues?
- **Security**: Are there potential security concerns?
- **Edge Cases**: Are edge cases handled?

When reviewing:
1. Read through the work carefully
2. Identify issues by severity (Critical/Major/Minor/Suggestion)
3. Provide specific, actionable feedback
4. Give an overall assessment with clear recommendation

Your feedback format:
- 🔴 Critical: Must fix before approval
- 🟡 Major: Should fix
- 🟢 Minor: Nice to fix
- 💡 Suggestion: Improvement idea"""


class ReviewerAgent(SpecializedAgent):
    """Code review and quality assurance specialist. Can read files and run code."""

    def __init__(self, llm=None, tools=None):
        super().__init__(
            name="Reviewer",
            role="Review & Quality Assurance",
            llm=llm or create_llm(),
            system_prompt=REVIEWER_PROMPT,
            # flatten: create_file_ops_tool() returns a list of 3 tools
            tools=tools if tools is not None else [create_code_executor_tool(), *create_file_ops_tool()],
        )

    async def review_code(self, code: str, requirements: str = None) -> str:
        """Review code for quality and correctness."""
        prompt = f"Review this code:\n```\n{code}\n```"
        if requirements:
            prompt += f"\n\nRequirements:\n{requirements}"
        prompt += "\n\nProvide a thorough code review."
        return await self.execute(prompt)

    async def review_plan(self, plan: str) -> str:
        """Review an execution plan."""
        prompt = f"Review this plan:\n\n{plan}\n\nEvaluate its feasibility, completeness, and suggest improvements."
        return await self.execute(prompt)

    async def evaluate_solution(self, solution: str, criteria: list[str] = None) -> str:
        """Evaluate a solution against criteria."""
        prompt = f"Evaluate this solution:\n\n{solution}"
        if criteria:
            prompt += f"\n\nEvaluation criteria:\n" + "\n".join(f"- {c}" for c in criteria)
        prompt += "\n\nProvide a detailed evaluation with scores and recommendations."
        return await self.execute(prompt)
