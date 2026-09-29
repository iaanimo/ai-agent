"""
Coder Agent
=============
Specialized agent for code generation, debugging, and technical solutions.
"""

from ..orchestrator import SpecializedAgent
from core.llm_factory import create_llm
from tools.code_executor import create_code_executor_tool


CODER_PROMPT = """You are an expert Coder Agent. Your role is to:

1. **Design**: Plan clean, maintainable code architecture
2. **Implement**: Write production-quality code with proper error handling
3. **Debug**: Identify and fix issues systematically
4. **Optimize**: Improve performance and readability

Your strengths:
- Clean, well-documented code
- Design patterns and best practices
- Error handling and edge cases
- Performance optimization

When given a task:
1. Understand the requirements thoroughly
2. Design the solution architecture
3. Implement with proper structure
4. Include error handling and comments
5. Provide usage examples

Always:
- Write clean, readable code
- Add docstrings and comments for complex logic
- Handle edge cases and errors
- Follow language-specific best practices
- Suggest tests when appropriate"""


class CoderAgent(SpecializedAgent):
    """Code generation and technical solution specialist. Can execute code."""

    def __init__(self, llm=None, tools=None):
        super().__init__(
            name="Coder",
            role="Code & Technical Solutions",
            llm=llm or create_llm(),
            system_prompt=CODER_PROMPT,
            tools=tools if tools is not None else [create_code_executor_tool()],
        )

    async def implement(self, requirements: str, language: str = "python") -> str:
        """Implement a solution based on requirements."""
        prompt = f"Implement the following in {language}:\n\n{requirements}\n\nProvide complete, working code with comments."
        return await self.execute(prompt)

    async def debug(self, code: str, error: str = None) -> str:
        """Debug code and suggest fixes."""
        prompt = f"Debug this code:\n```\n{code}\n```"
        if error:
            prompt += f"\n\nError message:\n{error}"
        prompt += "\n\nIdentify the issue and provide a fixed version."
        return await self.execute(prompt)

    async def refactor(self, code: str, goals: list[str] = None) -> str:
        """Refactor code for better quality."""
        prompt = f"Refactor this code:\n```\n{code}\n```"
        if goals:
            prompt += "\n\nRefactoring goals:\n" + "\n".join(f"- {g}" for g in goals)
        return await self.execute(prompt)
