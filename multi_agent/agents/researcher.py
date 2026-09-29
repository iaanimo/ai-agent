"""
Researcher Agent
=================
Specialized agent for research, analysis, and information gathering.
"""

from ..orchestrator import SpecializedAgent
from core.llm_factory import create_llm
from tools.search import create_search_tool


RESEARCHER_PROMPT = """You are an expert Researcher Agent. Your role is to:

1. **Research**: Thoroughly investigate topics, gather facts, and find relevant information
2. **Analyze**: Break down complex problems into components and analyze each
3. **Synthesize**: Combine findings into clear, structured insights
4. **Cite**: Always provide reasoning and evidence for your conclusions

Your strengths:
- Deep analytical thinking
- Comprehensive information gathering
- Structured problem decomposition
- Evidence-based reasoning

When given a task:
1. Break it into research questions
2. For each question, provide your analysis
3. Synthesize findings into actionable insights
4. Highlight any uncertainties or areas needing more research

Always be thorough but concise. Focus on accuracy over verbosity."""


class ResearcherAgent(SpecializedAgent):
    """Research and analysis specialist. Can search the web."""

    def __init__(self, llm=None, tools=None):
        super().__init__(
            name="Researcher",
            role="Research & Analysis",
            llm=llm or create_llm(),
            system_prompt=RESEARCHER_PROMPT,
            tools=tools if tools is not None else [create_search_tool()],
        )

    async def deep_research(self, topic: str, aspects: list[str] = None) -> str:
        """Perform deep research on a topic."""
        prompt = f"Deep research on: {topic}"
        if aspects:
            prompt += "\n\nSpecific aspects to investigate:\n" + "\n".join(f"- {a}" for a in aspects)
        prompt += "\n\nProvide a comprehensive research report."
        return await self.execute(prompt)

    async def compare_options(self, options: list[str], criteria: list[str] = None) -> str:
        """Compare multiple options against criteria."""
        prompt = "Compare the following options:\n" + "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))
        if criteria:
            prompt += "\n\nEvaluation criteria:\n" + "\n".join(f"- {c}" for c in criteria)
        prompt += "\n\nProvide a detailed comparison table and recommendation."
        return await self.execute(prompt)
