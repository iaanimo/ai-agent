"""
Multi-Agent Module
===================
Multi-agent collaboration with specialized agents.
"""

from .orchestrator import Orchestrator, TaskResult
from .agents.researcher import ResearcherAgent
from .agents.coder import CoderAgent
from .agents.reviewer import ReviewerAgent

__all__ = [
    "Orchestrator", "TaskResult",
    "ResearcherAgent", "CoderAgent", "ReviewerAgent",
]
