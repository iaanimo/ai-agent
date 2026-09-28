"""
Demo: Multi-Agent Collaboration
==================================
演示多智能体协作：研究员 + 程序员 + 审查员。
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent import Orchestrator, ResearcherAgent, CoderAgent, ReviewerAgent
from multi_agent.orchestrator import CollaborationMode


async def main():
    print("=" * 60)
    print("👥 Multi-Agent Collaboration Demo")
    print("=" * 60)

    # Create specialized agents
    researcher = ResearcherAgent()
    coder = CoderAgent()
    reviewer = ReviewerAgent()

    # ─── Example 1: Sequential Collaboration ───────────────────
    print("\n\n📌 Example 1: Sequential Collaboration")
    print("  Researcher → Coder → Reviewer")
    print("-" * 40)

    orchestrator = Orchestrator(verbose=True)
    orchestrator.add_agent(researcher)
    orchestrator.add_agent(coder)
    orchestrator.add_agent(reviewer)

    task = "Design and implement a Python function to calculate the Fibonacci sequence up to n terms, with memoization optimization."

    result = await orchestrator.run(task, mode=CollaborationMode.SEQUENTIAL)
    print(f"\n{result.to_display()}")

    # ─── Example 2: Parallel Collaboration ─────────────────────
    print("\n\n📌 Example 2: Parallel Collaboration")
    print("  All agents work simultaneously")
    print("-" * 40)

    orchestrator2 = Orchestrator(verbose=True)
    orchestrator2.add_agent(researcher)
    orchestrator2.add_agent(coder)
    orchestrator2.add_agent(reviewer)

    task2 = "What are the best practices for building a REST API with authentication?"

    result2 = await orchestrator2.run(task2, mode=CollaborationMode.PARALLEL)
    print(f"\n{result2.to_display()}")

    # ─── Example 3: Debate Mode ────────────────────────────────
    print("\n\n📌 Example 3: Debate Mode")
    print("  Agents cross-review and refine")
    print("-" * 40)

    orchestrator3 = Orchestrator(verbose=True)
    orchestrator3.add_agent(researcher)
    orchestrator3.add_agent(coder)

    task3 = "Should we use SQL or NoSQL for a social media application with millions of users?"

    result3 = await orchestrator3.run(task3, mode=CollaborationMode.DEBATE)
    print(f"\n{result3.to_display()}")


if __name__ == "__main__":
    asyncio.run(main())
