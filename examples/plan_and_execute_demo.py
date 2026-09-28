"""
Demo: Plan-and-Execute Mode
==============================
演示 Agent 的规划-执行模式：先制定计划，再逐步执行。
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import Agent, AgentMode
from tools import get_all_tools


async def main():
    print("=" * 60)
    print("📋 Plan-and-Execute Demo")
    print("=" * 60)

    # Create agent in plan mode
    agent = Agent(
        mode=AgentMode.PLAN_AND_EXECUTE,
        tools=get_all_tools(),
        max_iterations=10,
        verbose=True,
    )

    # Complex task that benefits from planning
    task = """
    I need to analyze the number 2024:
    1. Find all prime factors of 2024
    2. Calculate the sum of its digits
    3. Determine if it's a perfect square
    4. Find the next 3 prime numbers after 2024
    5. Summarize all findings
    """

    print(f"\n🎯 Task: {task}")
    result = await agent.run(task)
    print(f"\n{'=' * 60}")
    print(f"📋 Final Result:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
