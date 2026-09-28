"""
Demo: Single Agent with ReAct Mode
=====================================
演示单个 Agent 的 ReAct（推理-行动-观察）循环。
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import Agent, AgentMode
from tools import get_all_tools


async def main():
    print("=" * 60)
    print("🤖 Single Agent Demo — ReAct Mode")
    print("=" * 60)

    # Create agent with all built-in tools
    agent = Agent(
        mode=AgentMode.REACT,
        tools=get_all_tools(),
        max_iterations=5,
        verbose=True,
    )

    # Example 1: Math question (will use calculator tool)
    print("\n\n📌 Example 1: Math Question")
    print("-" * 40)
    result = await agent.run("What is the square root of 144 plus 25% of 800?")
    print(f"\n🎯 Answer: {result}")

    # Example 2: Code generation (will use code_executor)
    print("\n\n📌 Example 2: Code Task")
    print("-" * 40)
    agent.reset()
    result = await agent.run("Write Python code to find the first 10 prime numbers and show the result.")
    print(f"\n🎯 Answer: {result}")

    # Example 3: Direct chat (no tools needed)
    print("\n\n📌 Example 3: Direct Chat")
    print("-" * 40)
    result = await agent.chat("Explain what a neural network is in 3 sentences.")
    print(f"\n🎯 Answer: {result}")


if __name__ == "__main__":
    asyncio.run(main())
