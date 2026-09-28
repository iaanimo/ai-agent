"""
Demo: Chain of Thought (CoT) Reasoning
========================================
演示不同 CoT 策略：Zero-shot、Few-shot、Self-consistency、Tree-of-Thought。
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reasoning import ChainOfThought, CoTStrategy


async def main():
    print("=" * 60)
    print("🧠 Chain of Thought (CoT) Demo")
    print("=" * 60)

    problem = """
    A farmer has a rectangular field that is 120 meters long and 80 meters wide.
    He wants to divide it into smaller square plots, each with the maximum possible side length,
    without any leftover space. What is the side length of each square plot, and how many plots will there be?
    """

    # Strategy 1: Zero-shot CoT
    print("\n\n📌 Strategy 1: Zero-shot CoT")
    print("-" * 40)
    cot = ChainOfThought(strategy=CoTStrategy.ZERO_SHOT)
    chain = await cot.reason(problem)
    print(chain.to_display())

    # Strategy 2: Self-consistency
    print("\n\n📌 Strategy 2: Self-consistency (3 paths)")
    print("-" * 40)
    cot_sc = ChainOfThought(strategy=CoTStrategy.SELF_CONSISTENCY, num_paths=3)
    chain_sc = await cot_sc.reason(problem)
    print(chain_sc.to_display())

    # Strategy 3: Tree of Thought
    print("\n\n📌 Strategy 3: Tree of Thought")
    print("-" * 40)
    cot_tot = ChainOfThought(strategy=CoTStrategy.TREE_OF_THOUGHT)
    chain_tot = await cot_tot.reason(problem)
    print(chain_tot.to_display())


if __name__ == "__main__":
    asyncio.run(main())
