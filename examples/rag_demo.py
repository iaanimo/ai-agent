"""
Demo: RAG (Retrieval-Augmented Generation)
============================================
演示 RAG 流程：文档加载 → 向量化 → 检索增强生成。
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import RAGRetriever


async def main():
    print("=" * 60)
    print("📚 RAG Demo — Retrieval-Augmented Generation")
    print("=" * 60)

    # Create RAG retriever
    rag = RAGRetriever()

    # Step 1: Ingest some knowledge
    print("\n📥 Step 1: Ingesting knowledge...")

    # Option A: Ingest from text directly
    knowledge_chunks = [
        "Python is a high-level, interpreted programming language. Created by Guido van Rossum and first released in 1991.",
        "Python's design philosophy emphasizes code readability with its notable use of significant indentation.",
        "Python supports multiple programming paradigms, including structured, object-oriented, and functional programming.",
        "LangChain is a framework for developing applications powered by language models. It provides tools for chains, agents, and RAG.",
        "RAG (Retrieval-Augmented Generation) combines retrieval from a knowledge base with LLM generation for more accurate answers.",
        "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.",
        "Chain of Thought (CoT) prompting encourages LLMs to show their reasoning process step by step.",
        "ReAct (Reasoning + Acting) is a framework where agents alternate between reasoning about tasks and taking actions.",
    ]

    for chunk in knowledge_chunks:
        rag.vectorstore.add_texts([chunk])

    print(f"  ✅ Ingested {len(knowledge_chunks)} knowledge chunks")

    # Step 2: Query
    print("\n🔍 Step 2: Querying...")

    questions = [
        "What is Python and who created it?",
        "How does RAG work?",
        "What is the ReAct framework?",
    ]

    for q in questions:
        print(f"\n  ❓ {q}")
        answer = await rag.query(q)
        print(f"  💡 {answer[:300]}")

    # Step 3: Ingest from files (example)
    print("\n\n📥 Step 3: File ingestion example (commented out)")
    print("  To ingest files:")
    print('    rag.ingest("./path/to/document.pdf")')
    print('    rag.ingest("./path/to/knowledge_base/")')


if __name__ == "__main__":
    asyncio.run(main())
