"""
RAG Retriever
==============
High-level RAG interface: load documents → embed → retrieve → augment prompt.
"""

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .document_loader import DocumentLoader
from .vectorstore import VectorStore
from config.settings import get_settings


class RAGRetriever:
    """
    End-to-end RAG pipeline.

    Usage:
        rag = RAGRetriever(llm=my_llm)
        rag.ingest("./knowledge_base/")  # Load documents
        answer = await rag.query("What is...?")  # Ask questions
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        vectorstore: Optional[VectorStore] = None,
        loader: Optional[DocumentLoader] = None,
        top_k: int = 5,
    ):
        from core.llm_factory import get_llm
        self.llm = llm or get_llm()
        self.vectorstore = vectorstore or VectorStore()
        self.loader = loader or DocumentLoader()
        self.top_k = top_k

    def ingest(self, source: str, glob_pattern: str = "**/*.*") -> int:
        """
        Ingest documents into the vector store.

        Args:
            source: File path, directory path, or URL.
            glob_pattern: Pattern for directory loading.

        Returns:
            Number of chunks added.
        """
        path = Path(source) if not source.startswith("http") else None

        if source.startswith("http"):
            docs = self.loader.load_url(source)
        elif path and path.is_dir():
            docs = self.loader.load_directory(source, glob_pattern)
        elif path and path.is_file():
            docs = self.loader.load_file(source)
        else:
            docs = self.loader.load_text(source)

        if docs:
            count = self.vectorstore.add_documents(docs)
            print(f"✅ Ingested {count} chunks from {source}")
            return count
        return 0

    def retrieve(self, query: str, top_k: int = None) -> list[Document]:
        """Retrieve relevant documents for a query."""
        k = top_k or self.top_k
        return self.vectorstore.search(query, k=k)

    async def query(self, question: str, top_k: int = None) -> str:
        """
        Full RAG pipeline: retrieve → augment → generate.

        Args:
            question: The user's question.
            top_k: Number of documents to retrieve.

        Returns:
            The LLM's answer based on retrieved context.
        """
        # Retrieve relevant docs
        docs = self.retrieve(question, top_k)
        if not docs:
            return "No relevant documents found. Please ingest documents first using rag.ingest()."

        # Build context from retrieved docs
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(f"[Document {i}] (Source: {source})\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        # Generate answer with context
        prompt = f"""Answer the question based on the following context. If the context doesn't contain enough information, say so honestly.

Context:
{context}

Question: {question}

Provide a clear, accurate answer based on the context above."""

        response = await self.llm.ainvoke([
            SystemMessage(content="You are a helpful assistant that answers questions based on provided context. Always cite which document(s) you used."),
            HumanMessage(content=prompt),
        ])

        return response.content

    def get_context_string(self, query: str, top_k: int = None) -> str:
        """Get retrieved context as a string (for integration with Agent)."""
        docs = self.retrieve(query, top_k)
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Doc {i}] {doc.page_content[:500]}")
        return "\n\n".join(parts)


def create_rag_retriever(**kwargs) -> RAGRetriever:
    """Factory function to create a RAG retriever."""
    return RAGRetriever(**kwargs)
