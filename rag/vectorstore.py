"""
Vector Store
=============
Unified vector store interface supporting FAISS and ChromaDB.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import get_settings


class SimpleEmbeddings(Embeddings):
    """Lightweight TF-IDF-style embeddings using only stdlib + numpy.
    Good enough for demo/small-scale RAG. No heavy dependencies."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._vocab: dict[str, int] = {}

    def _tokenize(self, text: str) -> list[str]:
        import re
        text = text.lower()
        tokens = re.findall(r'[a-z0-9\u4e00-\u9fff]+', text)
        return tokens

    def _vectorize(self, tokens: list[str]) -> list[float]:
        import hashlib
        vec = [0.0] * self.dim
        for token in tokens:
            # Hash token to a consistent index
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        # Normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(self._tokenize(t)) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(self._tokenize(text))


def _get_embeddings(model_name: str = None) -> Embeddings:
    """Create an embedding model. Priority: sentence-transformers > API > simple hash."""
    settings = get_settings()
    model_name = model_name or settings.embedding.model_name

    # 1. Try local HuggingFace embeddings
    if settings.embedding.use_local or "sentence-transformers" in model_name:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        except ImportError:
            pass

    # 2. Try OpenAI-compatible API (only if provider has embedding endpoint)
    if settings.llm.provider == "openai" and settings.llm.openai_api_key:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=settings.llm.openai_api_key,
                openai_api_base=settings.llm.openai_base_url,
            )
        except Exception:
            pass

    # 3. Fallback: simple hash-based embeddings (no dependencies)
    print("Using simple hash embeddings (install sentence-transformers for better quality)")
    return SimpleEmbeddings()


class VectorStore:
    """
    Unified vector store wrapper.
    Supports FAISS (in-memory/file) and ChromaDB (persistent).
    """

    def __init__(
        self,
        store_type: str = None,
        persist_dir: str = None,
        collection_name: str = None,
        embedding_model: str = None,
    ):
        settings = get_settings()
        self.store_type = store_type or settings.vector_store.store_type
        self.persist_dir = Path(persist_dir or settings.vector_store.persist_dir)
        self.collection_name = collection_name or settings.vector_store.collection_name
        self.embedding_model = embedding_model or settings.embedding.model_name
        # Embeddings are created lazily on first use, so constructing a
        # VectorStore (or RAGRetriever) is cheap and doesn't download models.
        self.embeddings = None
        self._store = None

    def _init_store(self):
        """Initialize the underlying vector store."""
        if self._store is not None:
            return

        if self.embeddings is None:
            self.embeddings = _get_embeddings(self.embedding_model)

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        if self.store_type == "faiss":
            index_path = self.persist_dir / "faiss_index"
            self._faiss_path = str(index_path)  # Always set, needed for save
            if (index_path / "index.faiss").exists():
                from langchain_community.vectorstores import FAISS
                self._store = FAISS.load_local(
                    str(index_path), self.embeddings, allow_dangerous_deserialization=True
                )
            else:
                # Will be created on first add
                self._store = None

        elif self.store_type == "chroma":
            import chromadb
            from langchain_community.vectorstores import Chroma
            client = chromadb.PersistentClient(path=str(self.persist_dir / "chroma_db"))
            self._store = Chroma(
                client=client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")

    def add_documents(self, documents: list[Document]) -> int:
        """Add documents to the vector store."""
        self._init_store()

        if self.store_type == "faiss":
            if self._store is None:
                from langchain_community.vectorstores import FAISS
                self._store = FAISS.from_documents(documents, self.embeddings)
            else:
                self._store.add_documents(documents)
            # Save to disk
            self._store.save_local(self._faiss_path)

        elif self.store_type == "chroma":
            self._store.add_documents(documents)

        return len(documents)

    def add_texts(self, texts: list[str], metadatas: list[dict] = None) -> int:
        """Add raw texts to the vector store."""
        self._init_store()
        if self.store_type == "faiss":
            if self._store is None:
                from langchain_community.vectorstores import FAISS
                self._store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            else:
                self._store.add_texts(texts, metadatas=metadatas)
            self._store.save_local(self._faiss_path)
        elif self.store_type == "chroma":
            self._store.add_texts(texts, metadatas=metadatas)
        return len(texts)

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Search for similar documents."""
        self._init_store()
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=k)

    def search_with_score(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """Search with relevance scores."""
        self._init_store()
        if self._store is None:
            return []
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def as_retriever(self, search_kwargs: dict = None):
        """Get a LangChain retriever interface."""
        self._init_store()
        if self._store is None:
            return None
        return self._store.as_retriever(search_kwargs=search_kwargs or {"k": 5})

    def get_doc_count(self) -> int:
        """Get the number of documents in the store."""
        self._init_store()
        if self._store is None:
            return 0
        if self.store_type == "chroma":
            return self._store._collection.count()
        return -1  # FAISS doesn't have a direct count method


def create_vectorstore(**kwargs) -> VectorStore:
    """Factory function to create a vector store."""
    return VectorStore(**kwargs)
