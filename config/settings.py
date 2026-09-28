"""
Agent Configuration
===================
Centralized configuration management using pydantic-settings.
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "deepseek"))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    openai_base_url: str = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    deepseek_api_key: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_model: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    deepseek_base_url: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    local_model: str = Field(default_factory=lambda: os.getenv("LOCAL_MODEL", "llama3"))
    local_base_url: str = Field(default_factory=lambda: os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))
    max_tokens: int = 4096


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_name: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    use_local: bool = True


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    store_type: str = Field(default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "faiss"))
    persist_dir: str = str(PROJECT_ROOT / "data" / "vector_store")
    collection_name: str = "knowledge_base"


class AgentConfig(BaseModel):
    """Agent runtime configuration."""
    max_iterations: int = Field(default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "10")))
    verbose: bool = Field(default_factory=lambda: os.getenv("VERBOSE", "true").lower() == "true")
    memory_window_size: int = 20
    enable_cot: bool = True
    enable_rag: bool = True
    enable_tools: bool = True
    auto_memory: bool = Field(default_factory=lambda: os.getenv("AUTO_MEMORY", "true").lower() == "true")


class Settings(BaseModel):
    """Root settings container."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    knowledge_base_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
