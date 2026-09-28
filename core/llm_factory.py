"""
LLM Factory
============
统一的 LLM 创建工厂，支持 OpenAI / DeepSeek / Local(Ollama) 多后端。
"""

from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

# Ensure .env is loaded before settings
load_dotenv(override=True)

from config.settings import get_settings


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> BaseChatModel:
    """
    Create an LLM instance based on provider.

    Args:
        provider: "openai" | "deepseek" | "local". Defaults to config.
        model: Model name override.
        temperature: Temperature override.
        max_tokens: Max tokens override.
        **kwargs: Additional parameters passed to ChatOpenAI.

    Returns:
        A LangChain-compatible chat model.
    """
    settings = get_settings()
    provider = provider or settings.llm.provider
    temperature = temperature if temperature is not None else settings.llm.temperature
    max_tokens = max_tokens or settings.llm.max_tokens

    if provider == "openai":
        return ChatOpenAI(
            model=model or settings.llm.openai_model,
            api_key=settings.llm.openai_api_key,
            base_url=settings.llm.openai_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif provider == "deepseek":
        return ChatOpenAI(
            model=model or settings.llm.deepseek_model,
            api_key=settings.llm.deepseek_api_key,
            base_url=settings.llm.deepseek_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif provider == "local":
        return ChatOpenAI(
            model=model or settings.llm.local_model,
            api_key="not-needed",
            base_url=settings.llm.local_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'deepseek', or 'local'.")


@lru_cache()
def get_llm() -> BaseChatModel:
    """Get a cached default LLM instance."""
    return create_llm()
