"""Shared LLM factory for all agents.

Uses Ollama for local free inference, falls back to OpenRouter if available.
"""

import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_llm():
    """Return a ChatOllama client for local free inference."""
    use_ollama = os.getenv("USE_OLLAMA", "true").lower() == "true"
    
    if use_ollama:
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            temperature=0.3,
        )
    
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )