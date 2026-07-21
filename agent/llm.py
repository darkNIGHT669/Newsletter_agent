"""
LLM provider factory.

Reads LLM_PROVIDER from the environment ("anthropic" | "openai" | "gemini")
and returns a configured LangChain chat model. Keeping this in one place
means every node gets its LLM the same way and the whole project can be
re-pointed at a different provider by changing one .env value.
"""

import os


def get_llm(temperature: float = 0.3):
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022"),
            temperature=temperature,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            temperature=temperature,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL", "gemini-1.5-flash"),
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use 'anthropic', 'openai', or 'gemini'."
    )
