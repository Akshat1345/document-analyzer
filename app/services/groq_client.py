"""LLM client utilities for entity extraction and summarization."""

import logging
from typing import Any, List

import instructor
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
OLLAMA_MODEL = "llama3.2:3b"


def _build_client():
    """Build and return an instructor client plus selected model name."""

    if settings.USE_LOCAL_LLM:
        from openai import OpenAI

        raw = OpenAI(base_url=f"{settings.LOCAL_LLM_URL}/v1", api_key="ollama")
        return instructor.from_openai(raw, mode=instructor.Mode.JSON), OLLAMA_MODEL

    from groq import Groq

    raw = Groq(api_key=settings.GROQ_API_KEY)
    return instructor.from_groq(raw, mode=instructor.Mode.JSON), GROQ_MODEL


_client, _model = _build_client()


class ClaudeEntities(BaseModel):
    """Structured entity model retained for backward compatibility."""

    names: List[str] = Field(default_factory=list, description="Full names of individual people only. Empty list if none.")
    dates: List[str] = Field(default_factory=list, description="Dates exactly as they appear in text. Empty list if none.")
    organizations: List[str] = Field(default_factory=list, description="Company, institution, agency names. Empty list if none.")
    amounts: List[str] = Field(default_factory=list, description="Monetary values with currency symbols. Empty list if none.")


def get_entities_from_claude(text: str) -> ClaudeEntities:
    """Extract entities from text with schema-constrained LLM output."""

    try:
        return _client.chat.completions.create(
            model=_model,
            response_model=ClaudeEntities,
            max_retries=3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise NER system. Extract ONLY entities "
                        "explicitly present in the text. Never infer or hallucinate. "
                        "Return empty lists when absent."
                    ),
                },
                {"role": "user", "content": f"Extract named entities:\n\n{text[:3000]}"},
            ],
        )
    except Exception as exc:
        logger.error("Entity extraction failed: %s", exc)
        return ClaudeEntities()


def get_summary_from_claude(prompt: str) -> str:
    """Generate concise factual summary from prompt using configured LLM."""

    try:
        response_model: Any = None
        resp = _client.chat.completions.create(
            model=_model,
            response_model=response_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise document summarizer. "
                        "Write exactly 2-4 dense factual sentences. "
                        "Include key names, numbers, outcomes. "
                        "Never start with 'This document'. "
                        "Return ONLY the summary, nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("Summarization failed: %s", exc)
        return ""


