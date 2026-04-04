"""LLM client utilities for entity extraction and summarization."""

import json
import logging
import re
from typing import List

from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
OLLAMA_MODEL = "llama3.2:3b"


def _build_document_excerpt(text: str, max_chars: int = 6000) -> str:
    """Build a representative excerpt from a long document without biasing toward the opening section."""

    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped

    slice_size = max(1200, max_chars // 3)
    middle_start = max(0, len(stripped) // 2 - slice_size // 2)
    segments = [
        stripped[:slice_size].strip(),
        stripped[middle_start:middle_start + slice_size].strip(),
        stripped[-slice_size:].strip(),
    ]
    unique_segments = []
    for segment in segments:
        if segment and segment not in unique_segments:
            unique_segments.append(segment)
    return "\n\n".join(unique_segments)


def _build_client():
    """Build and return a chat client plus selected model name."""

    if settings.USE_LOCAL_LLM:
        from openai import OpenAI

        return OpenAI(base_url=f"{settings.LOCAL_LLM_URL}/v1", api_key="ollama"), OLLAMA_MODEL

    from groq import Groq

    return Groq(api_key=settings.GROQ_API_KEY), GROQ_MODEL


_client, _model = _build_client()


class ClaudeEntities(BaseModel):
    """Structured entity model retained for backward compatibility."""

    names: List[str] = Field(default_factory=list, description="Full names of individual people only. Empty list if none.")
    dates: List[str] = Field(default_factory=list, description="Dates exactly as they appear in text. Empty list if none.")
    organizations: List[str] = Field(default_factory=list, description="Company, institution, agency names. Empty list if none.")
    amounts: List[str] = Field(default_factory=list, description="Monetary values with currency symbols. Empty list if none.")
    emails: List[str] = Field(default_factory=list, description="Email addresses exactly as they appear in text. Empty list if none.")
    phones: List[str] = Field(default_factory=list, description="Phone numbers exactly as they appear in text. Empty list if none.")


def get_entities_from_claude(text: str) -> ClaudeEntities:
    """Extract entities from text with schema-constrained LLM output."""

    try:
        excerpt = _build_document_excerpt(text)
        response = _client.chat.completions.create(
            model=_model,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise document entity extractor. Extract ONLY entities "
                        "explicitly present in the text. Never infer, guess, or use the document type. "
                        "Ignore OCR-garbled tokens, partial words, section headers, and sentence fragments. "
                        "For organizations: include only real organization names explicitly visible in the document, "
                        "including companies, brands, institutions, universities, nonprofits, government bodies, "
                        "hospitals, labs, and associations. Do not include vague descriptions or role titles. "
                        "For dates: include only valid date-like expressions or years. "
                        "For amounts: include only currency values or percentages. "
                        "Return ONLY valid JSON with keys: names, dates, organizations, amounts. "
                        "Use [] when a key has no values."
                    ),
                },
                {"role": "user", "content": f"Extract named entities from this document:\n\n{excerpt}"},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            return ClaudeEntities()

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                return ClaudeEntities()
            payload = json.loads(match.group(0))

        return ClaudeEntities.model_validate(payload)
    except Exception as exc:
        logger.error("Entity extraction failed: %s", exc)
        return ClaudeEntities()


def get_summary_from_claude(prompt: str) -> str:
    """Generate concise factual summary from prompt using configured LLM."""

    try:
        resp = _client.chat.completions.create(
            model=_model,
            temperature=0,
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


def get_answer_from_context(question: str, context_chunks: list[str]) -> str:
    """Answer strictly from provided document context chunks."""

    context = "\n\n".join(context_chunks[:8])
    try:
        resp = _client.chat.completions.create(
            model=_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict document QA assistant. "
                        "Answer ONLY using the provided context from one document. "
                        "If the answer is not present, reply exactly: Not found in this document."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Document Context:\n{context}\n\n"
                        "Provide a concise factual answer."
                    ),
                },
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("Context QA failed: %s", exc)
        return ""


