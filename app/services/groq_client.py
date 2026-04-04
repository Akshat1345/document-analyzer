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
    """
    Extract named entities using Llama 3.3 70B via Groq.
    Uses a highly engineered prompt for maximum precision and recall.
    Auto-retries are handled by the Instructor library internally.
    """
    try:
        response = _client.chat.completions.create(
            model=_model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Named Entity Recognition "
                        "engine trained on millions of business, legal, "
                        "financial, medical, and technical documents.\n\n"
                        "YOUR MISSION: Extract named entities with "
                        "PERFECT RECALL (miss nothing) and PERFECT "
                        "PRECISION (add nothing that is not there).\n\n"
                        "━━━ ENTITY DEFINITIONS ━━━\n\n"
                        "NAMES — Full names of individual human beings "
                        "only.\n"
                        "  ✅ Include: 'Ravi Kumar', 'Nina Lane', "
                        "'Tim Cook', 'Dr. Sarah Johnson'\n"
                        "  ❌ Exclude: job titles alone ('CEO', "
                        "'Manager'), pronouns ('he', 'she'), "
                        "generic references ('the author', "
                        "'the victim'), organisation names\n\n"
                        "ORGANIZATIONS — Named companies, institutions, "
                        "agencies, universities, banks, government "
                        "bodies, NGOs, brands.\n"
                        "  ✅ Include: 'Apple Inc.', 'Reserve Bank of "
                        "India', 'Parsons School of Design', "
                        "'Brightline Agency', 'Google', 'Microsoft', "
                        "'NVIDIA'\n"
                        "  ❌ Exclude: generic references ('the company',"
                        " 'the bank', 'the institution'), "
                        "unnamed entities\n\n"
                        "DATES — All specific date references exactly "
                        "as written in the source text.\n"
                        "  ✅ Include: '10 March 2026', 'June 2020', "
                        "'Q3 2024', 'March 2017', '2017', "
                        "'May 2020 - Present'\n"
                        "  ❌ Exclude: vague references with no "
                        "specific time ('recently', 'last year', "
                        "'in the past') unless a year is stated\n\n"
                        "AMOUNTS — All monetary values with their "
                        "currency symbol or code, exactly as written.\n"
                        "  ✅ Include: '₹10,000', '$94.8 billion', "
                        "'€500', 'USD 1.2M', '30%' (if financial "
                        "percentage), '£2,500'\n"
                        "  ❌ Exclude: plain numbers without currency "
                        "context, percentages that are not monetary\n\n"
                        "━━━ EXTRACTION RULES ━━━\n\n"
                        "1. Read the ENTIRE text before extracting. "
                        "Do not stop at the first mention.\n"
                        "2. Extract entities from every part: "
                        "headings, body, tables, footers, captions.\n"
                        "3. Preserve original text exactly — "
                        "do not correct spelling or reformat.\n"
                        "4. Each unique entity appears ONCE per list "
                        "— no duplicates.\n"
                        "5. If a name appears multiple times "
                        "(e.g. 'Tim Cook' and 'Cook') keep the "
                        "most complete version only.\n"
                        "6. Return empty list [] for any category "
                        "where zero entities of that type exist.\n"
                        "7. NEVER hallucinate. If unsure whether "
                        "something is an entity: leave it out."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Extract all named entities from the text "
                        f"below. Be exhaustive — miss nothing.\n\n"
                        f"━━━ TEXT BEGIN ━━━\n"
                        f"{text[:4000]}\n"
                        f"━━━ TEXT END ━━━\n\n"
                        f"Return a JSON object with exactly these "
                        f"4 keys:\n"
                        f"  names         → full person names only\n"
                        f"  dates         → dates as written\n"
                        f"  organizations → org/company/institution "
                        f"names\n"
                        f"  amounts       → monetary values with "
                        f"currency\n\n"
                        f"Extract every entity present. "
                        f"Return empty lists where none exist."
                    )
                }
            ]
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
    except Exception as e:
        logger.error("Entity extraction failed: %s", e)
        return ClaudeEntities()


def get_summary_from_claude(prompt: str) -> str:
    """
    Generate text using Llama 3.3 70B via Groq.
    Used for both summarization and sentiment classification.
    """
    try:
        response = _client.chat.completions.create(
            model=_model,
            response_format=None,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a world-class document analyst, "
                        "technical writer, and sentiment classifier. "
                        "You produce factually precise outputs with "
                        "zero hallucination. You follow every "
                        "instruction exactly. You never add preamble, "
                        "never explain your reasoning, never use "
                        "filler language, and never deviate from "
                        "the requested output format. When asked for "
                        "one word, you return one word. When asked "
                        "for a summary, you return only the summary."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        result = response.choices[0].message.content
        if result:
            return result.strip()
        return ""
    except Exception as e:
        logger.error("Groq API call failed: %s", e)
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


