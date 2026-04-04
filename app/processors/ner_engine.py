"""
Two-layer NER fusion engine.
Layer 1: spaCy en_core_web_sm - fast, deterministic baseline.
Layer 2: Llama 3.3 70B via Groq - context-aware fallback.
Results are merged with fuzzy deduplication.
"""

import spacy
import logging
import re
from app.models.schemas import EntitiesResponse
from app.services.groq_client import get_entities_from_claude
from app.processors.entity_normalizer import (
    deduplicate_fuzzy,
    filter_false_positives,
    normalize_amount,
    normalize_date,
    normalize_name,
    normalize_organization,
)

logger = logging.getLogger(__name__)

MONTH_PATTERN = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_PATTERNS = [
    re.compile(rf"\b{MONTH_PATTERN}\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b{MONTH_PATTERN}\s+\d{{1,2}},?\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(r"\b(?:19|20)\d{2}\b"),
]
AMOUNT_PATTERNS = [
    re.compile(r"(?:[$₹€£]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?)"),
    re.compile(r"\b\d+(?:\.\d+)?%\b"),
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"),
]
EMAIL_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
]
PHONE_PATTERNS = [
    re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
]
ORG_CONTEXT_PATTERNS = [
    re.compile(r"(?i:\b(?:at|with|from|in|for|worked\s+at|worked\s+with|joined))\s+([A-Z][A-Za-z&.'-]+(?:\s+[A-Z][A-Za-z&.'-]+){0,5})"),
    re.compile(r"\b((?:[A-Z][A-Za-z&.'-]+\s+){0,5}(?:Agency|Media|University|Institute|Company|Corp|Inc|Ltd|LLC|Group|Studio|Consulting|College|Center|Labs|Solutions|Systems|Services))\b"),
    re.compile(r"\b([A-Z][A-Za-z&.'-]+\s+School\s+of\s+[A-Z][A-Za-z&.'-]+(?:\s+[A-Z][A-Za-z&.'-]+)?)\b"),
    re.compile(r"\b([A-Z][A-Za-z&.'-]+(?:\s+[A-Z][A-Za-z&.'-]+){0,4}\s+(?:of\s+)?(?:Design|Technology|Arts|Science|Business|Engineering))\b"),
]
ORG_LIST_PATTERNS = [
    re.compile(r"(?i:(?:companies\s+such\s+as|such\s+as|including|like))\s+([A-Z0-9][A-Za-z0-9&.'-]*(?:\s*,\s*[A-Z0-9][A-Za-z0-9&.'-]*)*(?:\s*,?\s*(?:and|or)\s+[A-Z0-9][A-Za-z0-9&.'-]*)?)"),
    re.compile(r"([A-Z][A-Za-z0-9&.'-]+(?:\s*,\s*[A-Z][A-Za-z0-9&.'-]+)+(?:\s*,?\s*(?:and|or)\s+[A-Z][A-Za-z0-9&.'-]+)?)"),
]
WINDOW_MAX_CHARS = 3500
WINDOW_OVERLAP = 1


class NEREngine:
    """
    Two-layer NER: spaCy sm for baseline + Groq Llama 3.3 70B
    for context-aware entity extraction. Results fused and deduped.
    """

    _spacy_model = None

    @classmethod
    def initialize(cls):
        """
        Load spaCy small model once at startup.
        ~50MB, no PyTorch, fast cold start.
        """
        try:
            cls._spacy_model = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm loaded successfully")
        except OSError:
            logger.warning(
                "spaCy model not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            cls._spacy_model = None

    def _iter_text_windows(self, text: str, max_chars: int = WINDOW_MAX_CHARS) -> list[str]:
        """Split long text into paragraph windows so extraction covers the full document."""

        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
        if not paragraphs:
            return [text]

        windows: list[str] = []
        current: list[str] = []
        current_chars = 0
        for paragraph in paragraphs:
            paragraph_chars = len(paragraph)
            if current and current_chars + paragraph_chars > max_chars:
                windows.append("\n\n".join(current))
                current = current[-WINDOW_OVERLAP:] if WINDOW_OVERLAP else []
                current_chars = sum(len(part) for part in current)
            current.append(paragraph)
            current_chars += paragraph_chars

        if current:
            windows.append("\n\n".join(current))
        return windows if windows else [text]

    def _build_llm_excerpt(self, text: str, max_chars: int = 6000) -> str:
        """Select representative text slices so the fallback LLM sees the whole document shape."""

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

    def _extract_spacy(self, text: str) -> dict:
        """
        Extract named entities using spaCy.
        Covers PERSON, ORG, DATE, MONEY reliably.
        """
        result = {
            "names": [],
            "dates": [],
            "organizations": [],
            "amounts": [],
            "emails": [],
            "phones": [],
        }
        if not self._spacy_model:
            return result
        try:
            for window in self._iter_text_windows(text):
                doc = self._spacy_model(window)
                for ent in doc.ents:
                    val = ent.text.strip()
                    if not val:
                        continue
                    if ent.label_ == "PERSON":
                        result["names"].append(val)
                    elif ent.label_ == "ORG":
                        result["organizations"].append(val)
                    elif ent.label_ == "DATE":
                        result["dates"].append(val)
                    elif ent.label_ == "MONEY":
                        result["amounts"].append(val)
        except Exception as e:
            logger.warning("spaCy NER error: %s", e)
        return result

    def _extract_llm(self, text: str) -> dict:
        """
        Extract entities using Llama 3.3 70B via Groq API.
        Catches domain-specific and context-dependent entities
        that rule-based spaCy misses.
        """
        try:
            result = get_entities_from_claude(self._build_llm_excerpt(text))
            return {
                "names": result.names or [],
                "dates": result.dates or [],
                "organizations": result.organizations or [],
                "amounts": result.amounts or [],
                "emails": result.emails or [],
                "phones": result.phones or [],
            }
        except Exception as e:
            logger.warning(
                "LLM NER failed, using spaCy results only: %s", e
            )
            return {
                "names": [],
                "dates": [],
                "organizations": [],
                "amounts": [],
                "emails": [],
                "phones": [],
            }

    def _merge(self, spacy_r: dict, llm_r: dict) -> dict:
        """
        Merge results from both layers.
        Fuzzy deduplication handles formatting variants
        (e.g. 'Apple Inc' vs 'Apple Inc.').
        """
        merged = {}
        for field in ["names", "dates", "organizations", "amounts", "emails", "phones"]:
            combined = (
                spacy_r.get(field, []) +
                llm_r.get(field, [])
            )
            if field == "names":
                combined = [normalize_name(v) for v in combined]
            elif field == "dates":
                combined = [normalize_date(v) for v in combined]
            elif field == "organizations":
                combined = [normalize_organization(v) for v in combined]
            elif field == "amounts":
                combined = [normalize_amount(v) for v in combined]
            elif field in {"emails", "phones"}:
                combined = [v.strip() for v in combined]

            combined = [v for v in combined if v]
            combined = filter_false_positives(combined, field)
            combined = deduplicate_fuzzy(combined)
            merged[field] = combined
        return merged

    def _extract_regex(self, text: str) -> dict:
        """Extract high-confidence dates and amounts via regex fallback."""

        out = {"names": [], "dates": [], "organizations": [], "amounts": [], "emails": [], "phones": []}
        normalized_lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines() if line.strip()]
        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(text):
                out["dates"].append(match.group(0).strip())
        for pattern in AMOUNT_PATTERNS:
            for match in pattern.finditer(text):
                out["amounts"].append(match.group(0).strip())
        for pattern in EMAIL_PATTERNS:
            for match in pattern.finditer(text):
                out["emails"].append(match.group(0).strip())
        for pattern in PHONE_PATTERNS:
            for match in pattern.finditer(text):
                out["phones"].append(match.group(0).strip())
        for pattern in ORG_CONTEXT_PATTERNS:
            for line in normalized_lines:
                for match in pattern.finditer(line):
                    value = match.group(1).strip()
                    if value:
                        out["organizations"].append(value)
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if value:
                    out["organizations"].append(value)

        for pattern in ORG_LIST_PATTERNS:
            for line in normalized_lines:
                for match in pattern.finditer(line):
                    raw = match.group(1).strip()
                    parts = re.split(r"\s*,\s*|\s+and\s+|\s+or\s+", raw, flags=re.IGNORECASE)
                    for part in parts:
                        candidate = part.strip().strip(" ,;:-")
                        if candidate:
                            out["organizations"].append(candidate)
        return out

    def extract_all(
        self,
        text: str,
        doc_category: str
    ) -> EntitiesResponse:
        """
        Main entry point. Run both layers and return merged result.
        Always returns valid EntitiesResponse even on partial failure.
        """
        spacy_result = self._extract_spacy(text)
        llm_result   = self._extract_llm(text)
        regex_result = self._extract_regex(text)
        merged       = self._merge(
            {
                "names": spacy_result.get("names", []),
                "dates": spacy_result.get("dates", []) + regex_result.get("dates", []),
                "organizations": spacy_result.get("organizations", []) + regex_result.get("organizations", []),
                "amounts": spacy_result.get("amounts", []) + regex_result.get("amounts", []),
                "emails": spacy_result.get("emails", []) + regex_result.get("emails", []),
                "phones": spacy_result.get("phones", []) + regex_result.get("phones", []),
            },
            {
                "names": llm_result.get("names", []),
                "dates": llm_result.get("dates", []),
                "organizations": llm_result.get("organizations", []),
                "amounts": llm_result.get("amounts", []),
                "emails": llm_result.get("emails", []),
                "phones": llm_result.get("phones", []),
            },
        )
        return EntitiesResponse(**merged)
