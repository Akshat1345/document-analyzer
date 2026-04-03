"""Triple-layer NER engine with source-aware fusion and filtering."""

from __future__ import annotations

import re
import logging
import threading
from collections import defaultdict

import spacy
from gliner import GLiNER

from app.models.schemas import EntitiesResponse
from app.processors.entity_normalizer import (
    deduplicate_exact_casefold,
    deduplicate_fuzzy,
    filter_false_positives,
    normalize_amount,
    normalize_date,
    normalize_name,
)
from app.services.groq_client import get_entities_from_claude

logger = logging.getLogger(__name__)

MONTH_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}|\(?\d{3}\)?[\s.-]?\d{4}[\s.-]?\d{4})(?!\d)"
)
EMAIL_PATTERN = re.compile(
    r"\b([A-Za-z0-9._%+-]+)\s*@\s*([A-Za-z0-9.-]+)\s*\.\s*([A-Za-z]{2,})\b"
)
DATE_PATTERNS = [
    re.compile(rf"\b{MONTH_PATTERN}\s+\d{{1,2}}(?:,?\s+\d{{4}})?\b"),
    re.compile(rf"\b{MONTH_PATTERN}\s+\d{{4}}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"(?:\byear\s+)?\b(?:19|20)\d{2}\b"),
]
AMOUNT_PATTERNS = [
    re.compile(r"(?:[$€£]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|thousand|k))?)", re.IGNORECASE),
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?%?\b"),
    re.compile(r"\b\d+(?:\.\d+)?%\b"),
]
ORG_SUFFIXES = {
    "agency",
    "agencies",
    "company",
    "companies",
    "corp",
    "corporation",
    "group",
    "inc",
    "institute",
    "institution",
    "llc",
    "ltd",
    "media",
    "school",
    "university",
    "universities",
    "bank",
    "banks",
    "services",
    "systems",
    "labs",
}


def _normalize_email_candidate(raw: str) -> str:
    """Normalize OCR-split email strings by removing intra-token spaces."""

    value = re.sub(r"\s+", "", raw.strip())
    return value.lower()


def _normalize_phone_candidate(raw: str) -> str:
    """Normalize phone display while preserving optional leading plus."""

    cleaned = raw.strip()
    has_plus = cleaned.startswith("+")
    digits = re.sub(r"\D", "", cleaned)
    if not digits:
        return ""
    return f"+{digits}" if has_plus else digits


class NEREngine:
    """Entity extraction orchestrator across spaCy, GLiNER, and LLM layers."""

    _spacy_model = None
    _gliner_model = None
    _lock = threading.Lock()

    @classmethod
    def initialize(cls) -> None:
        """Load NER models once at startup."""

        cls._ensure_initialized()

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Load NER models lazily and only once."""

        if cls._spacy_model is not None and cls._gliner_model is not None:
            return

        with cls._lock:
            if cls._spacy_model is None:
                try:
                    cls._spacy_model = spacy.load("en_core_web_trf")
                except Exception as exc:
                    logger.warning("Failed to load spaCy transformer model: %s", exc)
                    try:
                        cls._spacy_model = spacy.load("en_core_web_sm")
                    except Exception as fallback_exc:
                        logger.warning("Failed to load spaCy small model: %s", fallback_exc)
                        cls._spacy_model = None

            if cls._gliner_model is None:
                try:
                    cls._gliner_model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
                except Exception as exc:
                    logger.warning("Failed to load GLiNER model: %s", exc)
                    cls._gliner_model = None

    def _extract_spacy(self, text: str) -> dict[str, list[str]]:
        """Extract entities from spaCy layer and map into API schema buckets."""

        out = {"names": [], "dates": [], "organizations": [], "amounts": [], "emails": [], "phones": []}
        try:
            self._ensure_initialized()
            if self._spacy_model is None:
                return out
            doc = self._spacy_model(text[:5000])
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    out["names"].append(ent.text)
                elif ent.label_ == "ORG":
                    out["organizations"].append(ent.text)
                elif ent.label_ == "DATE":
                    out["dates"].append(ent.text)
                elif ent.label_ == "MONEY":
                    out["amounts"].append(ent.text)
        except Exception as exc:
            logger.warning("spaCy extraction failed: %s", exc)
        return out

    def _extract_gliner(self, text: str) -> dict[str, list[str]]:
        """Extract entities from GLiNER with required label mapping."""

        out = {"names": [], "dates": [], "organizations": [], "amounts": [], "emails": [], "phones": []}
        labels = [
            "person name",
            "organization name",
            "date",
            "monetary amount",
            "company",
            "institution",
            "government agency",
        ]
        try:
            self._ensure_initialized()
            if self._gliner_model is None:
                return out
            entities = self._gliner_model.predict_entities(text[:5000], labels, threshold=0.5)
            for item in entities:
                label = item.get("label", "")
                value = item.get("text", "")
                if label == "person name":
                    out["names"].append(value)
                elif label in {"organization name", "company", "institution", "government agency"}:
                    out["organizations"].append(value)
                elif label == "date":
                    out["dates"].append(value)
                elif label == "monetary amount":
                    out["amounts"].append(value)
        except Exception as exc:
            logger.warning("GLiNER extraction failed: %s", exc)
        return out

    def _extract_llm(self, text: str) -> dict[str, list[str]]:
        """Extract entities using LLM response model."""

        out = {"names": [], "dates": [], "organizations": [], "amounts": [], "emails": [], "phones": []}
        try:
            llm = get_entities_from_claude(text)
            out = {
                "names": llm.names,
                "dates": llm.dates,
                "organizations": llm.organizations,
                "amounts": llm.amounts,
            }
        except Exception as exc:
            logger.warning("LLM extraction failed: %s", exc)
        return out

    def _extract_regex(self, text: str) -> dict[str, list[str]]:
        """Extract obvious date, amount, email, and phone patterns as a fallback."""

        out = {"names": [], "dates": [], "organizations": [], "amounts": [], "emails": [], "phones": []}
        
        # Remove phone numbers from consideration to avoid false date matches
        text_cleaned = PHONE_PATTERN.sub("", text)

        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(text_cleaned):
                candidate = match.group(0).strip()
                if candidate not in out["dates"]:
                    out["dates"].append(candidate)

        for pattern in AMOUNT_PATTERNS:
            for match in pattern.finditer(text):
                candidate = match.group(0).strip()
                if candidate not in out["amounts"]:
                    out["amounts"].append(candidate)

        # Extract emails
        for match in EMAIL_PATTERN.finditer(text):
            candidate = _normalize_email_candidate(
                f"{match.group(1)}@{match.group(2)}.{match.group(3)}"
            )
            if candidate not in out["emails"]:
                out["emails"].append(candidate)

        # Extract phones
        for match in PHONE_PATTERN.finditer(text):
            candidate = _normalize_phone_candidate(match.group(0))
            if candidate not in out["phones"]:
                out["phones"].append(candidate)

        return out

    def _merge(
        self,
        spacy_r: dict[str, list[str]],
        gliner_r: dict[str, list[str]],
        llm_r: dict[str, list[str]],
        regex_r: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Merge outputs with multi-source priority and type-specific cleaning."""

        merged = {"names": [], "dates": [], "organizations": [], "amounts": [], "emails": [], "phones": []}

        for key in merged.keys():
            evidence: dict[str, set[str]] = defaultdict(set)
            for source, values in [
                ("spacy", spacy_r.get(key, [])),
                ("gliner", gliner_r.get(key, [])),
                ("llm", llm_r.get(key, [])),
                ("regex", regex_r.get(key, [])),
            ]:
                for value in values:
                    candidate = value.strip()
                    if not candidate:
                        continue

                    if key in {"emails", "phones"}:
                        evidence[candidate].add(source)
                        continue

                    matched = None
                    for existing in evidence:
                        from fuzzywuzzy import fuzz

                        if fuzz.ratio(existing.lower(), candidate.lower()) > 85:
                            matched = existing
                            break
                    if matched is None:
                        evidence[candidate].add(source)
                    else:
                        evidence[matched].add(source)

            primary = [v for v, src in evidence.items() if len(src) >= 2]
            secondary = [v for v, src in evidence.items() if len(src) < 2]
            ordered = deduplicate_fuzzy(primary + secondary)

            if key == "names":
                ordered = [normalize_name(v) for v in ordered]
                ordered = [v for v in ordered if v]
            elif key == "dates":
                ordered = [normalize_date(v) for v in ordered]
            elif key == "amounts":
                ordered = [normalize_amount(v) for v in ordered]
            elif key in {"emails", "phones"}:
                ordered = deduplicate_exact_casefold(ordered)

            merged[key] = filter_false_positives(ordered, key)

        return merged

    def extract_all(self, text: str, doc_category: str) -> EntitiesResponse:
        """Run all extraction layers and return robust merged entities."""

        del doc_category
        try:
            spacy_result = self._extract_spacy(text)
        except Exception:
            spacy_result = {"names": [], "dates": [], "organizations": [], "amounts": [], "emails": [], "phones": []}

        try:
            gliner_result = self._extract_gliner(text)
        except Exception:
            gliner_result = {"names": [], "dates": [], "organizations": [], "amounts": [], "emails": [], "phones": []}

        try:
            llm_result = self._extract_llm(text)
        except Exception:
            llm_result = {"names": [], "dates": [], "organizations": [], "amounts": [], "emails": [], "phones": []}

        try:
            regex_result = self._extract_regex(text)
        except Exception:
            regex_result = {"names": [], "dates": [], "organizations": [], "amounts": [], "emails": [], "phones": []}

        return EntitiesResponse(**self._merge(spacy_result, gliner_result, llm_result, regex_result))
