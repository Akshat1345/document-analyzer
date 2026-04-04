"""
Two-layer NER fusion engine.
Layer 1: spaCy en_core_web_sm - fast, deterministic baseline.
Layer 2: Llama 3.3 70B via Groq - context-aware, catches edge cases.
Results are merged with fuzzy deduplication.
"""

import spacy
import logging
from app.models.schemas import EntitiesResponse
from app.services.groq_client import get_entities_from_claude
from app.processors.entity_normalizer import (
    deduplicate_fuzzy,
    filter_false_positives
)

logger = logging.getLogger(__name__)


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

    def _extract_spacy(self, text: str) -> dict:
        """
        Extract named entities using spaCy.
        Covers PERSON, ORG, DATE, MONEY reliably.
        """
        result = {
            "names": [],
            "dates": [],
            "organizations": [],
            "amounts": []
        }
        if not self._spacy_model:
            return result
        try:
            doc = self._spacy_model(text[:5000])
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
            result = get_entities_from_claude(text)
            return {
                "names": result.names or [],
                "dates": result.dates or [],
                "organizations": result.organizations or [],
                "amounts": result.amounts or []
            }
        except Exception as e:
            logger.warning(
                "LLM NER failed, using spaCy results only: %s", e
            )
            return {
                "names": [],
                "dates": [],
                "organizations": [],
                "amounts": []
            }

    def _merge(self, spacy_r: dict, llm_r: dict) -> dict:
        """
        Merge results from both layers.
        Fuzzy deduplication handles formatting variants
        (e.g. 'Apple Inc' vs 'Apple Inc.').
        """
        merged = {}
        for field in ["names", "dates", "organizations", "amounts"]:
            combined = (
                spacy_r.get(field, []) +
                llm_r.get(field, [])
            )
            combined = filter_false_positives(combined, field)
            combined = deduplicate_fuzzy(combined)
            merged[field] = combined
        return merged

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
        merged       = self._merge(spacy_result, llm_result)
        return EntitiesResponse(**merged)
