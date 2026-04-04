"""
Two-signal sentiment ensemble.
Signal 1: Llama 3.3 70B via Groq - primary, full document context.
Signal 2: VADER - reliable fallback, pure Python, always available.
Document-type calibration prevents false positives on formal docs.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from app.services.groq_client import get_summary_from_claude
import logging
import re

logger = logging.getLogger(__name__)

VALID_SENTIMENTS = {"Positive", "Neutral", "Negative"}

# These document types default to Neutral unless
# there is strong signal pointing elsewhere
NEUTRAL_BIASED = {"invoice", "contract", "academic"}

NEUTRAL_HINTS = {
    "invoice",
    "contract",
    "agreement",
    "terms",
    "clause",
    "receipt",
    "technical",
    "documentation",
    "report",
    "abstract",
    "methodology",
    "findings",
}

POSITIVE_HINTS = {
    "excellent",
    "outstanding",
    "great",
    "success",
    "achievement",
    "happy",
}

NEGATIVE_HINTS = {
    "breach",
    "attack",
    "failure",
    "loss",
    "damage",
    "critical",
    "risk",
}

NEGATIVE_EVENT_TERMS = {
    "breach",
    "attack",
    "incident",
    "fraud",
    "outage",
    "failure",
    "lawsuit",
    "penalty",
    "violation",
    "compromised",
}

STRONG_SENTIMENT_TERMS = {
    "excellent",
    "outstanding",
    "crisis",
    "breach",
    "disaster",
    "severe",
    "catastrophic",
}


class SentimentEngine:
    """
    Groq LLM primary + VADER fallback sentiment ensemble.
    Returns exactly: 'Positive', 'Neutral', or 'Negative'.
    """

    _vader = None

    @classmethod
    def initialize(cls):
        """
        Initialize VADER at startup.
        Pure Python, instant, no download needed.
        """
        cls._vader = SentimentIntensityAnalyzer()
        logger.info("VADER sentiment analyzer initialized")

    def _vader_sentiment(self, text: str) -> str:
        """
        VADER rule-based sentiment classification.
        Used as fallback when Groq API is unavailable.
        """
        try:
            compound = self._vader.polarity_scores(
                text[:1000]
            )["compound"]
            if compound >= 0.05:
                return "Positive"
            elif compound <= -0.05:
                return "Negative"
            return "Neutral"
        except Exception as e:
            logger.warning("VADER error: %s", e)
            return "Neutral"

    def _llm_sentiment(
        self,
        text: str,
        doc_category: str
    ) -> str | None:
        """
        Primary sentiment classification using Llama 3.3 70B.
        Uses document-type-aware prompting for maximum accuracy.
        Returns None on failure — VADER fallback takes over.
        """
        try:
            # Use intro + conclusion for best sentiment signal
            # Intro sets the tone, conclusion gives final verdict
            if len(text) > 2000:
                analysis_text = text[:1500] + "\n...\n" + text[-500:]
            else:
                analysis_text = text

            prompt = (
                f"You are a calibrated, expert sentiment classifier.\n\n"
                f"━━━ YOUR TASK ━━━\n"
                f"Classify the OVERALL sentiment of the document below "
                f"as exactly one of: Positive, Negative, or Neutral.\n\n"
                f"━━━ DEFINITIONS ━━━\n\n"
                f"POSITIVE — The document's overall message is "
                f"optimistic, successful, beneficial, or constructive. "
                f"The reader comes away feeling informed about something "
                f"good, growing, innovative, or achieved.\n"
                f"Core signals: growth, success, innovation, "
                f"achievement, improvement, profit, opportunity, "
                f"advancement, record performance, breakthrough\n\n"
                f"NEGATIVE — The document's overall message is "
                f"alarming, harmful, failing, or concerning. "
                f"The reader comes away feeling worried, warned, "
                f"or informed about something bad that happened "
                f"or could happen.\n"
                f"Core signals: breach, attack, failure, crisis, "
                f"damage, threat, violation, loss, fraud, declining, "
                f"vulnerability, incident, unauthorized, compromised\n\n"
                f"NEUTRAL — The document is factual, administrative, "
                f"or informational with no clear emotional direction. "
                f"It reports facts, obligations, or procedures without "
                f"editorial tone or emotional language.\n"
                f"Core signals: invoices, contracts, academic research, "
                f"technical specifications, balanced reporting, "
                f"procedural documents\n\n"
                f"━━━ DECISION RULES BY DOCUMENT TYPE ━━━\n\n"
                f"invoice → NEUTRAL (it is a billing document)\n"
                f"contract → NEUTRAL (it is a legal agreement)\n"
                f"academic → NEUTRAL (research is factual by default)\n"
                f"resume/CV → POSITIVE (achievements and career focus)\n"
                f"cybersecurity incident → NEGATIVE (breach = bad)\n"
                f"data breach report → NEGATIVE always\n"
                f"technology innovation article → POSITIVE\n"
                f"AI advancement / research expansion → POSITIVE\n"
                f"financial growth / record earnings → POSITIVE\n"
                f"financial loss / fraud / scandal → NEGATIVE\n"
                f"news article → judge by whether described "
                f"events are good or bad for people involved\n\n"
                f"━━━ CRITICAL RULE ━━━\n"
                f"Base sentiment on the OVERALL document tone, not one "
                f"sentence. A cybersecurity report recommending "
                f"improvements is still NEGATIVE because it describes "
                f"a breach. A document about AI risks is NEGATIVE. "
                f"A document about AI benefits is POSITIVE.\n\n"
                f"━━━ DOCUMENT ━━━\n"
                f"Type: {doc_category}\n\n"
                f"{analysis_text}\n\n"
                f"━━━ YOUR RESPONSE ━━━\n"
                f"Reply with EXACTLY ONE WORD.\n"
                f"No punctuation. No explanation. No quotes. "
                f"No preamble.\n"
                f"The only valid responses are:\n"
                f"Positive\n"
                f"Negative\n"
                f"Neutral\n\n"
                f"Classification:"
            )

            raw = get_summary_from_claude(prompt)
            if not raw:
                return None

            # Clean and validate response
            cleaned = raw.strip().rstrip(".,!\"'").strip()

            # Direct match first
            if cleaned in VALID_SENTIMENTS:
                return cleaned

            # Capitalize first letter and check again
            capitalized = cleaned.capitalize()
            if capitalized in VALID_SENTIMENTS:
                return capitalized

            # Substring match as last resort
            lower = cleaned.lower()
            if "positive" in lower:
                return "Positive"
            if "negative" in lower:
                return "Negative"
            if "neutral" in lower:
                return "Neutral"

            logger.warning(
                "LLM returned unexpected sentiment value: '%s'", raw
            )
            return None

        except Exception as e:
            logger.warning("LLM sentiment call failed: %s", e)
            return None

    def _looks_formal_or_factual(self, text: str, doc_category: str) -> bool:
        """Return True when lexical signals suggest neutral formal writing."""

        if doc_category in NEUTRAL_BIASED:
            return True
        lowered = text[:2500].lower()
        neutral_hits = sum(1 for token in NEUTRAL_HINTS if token in lowered)
        emotional_hits = sum(1 for token in POSITIVE_HINTS | NEGATIVE_HINTS if token in lowered)
        punctuation_emphasis = len(re.findall(r"[!]{2,}", lowered))
        return neutral_hits >= 2 and emotional_hits == 0 and punctuation_emphasis == 0

    def _has_strong_sentiment_language(self, text: str) -> bool:
        """Detect strong clearly-opinionated language in text snippet."""

        lowered = text[:2500].lower()
        return any(term in lowered for term in STRONG_SENTIMENT_TERMS)

    def _has_negative_event_language(self, text: str) -> bool:
        """Detect explicit negative event terms that should override optimistic defaults."""

        lowered = text[:2500].lower()
        return any(term in lowered for term in NEGATIVE_EVENT_TERMS)

    def _has_positive_tech_language(self, text: str) -> bool:
        """Detect positive technology/growth indicators for tech documents."""

        positive_tech_terms = {
            "growth", "innovation", "breakthrough", "advancement",
            "record", "success", "expand", "lead", "leader", "leading",
            "achieve", "achievement", "improve", "improvement", "investment",
            "research", "development", "ai", "artificial", "technology",
        }
        lowered = text[:2500].lower()
        return sum(1 for term in positive_tech_terms if term in lowered) >= 3

    def analyze(self, text: str, doc_category: str) -> str:
        """
        Main sentiment analysis method.
        Runs both signals and resolves disagreements intelligently.
        Always returns exactly: 'Positive', 'Neutral', or 'Negative'.
        """
        try:
            # VADER first - always works, no API dependency
            vader_result = self._vader_sentiment(text)

            # Technology/innovation documents with strong positive signals should be Positive
            if self._has_positive_tech_language(text):
                llm_result = self._llm_sentiment(text, doc_category)
                if llm_result == "Positive":
                    return "Positive"

            # Formal categories are factual by default, unless they have positive tech signals
            if doc_category in NEUTRAL_BIASED and not self._has_strong_sentiment_language(text) and not self._has_positive_tech_language(text):
                return "Neutral"

            # Incident and breach-style reports should remain negative by default.
            if doc_category == "incident_report" or self._has_negative_event_language(text):
                return "Negative"

            # Resumes are achievement-focused unless they explicitly describe negative events.
            if doc_category == "resume":
                return "Positive"

            # LLM primary - more accurate with full context
            llm_result = self._llm_sentiment(text, doc_category)

            # LLM unavailable - fall back to VADER
            if not llm_result:
                logger.info(
                    "LLM unavailable, using VADER: %s", vader_result
                )
                return vader_result

            # Both agree - high confidence
            if llm_result == vader_result:
                return llm_result

            # Disagreement on formal document types -> prefer Neutral
            if doc_category in NEUTRAL_BIASED:
                if "Neutral" in [llm_result, vader_result]:
                    return "Neutral"

            # Additional neutral guardrail for factual/formal text.
            if self._looks_formal_or_factual(text, doc_category):
                if "Neutral" in [llm_result, vader_result]:
                    return "Neutral"

            # General documents are often factual; avoid optimistic drift when no strong sentiment exists.
            if doc_category == "general":
                if vader_result == "Neutral" and not self._has_strong_sentiment_language(text):
                    return "Neutral"

            # Default: trust LLM over VADER when they disagree
            # LLM has full document context, VADER only first 1000 chars
            return llm_result

        except Exception as e:
            logger.error("Sentiment analysis crashed: %s", e)
            return "Neutral"
