"""
Two-signal sentiment ensemble.
Signal 1: Llama 3.3 70B via Groq - primary, full document context.
Signal 2: VADER - reliable fallback, pure Python, always available.
Document-type calibration prevents false positives on formal docs.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from app.services.groq_client import get_summary_from_claude
import logging

logger = logging.getLogger(__name__)

VALID_SENTIMENTS = {"Positive", "Neutral", "Negative"}

# These document types default to Neutral unless
# there is strong signal pointing elsewhere
NEUTRAL_BIASED = {"invoice", "contract", "academic"}


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
        Ask Llama 3.3 70B to classify sentiment.
        Has full document context for accurate classification.
        Returns None on failure so fallback can take over.
        """
        try:
            prompt = f"""You are a precise sentiment classifier.

Classify the OVERALL sentiment of this {doc_category} document.

CLASSIFICATION RULES:
- Positive: growth, success, innovation, achievement,
            improvement, profit, opportunity, benefit,
            advancement, optimism
- Negative: breach, attack, failure, crisis, damage,
            threat, violation, loss, concern, declining,
            warning, risk, problem
- Neutral:  factual reporting, invoices, contracts,
            academic research, technical documentation,
            balanced news with no clear emotional tone

DOCUMENT TYPE DEFAULTS:
- invoice / contract / academic paper -> Neutral by default
  unless strongly emotional language is present
- cybersecurity incident / breach / attack report -> Negative
- technology innovation / business growth / success -> Positive
- news article -> depends entirely on the content

Document type: {doc_category}
Document text:
{text[:2000]}

Reply with EXACTLY ONE WORD.
No punctuation. No explanation. No quotes.
Valid options: Positive  Negative  Neutral

Your classification:"""

            raw = get_summary_from_claude(prompt)
            if not raw:
                return None

            # Clean response - LLM sometimes adds punctuation
            cleaned = raw.strip().rstrip(".,!\"'").capitalize()

            if cleaned in VALID_SENTIMENTS:
                return cleaned

            # Handle variations in case LLM adds extra words
            lower = cleaned.lower()
            if "positive" in lower:
                return "Positive"
            if "negative" in lower:
                return "Negative"
            if "neutral" in lower:
                return "Neutral"

            logger.warning(
                "LLM returned unexpected sentiment: '%s'", raw
            )
            return None

        except Exception as e:
            logger.warning("LLM sentiment error: %s", e)
            return None

    def analyze(self, text: str, doc_category: str) -> str:
        """
        Main sentiment analysis method.
        Runs both signals and resolves disagreements intelligently.
        Always returns exactly: 'Positive', 'Neutral', or 'Negative'.
        """
        try:
            # VADER first - always works, no API dependency
            vader_result = self._vader_sentiment(text)

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

            # Default: trust LLM over VADER when they disagree
            # LLM has full document context, VADER only first 1000 chars
            return llm_result

        except Exception as e:
            logger.error("Sentiment analysis crashed: %s", e)
            return "Neutral"
