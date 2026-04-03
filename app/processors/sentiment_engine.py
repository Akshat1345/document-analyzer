"""Ensemble sentiment engine combining FinBERT, RoBERTa, and VADER."""

from __future__ import annotations

import logging
from typing import Any

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class SentimentEngine:
    """Document-type-aware sentiment inference using weighted model ensemble."""

    _finbert: Any = None
    _roberta: Any = None
    _vader: Any = None

    WEIGHTS = {
        "invoice": {"finbert": 0.6, "roberta": 0.2, "vader": 0.2},
        "contract": {"finbert": 0.5, "roberta": 0.2, "vader": 0.3},
        "financial_report": {"finbert": 0.6, "roberta": 0.2, "vader": 0.2},
        "news_article": {"finbert": 0.2, "roberta": 0.5, "vader": 0.3},
        "incident_report": {"finbert": 0.3, "roberta": 0.4, "vader": 0.3},
        "resume": {"finbert": 0.2, "roberta": 0.5, "vader": 0.3},
        "academic": {"finbert": 0.3, "roberta": 0.3, "vader": 0.4},
        "general": {"finbert": 0.34, "roberta": 0.33, "vader": 0.33},
    }

    NEUTRAL_BIASED = {"invoice", "contract", "academic"}

    @classmethod
    def initialize(cls) -> None:
        """Load sentiment models once at startup with graceful fallbacks."""

        try:
            cls._finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True,
            )
        except Exception as exc:
            logger.warning("Failed to load FinBERT: %s", exc)
            cls._finbert = None

        try:
            cls._roberta = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True,
            )
        except Exception as exc:
            logger.warning("Failed to load RoBERTa: %s", exc)
            cls._roberta = None

        try:
            cls._vader = SentimentIntensityAnalyzer()
        except Exception as exc:
            logger.warning("Failed to load VADER: %s", exc)
            cls._vader = None

    def _run_finbert(self, text: str) -> dict[str, float]:
        """Run FinBERT and return normalized score dictionary."""

        if self._finbert is None:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
        raw_scores: Any = self._finbert(text[:512])
        scores = raw_scores[0]
        out = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        for item in scores:
            out[item["label"].lower()] = float(item["score"])
        return out

    def _run_roberta(self, text: str) -> dict[str, float]:
        """Run RoBERTa and map label IDs to sentiment names."""

        if self._roberta is None:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
        raw_scores: Any = self._roberta(text[:512])
        scores = raw_scores[0]
        mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
        out = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        for item in scores:
            out[mapping.get(item["label"], "neutral")] = float(item["score"])
        return out

    def _run_vader(self, text: str) -> dict[str, float | str]:
        """Run VADER and return per-label scores plus coarse label."""

        if self._vader is None:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0, "label": "neutral"}
        scores = self._vader.polarity_scores(text[:1000])
        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return {
            "positive": float(scores.get("pos", 0.0)),
            "neutral": float(scores.get("neu", 0.0)),
            "negative": float(scores.get("neg", 0.0)),
            "label": label,
        }

    def analyze(self, text: str, doc_category: str) -> str:
        """Return final sentiment label as Positive, Neutral, or Negative."""

        try:
            analysis_text = f"{text[:1000]} {text[-500:]}" if len(text) > 1000 else text
            finbert_scores = self._run_finbert(analysis_text)
            roberta_scores = self._run_roberta(analysis_text)
            vader_scores = self._run_vader(analysis_text)
            weights = self.WEIGHTS.get(doc_category, self.WEIGHTS["general"])

            weighted = {}
            for label in ["positive", "neutral", "negative"]:
                weighted[label] = (
                    finbert_scores[label] * weights["finbert"]
                    + roberta_scores[label] * weights["roberta"]
                    + float(vader_scores[label]) * weights["vader"]
                )

            winner = max(weighted, key=lambda label: weighted[label])
            if doc_category in self.NEUTRAL_BIASED and winner != "neutral" and weighted[winner] < 0.80:
                winner = "neutral"
            return winner.capitalize()
        except Exception as exc:
            logger.warning("Sentiment analysis failed: %s", exc)
            return "Neutral"
