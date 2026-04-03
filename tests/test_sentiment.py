"""Sentiment engine logic tests for weighted label behavior."""

from app.processors.sentiment_engine import SentimentEngine


class StubSentiment(SentimentEngine):
    """Stub sentiment engine to validate weighted decision logic."""

    def _run_finbert(self, text: str):
        return {"positive": 0.8, "neutral": 0.1, "negative": 0.1}

    def _run_roberta(self, text: str):
        return {"positive": 0.6, "neutral": 0.2, "negative": 0.2}

    def _run_vader(self, text: str):
        return {"positive": 0.7, "neutral": 0.2, "negative": 0.1, "label": "positive"}


def test_sentiment_capitalized_label() -> None:
    """Ensemble output must always be one of three title-cased labels."""

    engine = StubSentiment()
    label = engine.analyze("Great growth and strong outcomes.", "financial_report")
    assert label in ["Positive", "Neutral", "Negative"]
    assert label == "Positive"
