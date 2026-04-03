"""Simple keyword-driven document categorization."""


class DocumentClassifier:
    """Classify a document into a predefined category using lexical cues."""

    CATEGORIES = {
        "invoice": ["invoice", "payment", "amount due", "billing", "total", "subtotal", "tax", "receipt"],
        "contract": ["agreement", "whereas", "party", "parties", "obligations", "clause", "hereby", "terms"],
        "news_article": ["reported", "according to", "sources", "incident", "announced", "officials"],
        "academic": ["abstract", "methodology", "findings", "conclusion", "research", "study", "results"],
        "resume": ["experience", "education", "skills", "work history", "objective", "references"],
        "financial_report": ["revenue", "profit", "quarter", "fiscal", "earnings", "forecast"],
        "incident_report": ["incident", "breach", "affected", "investigation", "security", "attack"],
        "general": [],
    }

    NEUTRAL_BIASED = {"invoice", "contract", "academic"}

    def classify(self, text: str) -> str:
        """Return best matching category by keyword hit count with tie-to-general."""

        lowered = text.lower()
        best_category = "general"
        best_score = 0
        tied = False

        for category, keywords in self.CATEGORIES.items():
            if category == "general":
                continue
            score = sum(1 for keyword in keywords if keyword in lowered)
            if score > best_score:
                best_score = score
                best_category = category
                tied = False
            elif score == best_score and score > 0:
                tied = True

        if tied or best_score == 0:
            return "general"
        return best_category
