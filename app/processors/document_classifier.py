"""Simple keyword-driven document categorization."""

import re


class DocumentClassifier:
    """Classify a document into a predefined category using lexical cues."""

    CATEGORIES = {
        "invoice": ["invoice", "payment", "amount due", "billing", "total", "subtotal", "tax", "receipt"],
        "contract": ["agreement", "whereas", "party", "parties", "obligations", "clause", "hereby", "terms"],
        "news_article": ["reported", "according to", "sources", "incident", "announced", "officials"],
        "academic": ["abstract", "methodology", "findings", "conclusion", "research", "study", "results"],
        "resume": [
            "work experience",
            "professional experience",
            "education",
            "skills",
            "contact",
            "portfolio",
            "references",
            "curriculum vitae",
            "resume",
        ],
        "financial_report": ["revenue", "profit", "quarter", "fiscal", "earnings", "forecast"],
        "incident_report": ["incident", "breach", "affected", "investigation", "security", "attack"],
        "general": [],
    }

    NEUTRAL_BIASED = {"invoice", "contract", "academic"}

    def _count_keyword_hits(self, lowered: str, keywords: list[str]) -> int:
        """Count keyword hits with boundary-aware matching for single terms."""

        hits = 0
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if " " in keyword_lower:
                if keyword_lower in lowered:
                    hits += 1
                continue

            pattern = rf"\b{re.escape(keyword_lower)}\b"
            if re.search(pattern, lowered):
                hits += 1
        return hits

    def classify(self, text: str) -> str:
        """Return best matching category by keyword hit count with tie-to-general."""

        lowered = text.lower()
        best_category = "general"
        best_score = 0
        tied = False

        for category, keywords in self.CATEGORIES.items():
            if category == "general":
                continue
            score = self._count_keyword_hits(lowered, keywords)

            # Resume should require strong CV-like structure cues.
            if category == "resume" and score < 3:
                score = 0

            if score > best_score:
                best_score = score
                best_category = category
                tied = False
            elif score == best_score and score > 0:
                tied = True

        if tied or best_score == 0:
            return "general"
        return best_category
