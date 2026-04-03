"""Entity-aware summarizer with chunking and extractive fallback."""

from app.models.schemas import EntitiesResponse
from app.services.groq_client import get_summary_from_claude


class Summarizer:
    """Summarize document text with category-specific focus rules."""

    def _build_entity_context(self, entities: EntitiesResponse) -> str:
        """Create compact string context from non-empty entity lists."""

        parts = []
        if entities.names:
            parts.append(f"People: {', '.join(entities.names)}")
        if entities.organizations:
            parts.append(f"Organizations: {', '.join(entities.organizations)}")
        if entities.dates:
            parts.append(f"Dates: {', '.join(entities.dates)}")
        if entities.amounts:
            parts.append(f"Amounts: {', '.join(entities.amounts)}")
        return " | ".join(parts) if parts else "No key entities identified"

    def _extractive_fallback(self, text: str) -> str:
        """Build fallback summary from first sentence of top paragraphs."""

        sentences = []
        for paragraph in [p.strip() for p in text.split("\n\n") if p.strip()]:
            first = paragraph.split(". ")[0].strip()
            if first:
                sentences.append(first if first.endswith(".") else f"{first}.")
        return " ".join(sentences[:3]) if sentences else "Summary unavailable."

    def _chunk_text(self, text: str, max_words: int = 3000) -> list[str]:
        """Split long documents into paragraph chunks below word threshold."""

        chunks = []
        current = []
        current_words = 0
        for paragraph in text.split("\n\n"):
            para = paragraph.strip()
            if not para:
                continue
            para_words = len(para.split())
            if current and current_words + para_words > max_words:
                chunks.append("\n\n".join(current))
                current = [para]
                current_words = para_words
            else:
                current.append(para)
                current_words += para_words
        if current:
            chunks.append("\n\n".join(current))
        return chunks if chunks else [text]

    def summarize(self, text: str, doc_category: str, entities: EntitiesResponse) -> str:
        """Return an adaptive length factual summary with model fallback handling."""

        entity_context = self._build_entity_context(entities)
        word_count = len(text.split())
        is_long_doc = word_count > 5000
        name_rule = (
            "Do not mention any person names unless they are explicitly present in the document text or Key entities identified."
            if not entities.names
            else "Only mention person names that are explicitly present in the document text or Key entities identified."
        )
        instructions = {
            "invoice": "Emphasize: parties involved, amounts, dates, purpose",
            "contract": "Emphasize: parties, key obligations, terms, dates",
            "news_article": "Emphasize: what happened, who, where, impact",
            "academic": "Emphasize: research question, method, key findings",
            "resume": "Emphasize: candidate name, role, top achievements",
            "financial_report": "Emphasize: company, key metrics, performance, outlook",
            "incident_report": "Emphasize: what happened, affected parties, response",
            "general": "Emphasize: main topic, key facts, outcomes",
        }
        focus = instructions.get(doc_category, instructions["general"])

        if word_count > 4000:
            chunk_summaries = []
            for chunk in self._chunk_text(text):
                chunk_prompt = (
                    f"Document type: {doc_category}\n"
                    f"Key entities identified: {entity_context}\n"
                    f"Focus instruction: {focus}\n\n"
                    "Write a concise factual summary. Prioritize accuracy and key facts over length.\n\n"
                    f"Document:\n{chunk[:3000]}"
                )
                partial = get_summary_from_claude(chunk_prompt)
                if partial.strip():
                    chunk_summaries.append(partial.strip())
            text_to_summarize = "\n\n".join(chunk_summaries) if chunk_summaries else text
        else:
            text_to_summarize = text

        prompt = (
            f"Document type: {doc_category}\n"
            f"Key entities identified: {entity_context}\n"
            f"Focus instruction: {focus}\n\n"
            f"Write a comprehensive factual summary. For brief documents, 2-3 sentences suffice. For longer or complex documents, expand to 4-6 sentences as needed. Prioritize accuracy and captured facts over word count.\n"
            "Rules:\n"
            "- Only use facts explicitly present in the text or identified entities\n"
            f"- {name_rule}\n"
            "- Never invent people, organizations, dates, or amounts\n"
            "- Never start with 'This document'\n"
            "- Use clear, technical language - no filler phrases\n"
            "- Include specific names, dates, and outcomes only when explicitly present\n\n"
            f"Document:\n{text_to_summarize[:4000]}"
        )
        result = get_summary_from_claude(prompt)
        if not result.strip():
            return self._extractive_fallback(text)
        return result.strip()
