"""Entity-aware summarizer with evidence ranking and extractive fallback."""

import logging
import re

from app.models.schemas import EntitiesResponse
from app.services.groq_client import get_summary_from_claude

logger = logging.getLogger(__name__)


class Summarizer:
    """Summarize document text with category-specific focus rules."""

    SUMMARY_KEYWORDS = {
        "invoice": {"invoice", "payment", "amount", "due", "total", "tax", "vendor", "client"},
        "contract": {"agreement", "party", "parties", "clause", "term", "obligation", "effective", "signed"},
        "news_article": {"reported", "announced", "said", "according", "impact", "government", "company"},
        "academic": {"study", "research", "method", "results", "findings", "abstract", "conclusion"},
        "resume": {"experience", "education", "skills", "role", "lead", "managed", "built", "developed"},
        "financial_report": {"revenue", "profit", "loss", "quarter", "earnings", "forecast", "growth"},
        "incident_report": {"breach", "attack", "incident", "affected", "investigation", "response", "recovery"},
        "general": {"reported", "announced", "developed", "introduced", "improved", "includes", "resulted"},
    }

    VAGUE_PATTERNS = re.compile(
        r"\b(the document|this document|it discusses|it covers|various|several|multiple|many|important|key points|broadly|generally|overall|highlights|explores|focuses on|describes)\b",
        re.IGNORECASE,
    )

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

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into coarse sentences while preserving document order."""

        raw_blocks = [block.strip() for block in re.split(r"\n{2,}", text.strip()) if block.strip()]
        if not raw_blocks:
            raw_blocks = [text.strip()]

        sentences: list[str] = []
        for block in raw_blocks:
            normalized_block = re.sub(r"\s*\n\s*", " ", block)
            pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", normalized_block)
            for piece in pieces:
                split_piece = self._expand_ocr_sentence(piece)
                for part in split_piece:
                    cleaned = self._clean_sentence(part)
                    if cleaned:
                        sentences.append(cleaned)
        if sentences:
            return sentences
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        return [self._clean_sentence(paragraph) for paragraph in paragraphs] if paragraphs else [self._clean_sentence(text.strip())]

    def _expand_ocr_sentence(self, sentence: str) -> list[str]:
        """Break long OCR-style lines on visual separators."""

        if len(sentence) < 140 and not any(marker in sentence for marker in ["|", "—", "-", "•", "·"]):
            return [sentence]
        parts = re.split(r"\s*(?:\||•|·|—|\s-\s)\s*", sentence)
        expanded = [part for part in parts if part and len(part.strip()) > 20]
        return expanded if expanded else [sentence]

    def _clean_sentence(self, sentence: str) -> str:
        """Normalize OCR noise and whitespace for cleaner summary sentences."""

        cleaned = re.sub(r"\s+", " ", sentence).strip()
        cleaned = re.sub(r"^[^A-Za-z0-9]+", "", cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return cleaned.strip()

    def _sentence_score(self, sentence: str, entities: EntitiesResponse, doc_category: str) -> int:
        """Score sentences by how much factual signal they carry."""

        sentence = self._clean_sentence(sentence)
        lowered = sentence.lower()
        score = 0
        if any(name.lower() in lowered for name in entities.names):
            score += 4
        if any(org.lower() in lowered for org in entities.organizations):
            score += 4
        if any(date.lower() in lowered for date in entities.dates):
            score += 3
        if any(amount.lower() in lowered for amount in entities.amounts):
            score += 3
        if re.search(r"\d", sentence):
            score += 2
        if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b", sentence):
            score += 2
        keywords = self.SUMMARY_KEYWORDS.get(doc_category, self.SUMMARY_KEYWORDS["general"])
        score += sum(1 for keyword in keywords if keyword in lowered)
        symbol_ratio = len(re.findall(r"[^A-Za-z0-9\s.,;:'\"()-]", sentence)) / max(len(sentence), 1)
        if symbol_ratio > 0.08:
            score -= 2
        if len(sentence.split()) >= 8:
            score += 1
        if len(sentence.split()) > 35:
            score -= 1
        return score

    def _select_evidence_sentences(
        self,
        text: str,
        entities: EntitiesResponse,
        doc_category: str,
        limit: int = 8,
    ) -> list[str]:
        """Pick the most fact-dense sentences to ground the summary."""

        sentences = self._split_sentences(text)
        ranked = [
            (index, sentence, self._sentence_score(sentence, entities, doc_category))
            for index, sentence in enumerate(sentences)
        ]
        ranked.sort(key=lambda item: (-item[2], item[0]))

        chosen: list[tuple[int, str]] = []
        seen: set[str] = set()
        for index, sentence, score in ranked:
            if score <= 0 and chosen:
                continue
            normalized = sentence.strip()
            if not normalized or normalized.casefold() in seen:
                continue
            seen.add(normalized.casefold())
            chosen.append((index, normalized))
            if len(chosen) >= limit:
                break

        if not chosen:
            for index, sentence in enumerate(sentences[:limit]):
                normalized = sentence.strip()
                if normalized and normalized.casefold() not in seen:
                    seen.add(normalized.casefold())
                    chosen.append((index, normalized))

        chosen.sort(key=lambda item: item[0])
        return [sentence for _, sentence in chosen]

    def _build_summary_context(self, text: str, entities: EntitiesResponse, doc_category: str) -> str:
        """Construct a grounded evidence packet for the LLM."""

        evidence = self._select_evidence_sentences(text, entities, doc_category, limit=10)
        if not evidence:
            evidence = self._split_sentences(text)[:6]
        return "\n".join(f"- {self._clean_sentence(sentence)}" for sentence in evidence)

    def _is_vague_summary(self, summary: str) -> bool:
        """Detect generic summary language that lacks concrete facts."""

        lowered = summary.strip().lower()
        if not lowered:
            return True
        if self.VAGUE_PATTERNS.search(lowered):
            return True
        if len(summary.split()) < 18:
            return True
        concrete_hits = len(re.findall(r"\b[A-Z][a-z]+\b", summary)) + len(re.findall(r"\d", summary))
        return concrete_hits < 2

    def _extractive_fallback(self, text: str, entities: EntitiesResponse, doc_category: str) -> str:
        """Build a fact-dense fallback summary from evidence sentences."""

        evidence = self._select_evidence_sentences(text, entities, doc_category, limit=6)
        if not evidence:
            return "Summary unavailable."
        cleaned = [self._clean_sentence(sentence) for sentence in evidence if self._clean_sentence(sentence)]
        return " ".join(sentence if sentence.endswith((".", "!", "?")) else f"{sentence}." for sentence in cleaned)

    def _chunk_text(self, text: str, max_words: int = 3000) -> list[str]:
        """Split long documents into section chunks below word threshold."""

        chunks = []
        current = []
        current_words = 0
        for section in text.split("\n\n"):
            para = section.strip()
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

    def summarize(
        self,
        text: str,
        doc_category: str,
        entities: EntitiesResponse
    ) -> str:
        """
        Generate a dense, factual, entity-anchored summary.
        Uses Chain-of-Density style prompting for maximum information
        density. Falls back to extractive summary if API fails.
        """

        # ── Build entity context string ───────────────────────────────
        entity_parts = []
        if entities.names:
            entity_parts.append(
                f"People: {', '.join(entities.names[:5])}"
            )
        if entities.organizations:
            entity_parts.append(
                f"Organizations: {', '.join(entities.organizations[:5])}"
            )
        if entities.dates:
            entity_parts.append(
                f"Dates: {', '.join(entities.dates[:5])}"
            )
        if entities.amounts:
            entity_parts.append(
                f"Amounts: {', '.join(entities.amounts[:5])}"
            )
        entity_context = (
            " | ".join(entity_parts)
            if entity_parts
            else "None identified"
        )

        # ── Document-type-specific focus instructions ─────────────────
        DOC_FOCUS = {
            "invoice": (
                "State: who issued it, to whom, for what amount, "
                "on what date, and for what product or service. "
                "Include invoice number if present."
            ),
            "contract": (
                "State: the parties involved (full names), the core "
                "obligation or agreement, the key terms, and the "
                "effective or expiry dates."
            ),
            "news_article": (
                "State: what event occurred, who was involved, where "
                "it happened, when it happened, and what the "
                "consequence or impact was."
            ),
            "academic": (
                "State: the research topic, the institution or "
                "researchers involved, the methodology used, and the "
                "key finding or conclusion reached."
            ),
            "resume": (
                "State: the candidate's full name, their most recent "
                "job title and employer, their total experience, their "
                "top 1-2 measurable achievements, and their "
                "educational background."
            ),
            "financial_report": (
                "State: the company name, the reporting period, the "
                "key financial metrics (revenue, profit, growth rate), "
                "and the outlook or strategic direction."
            ),
            "incident_report": (
                "State: what the incident was, what systems or "
                "organisations were affected, how it occurred, when "
                "it was discovered, and what response or "
                "investigation followed."
            ),
            "general": (
                "State: the main subject, the most important facts "
                "and figures, the key people or organisations "
                "involved, and the primary outcome or conclusion."
            ),
        }
        focus = DOC_FOCUS.get(doc_category, DOC_FOCUS["general"])

        # ── Handle long documents ─────────────────────────────────────
        word_count = len(text.split())
        if word_count > 4000:
            chunks = self._chunk_text(text, max_words=3000)
            if len(chunks) > 1:
                chunk_summaries = []
                for chunk in chunks:
                    chunk_prompt = (
                        f"Summarize this section in 1-2 sentences. "
                        f"Be specific and factual. Include names, "
                        f"numbers, and key facts:\n\n{chunk}"
                    )
                    s = get_summary_from_claude(chunk_prompt)
                    if s:
                        chunk_summaries.append(s)
                text_to_summarize = "\n\n".join(chunk_summaries)
            else:
                text_to_summarize = text[:4000]
        else:
            text_to_summarize = text

        # ── Build the master summary prompt ──────────────────────────
        prompt = (
            f"You are a world-class analyst and technical writer who "
            f"produces the most information-dense summaries possible.\n\n"
            f"━━━ YOUR TASK ━━━\n"
            f"Write a precise, factual summary of the document below.\n\n"
            f"━━━ DOCUMENT CONTEXT ━━━\n"
            f"Document type: {doc_category}\n"
            f"Key entities already extracted: {entity_context}\n"
            f"Summary focus: {focus}\n\n"
            f"━━━ STRICT RULES ━━━\n"
            f"1. Write EXACTLY 2 to 4 sentences. Not 1. Not 5.\n"
            f"2. EVERY sentence must contain specific facts, names, "
            f"numbers, or outcomes. No sentence can be vague.\n"
            f"3. You MUST include the key entities listed above "
            f"wherever they are relevant. Do not ignore them.\n"
            f"4. Include specific names, figures, dates, and amounts "
            f"from the document wherever they appear.\n"
            f"5. NEVER begin with:\n"
            f"   'This document...'\n"
            f"   'The document...'\n"
            f"   'This text...'\n"
            f"   'This report...'\n"
            f"   'This article...'\n"
            f"   'Overview:'\n"
            f"   'Summary:'\n"
            f"6. NEVER use these filler phrases:\n"
            f"   'Overall, ...'\n"
            f"   'In conclusion, ...'\n"
            f"   'It is important to note...'\n"
            f"   'The document discusses...'\n"
            f"   'It covers...'\n"
            f"   'This piece explores...'\n"
            f"7. Write in third person.\n"
            f"8. The summary must be self-contained — a reader with "
            f"zero access to the original must understand the core "
            f"facts from your 2-4 sentences alone.\n"
            f"9. Return ONLY the summary text. No labels, no "
            f"'Summary:', no quotes, no preamble.\n\n"
            f"━━━ BENCHMARK EXAMPLES ━━━\n\n"
            f"EXAMPLE 1 (invoice):\n"
            f"'ABC Pvt Ltd issued invoice #INV-2026-042 to Ravi Kumar "
            f"on 10 March 2026 for professional consulting services "
            f"totalling ₹10,000, with payment due within 30 days. "
            f"The invoice includes applicable GST and references "
            f"contract #C-2025-18.'\n\n"
            f"EXAMPLE 2 (cybersecurity incident):\n"
            f"'A large-scale data breach exposed customer account "
            f"records and transaction histories across multiple "
            f"financial institutions after attackers exploited a "
            f"zero-day vulnerability in a shared third-party "
            f"authentication platform. Affected banks temporarily "
            f"suspended online services while regulatory authorities "
            f"launched compliance investigations. Cybersecurity "
            f"experts recommended immediate implementation of "
            f"stronger encryption and real-time monitoring systems.'\n\n"
            f"EXAMPLE 3 (resume):\n"
            f"'Nina Lane is a graphic designer with over 5 years of "
            f"professional experience, currently serving as Senior "
            f"Graphic Designer at Brightline Agency in New York since "
            f"June 2020. She previously worked at Blue Horizon Media "
            f"from March 2017 to May 2020, where she boosted client "
            f"retention by 25% through eco-friendly packaging design. "
            f"She holds a Bachelor of Fine Arts in Graphic Design from "
            f"Parsons School of Design, graduating in 2017.'\n\n"
            f"EXAMPLE 4 (technology article):\n"
            f"'Google, Microsoft, and NVIDIA are leading a global "
            f"expansion of artificial intelligence infrastructure, "
            f"with governments and universities worldwide increasing "
            f"investment in machine learning research and AI-powered "
            f"applications. AI systems are demonstrating measurable "
            f"impact in healthcare diagnostics, manufacturing "
            f"automation, and financial analytics, while creating "
            f"new employment in data science and AI engineering. "
            f"Economists project continued economic growth from AI "
            f"adoption across sectors over the coming decade.'\n\n"
            f"━━━ DOCUMENT TO SUMMARIZE ━━━\n\n"
            f"{text_to_summarize[:4000]}\n\n"
            f"━━━ WRITE THE SUMMARY NOW ━━━"
        )

        result = get_summary_from_claude(prompt)

        if not result or len(result.strip()) < 20:
            logger.warning(
                "LLM summary empty or too short, using extractive fallback"
            )
            return self._extractive_fallback(text, entities, doc_category)

        return result.strip()
