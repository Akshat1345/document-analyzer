"""Lightweight document-scoped QA service using cached text retrieval."""

from __future__ import annotations

import re
from typing import List

from app.services.cache import CacheService
from app.services.groq_client import get_answer_from_context


class QAService:
    """Answer questions from text indexed by document id in Redis."""

    def __init__(self, cache: CacheService) -> None:
        self.cache = cache

    @staticmethod
    def _chunk_text(text: str, chunk_words: int = 180) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks: List[str] = []
        for i in range(0, len(words), chunk_words):
            chunk = " ".join(words[i : i + chunk_words]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _score_chunk(chunk: str, query_terms: set[str]) -> int:
        lowered = chunk.lower()
        return sum(1 for term in query_terms if term in lowered)

    def _select_context(self, text: str, question: str, top_k: int) -> List[str]:
        chunks = self._chunk_text(text)
        if not chunks:
            return []

        terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        if not terms:
            return chunks[:top_k]

        ranked = sorted(chunks, key=lambda c: self._score_chunk(c, terms), reverse=True)
        return ranked[:top_k]

    async def answer_question(self, document_id: str, question: str, top_k: int = 4) -> tuple[str, List[str]]:
        doc = await self.cache.get_document_text(document_id)
        if not doc or not doc.get("text"):
            return "Document context not found. Re-run analysis for this document.", []

        contexts = self._select_context(str(doc["text"]), question, max(1, min(top_k, 8)))
        if not contexts:
            return "Document context not found. Re-run analysis for this document.", []

        answer = get_answer_from_context(question=question, context_chunks=contexts)
        if not answer:
            answer = "Not found in this document."

        citations = [f"Chunk {idx + 1}" for idx in range(len(contexts))]
        return answer, citations


qa_service_instance: QAService | None = None


def set_qa_service_instance(instance: QAService) -> None:
    """Set singleton QA service instance for router access."""

    global qa_service_instance
    qa_service_instance = instance
