"""Document-scoped RAG service with strict per-document retrieval isolation."""

import json
import logging
import math
import threading
from typing import Any, Optional

import torch
from redis import asyncio as redis
from transformers import AutoModel, AutoTokenizer

from app.config import settings
from app.services.groq_client import get_rag_answer_from_context

logger = logging.getLogger(__name__)

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    words = _normalize_whitespace(text).split(" ")
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step
    return chunks


class EmbeddingModel:
    """Lazy-loaded sentence embedding model singleton."""

    _tokenizer = None
    _model = None
    _lock = threading.Lock()

    @classmethod
    def _load(cls) -> None:
        with cls._lock:
            if cls._tokenizer is not None and cls._model is not None:
                return
            cls._tokenizer = AutoTokenizer.from_pretrained(_EMBED_MODEL_NAME)
            cls._model = AutoModel.from_pretrained(_EMBED_MODEL_NAME)
            cls._model.eval()
            logger.info("RAG embedding model loaded: %s", _EMBED_MODEL_NAME)

    @classmethod
    def encode(cls, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        cls._load()
        assert cls._tokenizer is not None
        assert cls._model is not None

        with torch.inference_mode():
            batch = cls._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            output = cls._model(**batch)
            token_embeddings = output.last_hidden_state
            attention_mask = batch["attention_mask"].unsqueeze(-1)
            masked = token_embeddings * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

        return normalized.detach().cpu().tolist()


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class RAGService:
    """Persist and query document-specific embeddings with hard isolation by document ID."""

    def __init__(self) -> None:
        self._redis = None
        self._memory_store: dict[str, dict[str, Any]] = {}

    async def connect(self) -> None:
        try:
            self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
            await self._redis.ping()
        except Exception as exc:
            logger.warning("Redis unavailable for RAG, falling back to memory store: %s", exc)
            self._redis = None

    @staticmethod
    def _doc_chunks_key(document_id: str) -> str:
        return f"rag:doc:{document_id}:chunks"

    @staticmethod
    def _doc_meta_key(document_id: str) -> str:
        return f"rag:doc:{document_id}:meta"

    async def index_document(self, document_id: str, file_name: str, text: str) -> int:
        chunks = _chunk_text(text)
        if not chunks:
            return 0

        vectors = EmbeddingModel.encode(chunks)
        payload = {
            "fileName": file_name,
            "chunks": chunks,
            "vectors": vectors,
        }

        if self._redis is not None:
            try:
                await self._redis.set(self._doc_chunks_key(document_id), json.dumps(payload))
                await self._redis.set(
                    self._doc_meta_key(document_id),
                    json.dumps({"fileName": file_name, "chunkCount": len(chunks)}),
                )
                return len(chunks)
            except Exception as exc:
                logger.warning("Failed to persist RAG vectors for %s: %s", document_id, exc)

        self._memory_store[document_id] = payload
        return len(chunks)

    async def _get_doc_payload(self, document_id: str) -> Optional[dict[str, Any]]:
        if self._redis is not None:
            try:
                raw = await self._redis.get(self._doc_chunks_key(document_id))
                if raw:
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("RAG retrieval failed for %s: %s", document_id, exc)

        return self._memory_store.get(document_id)

    async def answer_question(self, document_id: str, question: str, top_k: int = 4) -> tuple[str, list[str]]:
        payload = await self._get_doc_payload(document_id)
        if not payload:
            raise ValueError("Document not indexed. Analyze/upload this document first.")

        chunks: list[str] = payload.get("chunks", [])
        vectors: list[list[float]] = payload.get("vectors", [])
        if not chunks or not vectors:
            raise ValueError("Document index is empty. Re-upload and analyze again.")

        q_vector = EmbeddingModel.encode([question])[0]
        scored = [
            (_cosine_similarity(q_vector, vec), idx)
            for idx, vec in enumerate(vectors)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        selected = scored[: min(top_k, len(scored))]
        context_chunks = [chunks[idx] for _, idx in selected]
        citations = [f"Chunk {idx + 1}" for _, idx in selected]

        answer = get_rag_answer_from_context(question=question, context_chunks=context_chunks)
        if not answer:
            answer = "I could not find enough evidence in this document to answer confidently."

        return answer, citations


rag_service_instance: Optional[RAGService] = None


def set_rag_service_instance(instance: RAGService) -> None:
    global rag_service_instance
    rag_service_instance = instance
