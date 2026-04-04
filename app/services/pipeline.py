"""Asynchronous analysis pipeline orchestration."""

import asyncio
import logging
from typing import Any, Optional

from app.extractors.docx_extractor import DOCXExtractor
from app.extractors.image_extractor import ImageExtractor
from app.extractors.pdf_extractor import PDFExtractor
from app.config import settings
from app.models.schemas import AnalysisResponse, DocumentRequest, ErrorResponse
from app.processors.document_classifier import DocumentClassifier
from app.utils.helpers import compute_hash, decode_base64

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Coordinates extraction, classification, NER, sentiment, and summary."""

    def __init__(self, ner_engine, sentiment_engine, summarizer, cache):
        """Store core processing engine instances."""

        self.ner_engine = ner_engine
        self.sentiment_engine = sentiment_engine
        self.summarizer = summarizer
        self.cache = cache

    async def process(self, request: DocumentRequest) -> AnalysisResponse | ErrorResponse:
        """Process document request and return success or error schema."""

        try:
            try:
                content = decode_base64(request.fileBase64)
            except ValueError as exc:
                return ErrorResponse(status="error", message=str(exc))

            content_hash = compute_hash(content)
            cache_key = self.cache.build_key(content_hash)
            if settings.USE_CACHE:
                cached = await self.cache.get(cache_key)
                if cached:
                    return AnalysisResponse(**cached)

            if request.fileType == "pdf":
                extractor = PDFExtractor()
            elif request.fileType == "docx":
                extractor = DOCXExtractor()
            else:
                extractor = ImageExtractor()

            text, _metadata = extractor.extract(content)
            if len(text.strip()) < 20:
                return ErrorResponse(status="error", message="Could not extract readable text")

            doc_category = DocumentClassifier().classify(text)

            ner_task = asyncio.create_task(
                asyncio.to_thread(self.ner_engine.extract_all, text, doc_category)
            )
            sentiment_task = asyncio.create_task(
                asyncio.to_thread(self.sentiment_engine.analyze, text, doc_category)
            )
            entities, sentiment = await asyncio.gather(ner_task, sentiment_task)

            summary = await asyncio.to_thread(self.summarizer.summarize, text, doc_category, entities)

            response = AnalysisResponse(
                status="success",
                fileName=request.fileName,
                documentId=content_hash,
                summary=summary or "Summary unavailable.",
                entities=entities,
                sentiment=sentiment,
            )

            await self.cache.set_document_text(
                document_id=content_hash,
                file_name=request.fileName,
                text=text,
            )

            if settings.USE_CACHE:
                await self.cache.set(cache_key, response.model_dump())
            return response
        except Exception as exc:
            logger.exception("Pipeline processing failed: %s", exc)
            return ErrorResponse(status="error", message="Processing failed. Please try again.")


pipeline_instance: Optional[AnalysisPipeline] = None


def set_pipeline_instance(instance: Any) -> None:
    """Set importable singleton pipeline instance for router usage."""

    global pipeline_instance
    pipeline_instance = instance
