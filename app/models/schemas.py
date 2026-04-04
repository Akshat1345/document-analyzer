"""Pydantic schemas for requests, responses, and internal processing models."""

from typing import List, Literal

from pydantic import BaseModel, Field


class DocumentRequest(BaseModel):
    """Input payload for document analysis requests."""

    fileName: str
    fileType: Literal["pdf", "docx", "image"]
    fileBase64: str


class EntitiesResponse(BaseModel):
    """Structured entities extracted from a document."""

    names: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    organizations: List[str] = Field(default_factory=list)
    amounts: List[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Successful API response payload."""

    status: str = "success"
    fileName: str
    documentId: str | None = None
    summary: str
    entities: EntitiesResponse
    sentiment: Literal["Positive", "Neutral", "Negative"]


class DocumentQuestionRequest(BaseModel):
    """Input payload for document-scoped question answering."""

    documentId: str
    question: str = Field(min_length=3, max_length=1000)
    topK: int = Field(default=4, ge=1, le=8)


class DocumentQuestionResponse(BaseModel):
    """Successful response for document-scoped question answering."""

    status: str = "success"
    documentId: str
    question: str
    answer: str
    citations: List[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Error API response payload."""

    status: str = "error"
    message: str


class RawEntity(BaseModel):
    """Internal representation for a raw extracted entity."""

    type: str
    value: str
    confidence: float = 0.7
    sources: List[str] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Internal metadata generated during extraction and classification."""

    file_type: str
    word_count: int = 0
    language: str = "en"
    doc_category: str = "general"
    extraction_method: str = ""
    has_tables: bool = False
    page_count: int = 1
