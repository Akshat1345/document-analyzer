"""Document-scoped Q&A API endpoint router."""

from typing import Union

from fastapi import APIRouter, Depends

from app.dependencies import validate_api_key
from app.models.schemas import DocumentQuestionRequest, DocumentQuestionResponse, ErrorResponse
from app.services import rag_service as rag_module

router = APIRouter()


@router.post("/api/document-qa")
async def ask_document_question(
    request: DocumentQuestionRequest,
    api_key: str = Depends(validate_api_key),
) -> Union[DocumentQuestionResponse, ErrorResponse]:
    """Answer a question using only embeddings/context from the specified document ID."""

    del api_key

    if rag_module.rag_service_instance is None:
        return ErrorResponse(status="error", message="RAG service not initialized")

    try:
        answer, citations = await rag_module.rag_service_instance.answer_question(
            document_id=request.documentId,
            question=request.question,
            top_k=request.topK,
        )
        return DocumentQuestionResponse(
            status="success",
            documentId=request.documentId,
            question=request.question,
            answer=answer,
            citations=citations,
        )
    except ValueError as exc:
        return ErrorResponse(status="error", message=str(exc))
    except Exception:
        return ErrorResponse(status="error", message="Question answering failed. Please try again.")
