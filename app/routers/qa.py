"""Document-scoped QA endpoint router."""

from typing import Union

from fastapi import APIRouter, Depends

from app.dependencies import validate_api_key
from app.models.schemas import (
    DocumentQuestionRequest,
    DocumentQuestionResponse,
    ErrorResponse,
)
from app.services import qa_service as qa_module

router = APIRouter()


@router.post("/api/document-qa")
async def document_qa(
    request: DocumentQuestionRequest,
    api_key: str = Depends(validate_api_key),
) -> Union[DocumentQuestionResponse, ErrorResponse]:
    """Answer questions from the analyzed document identified by documentId."""

    del api_key
    if qa_module.qa_service_instance is None:
        return ErrorResponse(status="error", message="QA service not initialized")

    answer, citations = await qa_module.qa_service_instance.answer_question(
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
