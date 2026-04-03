"""Document analysis API endpoint router."""

from typing import Union

from fastapi import APIRouter, Depends

from app.config import settings
from app.dependencies import validate_api_key, validate_file_size
from app.models.schemas import AnalysisResponse, DocumentRequest, ErrorResponse
from app.services import pipeline as pipeline_module

router = APIRouter()


@router.post("/api/document-analyze")
async def analyze_document(
    request: DocumentRequest,
    api_key: str = Depends(validate_api_key),
    file_size: int = Depends(validate_file_size),
) -> Union[AnalysisResponse, ErrorResponse]:
    """Analyze uploaded document payload and return strict JSON response."""

    del api_key, file_size
    if pipeline_module.pipeline_instance is None:
        return ErrorResponse(status="error", message="Pipeline not initialized")
    return await pipeline_module.pipeline_instance.process(request)
