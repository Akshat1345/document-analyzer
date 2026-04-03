"""FastAPI dependencies for authentication and request guards."""

import base64
import logging
import secrets
from typing import Optional

from fastapi import Header, HTTPException

from app.config import settings

logger = logging.getLogger(__name__)


async def validate_api_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    """Validate API key header using timing-safe comparison."""

    if not x_api_key or not secrets.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


async def validate_file_size(x_file_size: Optional[str] = Header(default=None)) -> int:
    """Validate file size header (optional, for early rejection of large files)."""

    try:
        if x_file_size:
            size_bytes = int(x_file_size)
            max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
            if size_bytes > max_bytes:
                logger.warning(
                    "File size %d bytes exceeds limit of %d MB", size_bytes, settings.MAX_FILE_SIZE_MB
                )
                raise HTTPException(
                    status_code=413, detail=f"File size exceeds {settings.MAX_FILE_SIZE_MB}MB limit"
                )
            return size_bytes
    except ValueError:
        logger.warning("Invalid x-file-size header value: %s", x_file_size)
    return 0
