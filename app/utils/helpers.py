"""Low-level helper utilities for decoding, file detection, and hashing."""

import base64
import hashlib


def decode_base64(encoded: str) -> bytes:
    """Decode base64 bytes with safe padding and strict validation."""

    try:
        normalized = "".join(encoded.split())
        padding = (-len(normalized)) % 4
        padded = normalized + ("=" * padding)
        return base64.b64decode(padded, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 input provided") from exc


def detect_file_type(filename: str, content: bytes) -> str:
    """Detect standardized file type from filename extension."""

    del content
    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        return "pdf"
    if lowered.endswith(".docx"):
        return "docx"
    if lowered.endswith((".jpg", ".jpeg", ".png")):
        return "image"
    return "image"


def compute_hash(content: bytes) -> str:
    """Return SHA-256 hex digest for content fingerprinting."""

    return hashlib.sha256(content).hexdigest()
