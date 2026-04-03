"""Text cleanup utilities to normalize extraction artifacts."""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Apply ordered cleanup transformations to extracted document text."""

    if not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Rejoin words split across lines by OCR/PDF wrapping.
    cleaned = re.sub(r"(\w+)-\n(\w+)", r"\1\2", cleaned)
    cleaned = re.sub(r"(?<=[a-z])\n(?=[a-z])", " ", cleaned, flags=re.IGNORECASE)

    # Normalize spacing while preserving paragraph boundaries.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Drop non-printable control chars except tabs/newlines.
    cleaned = "".join(ch for ch in cleaned if ch == "\n" or ch == "\t" or ch.isprintable())
    return cleaned.strip()
