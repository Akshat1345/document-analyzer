"""Extractor unit tests for DOCX and image/pipeline fixtures."""

from app.extractors.docx_extractor import DOCXExtractor


def test_docx_extractor_non_empty(sample_docx_b64):
    """DOCX extractor should parse synthetic fixture with content."""

    from app.utils.helpers import decode_base64

    content = decode_base64(sample_docx_b64)
    text, metadata = DOCXExtractor().extract(content)
    assert len(text.strip()) > 0
    assert metadata["word_count"] > 0
