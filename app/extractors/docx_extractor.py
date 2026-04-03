"""DOCX extraction pipeline including headings and tables."""

import logging
from io import BytesIO

from docx import Document

from app.extractors.base import BaseExtractor
from app.utils.text_cleaner import clean_text

logger = logging.getLogger(__name__)


class DOCXExtractor(BaseExtractor):
    """Extract text content from DOCX paragraphs and tables in document order."""

    def extract(self, content: bytes) -> tuple[str, dict]:
        """Extract DOCX textual content and metadata safely."""

        parts: list[str] = []
        has_tables = False
        paragraph_count = 0

        try:
            document = Document(BytesIO(content))

            for paragraph in document.paragraphs:
                raw = (paragraph.text or "").strip()
                if not raw:
                    continue
                paragraph_count += 1
                style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
                if "heading" in style_name:
                    parts.append(f"## {raw}")
                else:
                    parts.append(raw)

            for table in document.tables:
                has_tables = True
                parts.append("\n[TABLE]")
                for row in table.rows:
                    row_values = [cell.text.strip() for cell in row.cells]
                    parts.append(" | ".join(row_values))
        except Exception as exc:
            logger.warning("DOCX extraction failed: %s", exc)

        text = clean_text("\n".join(parts))
        metadata = {
            "has_tables": has_tables,
            "word_count": self.get_word_count(text),
            "paragraph_count": paragraph_count,
        }
        return text, metadata
