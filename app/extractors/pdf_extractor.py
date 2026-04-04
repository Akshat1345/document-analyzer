"""PDF extraction pipeline using PyMuPDF primary and pdfplumber fallback."""

import logging
from io import BytesIO
from typing import Any, cast

import fitz
import pdfplumber
import pytesseract
from PIL import Image

from app.extractors.base import BaseExtractor
from app.utils.text_cleaner import clean_text

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """Extract PDF text with OCR support for scanned pages."""

    def extract(self, content: bytes) -> tuple[str, dict]:
        """Extract text and metadata from PDF bytes with robust fallback strategy."""

        combined_parts: list[str] = []
        has_tables = False
        page_count = 1
        extraction_method = "pymupdf"
        used_ocr = False

        try:
            with fitz.open(stream=content, filetype="pdf") as doc:
                page_count = len(doc) if len(doc) > 0 else 1
                for page in doc:
                    page_obj = cast(Any, page)
                    page_text = page_obj.get_text() or ""
                    if len(page_text.strip()) < 50:
                        try:
                            pix = page_obj.get_pixmap(matrix=fitz.Matrix(2, 2))
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            ocr_text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
                            if ocr_text.strip():
                                page_text = ocr_text
                                used_ocr = True
                        except Exception as ocr_exc:
                            logger.warning("PDF OCR fallback failed for page: %s", ocr_exc)
                    combined_parts.append(page_text)
        except Exception as pymupdf_exc:
            logger.warning("PyMuPDF extraction failed: %s", pymupdf_exc)

        text = clean_text("\n".join(combined_parts))

        if len(text.strip()) < 100:
            try:
                extraction_method = "pdfplumber"
                plumber_parts: list[str] = []
                with pdfplumber.open(BytesIO(content)) as pdf:
                    page_count = len(pdf.pages) if pdf.pages else page_count
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        plumber_parts.append(page_text)
                        tables = page.extract_tables() or []
                        if tables:
                            has_tables = True
                            for table in tables:
                                for row in table:
                                    normalized_row = [cell.strip() if cell else "" for cell in row]
                                    plumber_parts.append(" | ".join(normalized_row))
                plumber_text = clean_text("\n".join(plumber_parts))
                if len(plumber_text.strip()) > len(text.strip()):
                    text = plumber_text
            except Exception as plumber_exc:
                logger.warning("pdfplumber fallback failed: %s", plumber_exc)

        if used_ocr and len(text.strip()) >= 100:
            extraction_method = "ocr"

        metadata = {
            "page_count": page_count,
            "has_tables": has_tables,
            "word_count": self.get_word_count(text),
            "extraction_method": extraction_method,
        }
        return text, metadata
