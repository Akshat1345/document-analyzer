"""Image extraction pipeline using EasyOCR primary and pytesseract fallback."""

import logging
from io import BytesIO
from typing import Any

import numpy as np
import pytesseract
from PIL import Image, ImageEnhance

from app.extractors.base import BaseExtractor
from app.utils.text_cleaner import clean_text

logger = logging.getLogger(__name__)


class ImageExtractor(BaseExtractor):
    """Extract text from image bytes with lightweight preprocessing."""

    _reader = None

    @classmethod
    def _get_reader(cls):
        """Initialize and cache EasyOCR reader for English OCR."""

        if cls._reader is None:
            import easyocr

            cls._reader = easyocr.Reader(["en"], gpu=False)
        return cls._reader

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Prepare image for OCR by normalizing mode, scale, and contrast."""

        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        if width < 300 or height < 300:
            img = img.resize((width * 2, height * 2), Image.Resampling.LANCZOS)

        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.3)

    def extract(self, content: bytes) -> tuple[str, dict]:
        """Extract OCR text and metadata from image bytes with fallback behavior."""

        text = ""
        avg_confidence = 0.0
        extraction_method = "easyocr"

        try:
            img = Image.open(BytesIO(content))
            img = self._preprocess_image(img)

            reader = self._get_reader()
            result: Any = reader.readtext(np.array(img), detail=1)
            if result:
                text = " ".join(str(entry[1]) for entry in result)
                avg_confidence = sum(float(entry[2]) for entry in result) / len(result)

            if avg_confidence < 0.6 or len(text.strip()) < 20:
                extraction_method = "tesseract"
                text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
        except Exception as exc:
            logger.warning("Image extraction failed: %s", exc)

        text = clean_text(text)
        metadata = {
            "word_count": self.get_word_count(text),
            "extraction_method": extraction_method,
            "avg_confidence": float(avg_confidence),
        }
        return text, metadata
