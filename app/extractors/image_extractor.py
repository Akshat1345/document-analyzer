"""
Image text extractor using Tesseract OCR with intelligent
preprocessing for varied image quality and document types.
"""

from PIL import Image, ImageEnhance
import pytesseract
from io import BytesIO
from app.extractors.base import BaseExtractor
from app.utils.text_cleaner import clean_text
import logging

logger = logging.getLogger(__name__)


class ImageExtractor(BaseExtractor):
    """
    Extracts text from images using Tesseract OCR.
    Applies contrast and sharpness enhancement before OCR.
    Tries multiple page segmentation modes for best result.
    """

    def _preprocess(self, img: Image.Image) -> Image.Image:
        """
        Enhance image quality before OCR.
        Handles small, low-contrast, and slightly blurry images.
        """
        img = img.convert("RGB")
        w, h = img.size

        # Upscale small images - Tesseract performs poorly below 300px
        if w < 300 or h < 300:
            scale = max(300 / w, 300 / h, 2.0)
            img = img.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS
            )

        # Boost contrast for faded or scan-quality images
        img = ImageEnhance.Contrast(img).enhance(1.5)

        # Boost sharpness for slightly blurry images
        img = ImageEnhance.Sharpness(img).enhance(1.5)

        return img

    def extract(self, content: bytes) -> tuple[str, dict]:
        """
        Extract text from image bytes.
        Tries PSM 6 -> PSM 3 -> PSM 11 for best result.
        """
        try:
            img = Image.open(BytesIO(content))
            img = self._preprocess(img)

            # PSM 6: uniform text block - best for clean documents
            text = pytesseract.image_to_string(
                img, config="--oem 3 --psm 6"
            )

            # PSM 3: fully automatic - fallback for mixed layouts
            if len(text.strip()) < 20:
                text = pytesseract.image_to_string(
                    img, config="--oem 3 --psm 3"
                )

            # PSM 11: sparse text - last resort
            if len(text.strip()) < 20:
                text = pytesseract.image_to_string(
                    img, config="--oem 3 --psm 11"
                )

            text = clean_text(text)
            word_count = len(text.split())
            logger.info(
                "Image extraction complete: %d words via Tesseract",
                word_count
            )

            return text, {
                "word_count": word_count,
                "extraction_method": "tesseract"
            }

        except Exception as e:
            logger.error("Image extraction failed: %s", e)
            return "", {
                "word_count": 0,
                "extraction_method": "failed"
            }
