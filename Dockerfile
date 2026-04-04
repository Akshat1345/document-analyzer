# -- Base image ----------------------------------------------------
FROM python:3.11-slim

# -- System dependencies -------------------------------------------
# tesseract-ocr: OCR engine for image text extraction
# libgl1: required by OpenCV / pdfplumber
# libglib2.0-0: required by OpenCV
# libmagic1: file type detection
# ghostscript: PDF rendering support for camelot/pdfplumber
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# -- Python dependencies -------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -- spaCy model ---------------------------------------------------
# en_core_web_sm: ~50MB, no PyTorch dependency
# Downloaded at build time -> zero startup penalty
RUN python -m pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# -- Pre-flight check ----------------------------------------------
# Verify all critical imports work before the image ships.
# If any import fails here, the build fails loudly - not silently
# at runtime when a judge is testing your API.
RUN python -c "import spacy; import pytesseract; from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer; from groq import Groq; import fitz; import docx; import pdfplumber; from PIL import Image; nlp = spacy.load('en_core_web_sm'); vader = SentimentIntensityAnalyzer(); print('SUCCESS: all critical imports verified')"

# -- Application code ----------------------------------------------
COPY . .

# -- Server --------------------------------------------------------
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
