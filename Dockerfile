FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_trf

RUN python - <<'PY' 2>/dev/null
from transformers import pipeline
from gliner import GLiNER
import easyocr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

pipeline('sentiment-analysis', model='ProsusAI/finbert', return_all_scores=True)
pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', return_all_scores=True)
GLiNER.from_pretrained('urchade/gliner_mediumv2.1')
easyocr.Reader(['en'], gpu=False)
SentimentIntensityAnalyzer()
PY

COPY . .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
