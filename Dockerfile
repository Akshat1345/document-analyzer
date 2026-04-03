FROM python:3.11-slim

# Reduce image size by disabling pip cache, torch cache, and huggingface cache during build
ENV PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/tmp/torch \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    HUGGINGFACE_HUB_CACHE=/tmp/hf

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
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache /tmp/torch /tmp/hf ~/.cache/pip && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

RUN python -m spacy download en_core_web_sm && \
    rm -rf /root/.cache /tmp/torch /tmp/hf ~/.cache/pip && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

COPY . .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
