# Data Extraction API

DocuMind AI is a production-ready document intelligence platform for extracting structured insights from PDF, DOCX, and image files.

## Description

Intelligent document processing API that accepts PDF, DOCX, and image files, extracts text, identifies named entities, classifies sentiment, and generates AI-powered summaries using a multi-model pipeline.

## Live Demo

- Frontend URL: https://your-frontend-domain.com
- API URL: https://your-api-domain.com

## Tech Stack

- Framework: FastAPI (Python 3.11)
- NER: spaCy en_core_web_trf + GLiNER (zero-shot SOTA) + Llama 3.3 70B
- Summarization: Llama 3.3 70B via Groq API (Chain-of-Density technique)
- Sentiment: FinBERT + cardiffnlp/RoBERTa + VADER (weighted ensemble)
- OCR: EasyOCR (primary) + Tesseract (fallback)
- PDF: PyMuPDF + pdfplumber
- DOCX: python-docx
- Cache: Redis
- Queue: Celery
- Deploy: Docker + Railway

## Setup Instructions

1. Clone the repository
2. `cp .env.example .env` and fill in `GROQ_API_KEY` and `API_KEY`
3. Get free Groq API key at https://console.groq.com
4. `docker compose up --build`
5. API ready at http://localhost:8000/api/document-analyze

## Approach

- Text extraction strategy by file format:
  - PDF: PyMuPDF parsing with pdfplumber fallback, plus OCR path for scanned pages
  - DOCX: paragraph, heading, and table-aware extraction
  - Image: EasyOCR primary pipeline with Tesseract fallback
- Triple-layer NER strategy:
  - spaCy detects standard entities with transformer context
  - GLiNER catches domain-specific and zero-shot entities
  - Llama 3.3 70B captures context-dependent entities
  - All three are fused with fuzzy deduplication and false-positive filtering
- Sentiment strategy:
  - Document category is classified first
  - Weighted ensemble of FinBERT + RoBERTa + VADER is applied
  - Weights are adjusted based on document type (for example, finance-heavy documents prioritize FinBERT)
- Summary strategy:
  - Chain-of-Density style prompting with entity anchoring
  - Prompts are constrained for factual grounding and hallucination prevention

## Core Capabilities

- Multi-format ingestion: PDF, DOCX, JPG, JPEG, PNG
- Entity extraction: names, dates, organizations, amounts, emails, phone numbers
- Sentiment output strictly from: Positive, Neutral, Negative
- Adaptive factual summarization
- Document-scoped RAG Q&A with strict per-document retrieval isolation
- Redis-backed caching and async worker support

## API Contracts

### Endpoint

- Method: POST
- Path: `/api/document-analyze`
- Header: `x-api-key: YOUR_SECRET_API_KEY`

Missing or invalid API key returns 401.

### Request Body

```json
{
  "fileName": "sample1.pdf",
  "fileType": "pdf",
  "fileBase64": "<base64>"
}
```

`fileType` values: `pdf | docx | image`

### Success Response (Official Spec Shape)

```json
{
  "status": "success",
  "fileName": "sample1.pdf",
  "summary": "...",
  "entities": {
    "names": ["Ravi Kumar"],
    "dates": ["10 March 2026"],
    "organizations": ["ABC Pvt Ltd"],
    "amounts": ["₹10,000"]
  },
  "sentiment": "Neutral"
}
```

### Error Response

```json
{
  "status": "error",
  "message": "..."
}
```

### Example Request

```bash
curl -X POST https://your-deployment.railway.app/api/document-analyze \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "fileName": "report.pdf",
    "fileType": "pdf",
    "fileBase64": "<base64_encoded_content>"
  }'
```

### Example Response

```json
{
  "status": "success",
  "fileName": "report.pdf",
  "summary": "...",
  "entities": {
    "names": [],
    "dates": [],
    "organizations": ["Google", "Microsoft"],
    "amounts": []
  },
  "sentiment": "Positive"
}
```

## Additional Endpoint (Project Extension)

- Method: POST
- Path: `/api/document-qa`
- Purpose: Ask questions from a single indexed document context only

## RAG Isolation Guarantee

- Each analyzed document receives a unique `documentId`
- Embeddings are stored per document namespace
- Retrieval for Q&A is restricted to the requested `documentId` only
- No global retrieval across other documents is used

## Project Structure Note

This project uses `app/` instead of the spec's suggested `src/`, following FastAPI best practices with proper package separation.
The entry point is `app/main.py` exposed via uvicorn.

## Project Structure

```text
.
├── app/
│   ├── extractors/
│   ├── processors/
│   ├── routers/
│   ├── services/
│   ├── models/
│   ├── utils/
│   └── main.py
├── frontend/
├── tests/
├── eval/
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

## Environment Variables

### Backend (.env)

- `GROQ_API_KEY` (required)
- `API_KEY` (required)
- `REDIS_URL` (default: `redis://localhost:6379`)
- `ENVIRONMENT` (default: `development`)
- `LOG_LEVEL` (default: `INFO`)
- `USE_CACHE` (default: `true`)
- `USE_LOCAL_LLM` (default: `false`)
- `LOCAL_LLM_URL` (default: `http://localhost:11434`)
- `MAX_FILE_SIZE_MB` (default: `50`)
- `REQUEST_TIMEOUT_SECONDS` (default: `300`)

### Frontend (frontend/.env.local)

- `NEXT_PUBLIC_API_URL`
- `NEXT_PUBLIC_API_KEY`

## Testing and Validation

```bash
python -m py_compile $(find app tests workers -name '*.py')
pytest -q tests/test_api.py tests/test_entities.py tests/test_sentiment.py tests/test_extractors.py
python run_sample_test.py
```

## Deployment Notes

- Deploy backend container and Redis with Docker/compose
- Set required secrets on hosting platform
- Set frontend API URL and API key env vars
- Verify health endpoint: `GET /health`

## AI Tools Used

(Mandatory disclosure — required by hackathon rules)

| Tool | Purpose |
|------|---------|
| GitHub Copilot Pro | Code generation and implementation |
| Claude (claude.ai) | Architecture design, model selection, technical strategy |

All AI-generated code was reviewed, tested, and validated.
Architecture decisions and system design were human-directed.

## Known Limitations

- Very large documents (>10,000 words) may have slower response
- Handwritten text in images may have reduced OCR accuracy
- Non-English documents not currently supported

## License

MIT
