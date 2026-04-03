# Data Extraction API

DocuMind AI is a production-ready document intelligence platform for extracting structured insights from PDF, DOCX, and image files.

## Description

Intelligent document processing API that accepts PDF, DOCX, and image files, extracts text, identifies named entities, classifies sentiment, and generates AI-powered summaries using a multi-model pipeline.

## Hackathon Compliance Matrix

| Requirement                                                                | Status      | Evidence                                  |
| -------------------------------------------------------------------------- | ----------- | ----------------------------------------- |
| Endpoint `POST /api/document-analyze`                                      | Implemented | API Contracts section                     |
| Auth header `x-api-key` with 401 on invalid/missing                        | Implemented | API Contracts + runtime behavior          |
| Request `fileName`, `fileType`, `fileBase64`                               | Implemented | API Contracts section                     |
| Response includes `status`, `fileName`, `summary`, `entities`, `sentiment` | Implemented | Success response (official spec shape)    |
| Sentiment limited to Positive/Neutral/Negative                             | Implemented | Core Capabilities + API contract          |
| README includes Description, Tech Stack, Setup Instructions, Approach      | Implemented | Required sections present                 |
| AI tools disclosure present                                                | Implemented | AI Tools Used section                     |
| No hardcoded response mapping                                              | Implemented | Dynamic extraction/summarization pipeline |

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

Quick health check:

```bash
curl http://localhost:8000/health
```

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

## System Diagram

```mermaid
flowchart TD
  A[Client Upload] --> B[POST /api/document-analyze]
  B --> C[Auth Validation<br/>x-api-key]
  C --> D[Base64 Decode]

  subgraph X[Extraction Layer]
    E{File Type}
    F[PDF Extractor<br/>PyMuPDF + pdfplumber]
    G[DOCX Extractor<br/>python-docx]
    H[Image Extractor<br/>EasyOCR + Tesseract]
    E -->|pdf| F
    E -->|docx| G
    E -->|image| H
  end

  D --> E
  F --> I[Cleaned Text]
  G --> I
  H --> I

  I --> J[Document Classifier]

  subgraph Y[Analysis Layer]
    K[NER Pipeline<br/>spaCy + GLiNER + Llama]
    L[Sentiment Ensemble<br/>FinBERT + RoBERTa + VADER]
    M[Entity Fusion + Filters]
    N[Summary Generator<br/>Chain-of-Density]
  end

  J --> K
  J --> L
  K --> M
  M --> N
  L --> O[Sentiment Label]

  N --> P[Response Builder]
  O --> P
  P --> Q[JSON Response]

  subgraph Z[RAG Indexing Layer]
    R[Chunking + Embeddings]
    S[Document-Scoped Vector Store]
  end

  I --> R
  R --> S
```

## End-to-End Flowchart

```mermaid
flowchart TD
  S[Start] --> U[Receive request]
  U --> V{API key valid?}
  V -->|No| W[Return 401]
  V -->|Yes| X{Base64 valid?}
  X -->|No| Y[Return error JSON]
  X -->|Yes| Z[Extract text by file type]
  Z --> AA{Text sufficient?}
  AA -->|No| AB[Return extraction error]
  AA -->|Yes| AC[Run NER and sentiment in parallel]
  AC --> AD[Generate summary]
  AD --> AE[Build success JSON]
  AE --> AF[Index document for RAG]
  AF --> AG[Return response]
```

## Core Capabilities

- Multi-format ingestion: PDF, DOCX, JPG, JPEG, PNG
- Entity extraction: names, dates, organizations, amounts, emails, phone numbers
- Sentiment output strictly from: Positive, Neutral, Negative
- Adaptive factual summarization
- Document-scoped RAG Q&A with strict per-document retrieval isolation
- Redis-backed caching and async worker support

## What Judges Should Evaluate First

1. API authenticity: send different files and observe non-hardcoded, content-dependent outputs
2. OCR resilience: test scanned/visual documents and compare extracted entities
3. RAG isolation: ask document-specific questions and verify no cross-document contamination
4. Error handling: test invalid API keys, invalid base64, and low-text files
5. Operational readiness: verify Docker startup, health endpoint, and frontend integration

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

### Current Implementation Response (Superset)

To support production workflows, the implementation may include extra fields beyond the minimum spec shape, while still preserving all mandatory spec fields.

```json
{
  "status": "success",
  "fileName": "sample1.pdf",
  "documentId": "<sha256-document-id>",
  "summary": "...",
  "entities": {
    "names": ["Ravi Kumar"],
    "dates": ["10 March 2026"],
    "organizations": ["ABC Pvt Ltd"],
    "amounts": ["₹10,000"],
    "emails": ["ravi@example.com"],
    "phones": ["+919876543210"]
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

Request body:

```json
{
  "documentId": "<documentId-from-analyze-response>",
  "question": "What is the current role?",
  "topK": 4
}
```

Response:

```json
{
  "status": "success",
  "documentId": "<sha256-document-id>",
  "question": "What is the current role?",
  "answer": "Senior Project Manager",
  "citations": ["Chunk 1", "Chunk 2", "Chunk 4", "Chunk 3"]
}
```

## Implemented Extensions Beyond Minimum Spec

The official hackathon contract is fully honored, and the project also implements the following production-grade enhancements:

- Analyze response enrichment for application workflow:
  - `documentId` for deterministic per-document indexing and retrieval
  - additional entity buckets: `emails` and `phones`
- Document-scoped RAG system:
  - automatic embedding generation after analysis
  - chunk-level retrieval restricted to one `documentId` namespace
  - citation list in QA response for transparency
- Frontend dashboard improvements:
  - drag-and-drop upload interface
  - robust retry/error UX
  - copy-JSON action
  - integrated per-document Q&A panel
- Extraction quality hardening:
  - OCR artifact filtering for noisy entity candidates
  - exact dedup for sensitive identifiers like email and phone
  - broader phone format handling (international and spaced formats)
- Reliability and ops hardening:
  - Redis cache versioning and graceful fallback
  - startup validation for required secrets
  - CORS preflight handling for local and production
  - Docker healthcheck and preloaded model startup

## RAG Isolation Guarantee

- Each analyzed document receives a unique `documentId`
- Embeddings are stored per document namespace
- Retrieval for Q&A is restricted to the requested `documentId` only
- No global retrieval across other documents is used

RAG Q&A request/answer sequence:

```mermaid
sequenceDiagram
  participant C as Client
  participant API as FastAPI
  participant RS as RAG Service
  participant VS as Vector Store
  participant LLM as Llama 3.3 (Groq)

  C->>API: POST /api/document-qa {documentId, question}
  API->>RS: answer_question(documentId, question)
  RS->>VS: load chunks/vectors for documentId only
  VS-->>RS: top-k matching chunks
  RS->>LLM: grounded prompt (question + chunks)
  LLM-->>RS: answer
  RS-->>API: answer + citations
  API-->>C: JSON response
```

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

Production verification checklist:

```bash
# 1) Backend health
curl https://your-api-domain.com/health

# 2) Auth enforcement (should be 401)
curl -X POST https://your-api-domain.com/api/document-analyze \
  -H "Content-Type: application/json" \
  -d '{"fileName":"x.pdf","fileType":"pdf","fileBase64":"eA=="}'

# 3) Frontend build check
cd frontend && npm run build
```

## Final Submission Checklist

- [ ] Public GitHub repo URL added to submission form
- [ ] Live deployment URL updated in README
- [ ] `README.md`, `requirements.txt`, `.env.example`, and backend source are present
- [ ] `.env` is not tracked by git
- [ ] API key validation returns 401 for invalid/missing key
- [ ] AI tools disclosure section is present
- [ ] At least one successful API demo call is validated after deployment

## AI Tools Used

(Mandatory disclosure — required by hackathon rules)

| Tool               | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| GitHub Copilot Pro | Code generation and implementation                       |
| Claude (claude.ai) | Architecture design, model selection, technical strategy |

All AI-generated code was reviewed, tested, and validated.
Architecture decisions and system design were human-directed.

## Known Limitations

- Very large documents (>10,000 words) may have slower response
- Handwritten text in images may have reduced OCR accuracy
- Non-English documents not currently supported

## License

MIT
