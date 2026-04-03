#!/bin/bash
set -e

echo "Initializing git repository"
git init
git branch -M main

echo "GROUP 1 - chore: initialize project with dependencies and config"
git add .gitignore requirements.txt .env.example
git commit -m "chore: initialize project with dependencies and config"

echo "GROUP 2 - feat: add Pydantic schemas, settings, and API key auth"
git add app/__init__.py app/config.py app/models/__init__.py app/models/schemas.py app/dependencies.py
git commit -m "feat: add Pydantic schemas, settings, and API key auth"

echo "GROUP 3 - feat: add text cleaning pipeline and file utility helpers"
git add app/utils/__init__.py app/utils/text_cleaner.py app/utils/helpers.py
git commit -m "feat: add text cleaning pipeline and file utility helpers"

echo "GROUP 4 - feat: add PDF extractor with scanned page OCR fallback"
git add app/extractors/__init__.py app/extractors/base.py app/extractors/pdf_extractor.py
git commit -m "feat: add PDF extractor with scanned page OCR fallback"

echo "GROUP 5 - feat: add DOCX extractor with heading and table support"
git add app/extractors/docx_extractor.py
git commit -m "feat: add DOCX extractor with heading and table support"

echo "GROUP 6 - feat: add image extractor with EasyOCR and preprocessing"
git add app/extractors/image_extractor.py
git commit -m "feat: add image extractor with EasyOCR and preprocessing"

echo "GROUP 7 - feat: add document type classifier and entity normalizer"
git add app/processors/__init__.py app/processors/document_classifier.py app/processors/entity_normalizer.py
git commit -m "feat: add document type classifier and entity normalizer"

echo "GROUP 8 - feat: add Groq client with Llama 3.3 70B and Ollama fallback"
git add app/services/__init__.py app/services/groq_client.py
git commit -m "feat: add Groq client with Llama 3.3 70B and Ollama fallback"

echo "GROUP 9 - feat: implement triple-layer NER fusion (spaCy+GLiNER+Llama)"
git add app/processors/ner_engine.py
git commit -m "feat: implement triple-layer NER fusion (spaCy+GLiNER+Llama)"

echo "GROUP 10 - feat: add document-adaptive sentiment ensemble"
git add app/processors/sentiment_engine.py
git commit -m "feat: add document-adaptive sentiment ensemble"

echo "GROUP 11 - feat: implement Chain-of-Density summarization"
git add app/processors/summarizer.py
git commit -m "feat: implement Chain-of-Density summarization"

echo "GROUP 12 - feat: add async processing pipeline with Redis caching"
git add app/services/cache.py app/services/pipeline.py app/services/rag_service.py workers/__init__.py workers/celery_app.py
git commit -m "feat: add async processing pipeline with Redis caching"

echo "GROUP 13 - feat: expose /api/document-analyze with model preloading"
git add app/routers/__init__.py app/routers/analyze.py app/routers/qa.py app/main.py
git commit -m "feat: expose /api/document-analyze with model preloading"

echo "GROUP 14 - test: add unit tests and rubric-based evaluation harness"
git add tests/__init__.py tests/conftest.py tests/test_extractors.py tests/test_sentiment.py tests/test_entities.py tests/test_api.py eval/scorer.py eval/ground_truth.json run_sample_test.py inspect_entities.py
git commit -m "test: add unit tests and rubric-based evaluation harness"

echo "GROUP 15 - chore: add Dockerfile with pre-loaded models and compose"
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "chore: add Dockerfile with pre-loaded models and compose"

echo "GROUP 16 - feat: add Next.js dashboard with drag-drop and results UI"
git add frontend/
git commit -m "feat: add Next.js dashboard with drag-drop and results UI"

echo "GROUP 17 - docs: add README with setup guide and AI tools disclosure"
git add README.md setup_git.sh
git commit -m "docs: add README with setup guide and AI tools disclosure"

echo "---"
echo "Now go to github.com and create an EMPTY PUBLIC repo"
echo "Name: document-analyzer"
echo "NO readme, NO gitignore, NO license"
echo "Then run:"
echo "git remote add origin https://github.com/YOURUSERNAME/document-analyzer.git"
echo "git push -u origin main"
