"""FastAPI app bootstrap with lifecycle-managed model initialization."""

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.processors.ner_engine import NEREngine
from app.processors.sentiment_engine import SentimentEngine
from app.processors.summarizer import Summarizer
from app.routers.analyze import router as analyze_router
from app.routers.qa import router as qa_router
from app.services.cache import CacheService
from app.services import pipeline as pipeline_module
from app.services.pipeline import AnalysisPipeline, set_pipeline_instance
from app.services.qa_service import QAService, set_qa_service_instance


def _configure_logging() -> None:
    """Configure standard logging and structlog renderers."""

    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and services at startup and release on shutdown."""

    _configure_logging()
    logger = logging.getLogger(__name__)

    cache = CacheService()
    await cache.connect()

    NEREngine.initialize()
    SentimentEngine.initialize()

    pipeline = AnalysisPipeline(
        ner_engine=NEREngine(),
        sentiment_engine=SentimentEngine(),
        summarizer=Summarizer(),
        cache=cache,
    )
    set_qa_service_instance(QAService(cache=cache))
    if pipeline_module.pipeline_instance is None:
        set_pipeline_instance(pipeline)
    app.state.models_loaded = True

    logger.info("All models initialized. Startup complete.")
    yield
    logger.info("Shutting down")


app = FastAPI(title="DocuMind AI", version="1.0.0", lifespan=lifespan)

# Configure CORS with environment-based origins
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost",
]
if settings.ENVIRONMENT.lower() == "production":
    # In production, allow deployed frontend in addition to localhost for quick ops checks.
    allowed_origins.append("https://your-frontend-domain.com")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(qa_router)


@app.get("/health")
async def health() -> dict[str, object]:
    """Always-on liveness endpoint returning health status."""

    return {"status": "healthy", "models_loaded": True}
