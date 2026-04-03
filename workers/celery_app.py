"""Celery application configuration backed by Redis."""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "document_analyzer",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)
