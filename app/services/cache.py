"""Redis cache service with non-fatal behavior."""

import json
import logging
from typing import Optional

from redis import asyncio as redis

from app.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-backed cache wrapper used by analysis pipeline."""

    def __init__(self) -> None:
        """Initialize service without active redis connection."""

        self._redis = None

    async def connect(self) -> None:
        """Connect to Redis and continue without cache on failure."""

        try:
            self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
            await self._redis.ping()
        except Exception as exc:
            logger.warning("Redis unavailable, continuing without cache: %s", exc)
            self._redis = None

    def build_key(self, content_hash: str) -> str:
        """Return versioned cache key from content hash."""

        return f"docuanalyze:v6:{content_hash}"

    async def get(self, key: str) -> Optional[dict]:
        """Fetch JSON payload from cache or return None on misses/errors."""

        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(key)
            if not raw:
                return None
            return json.loads(raw)
        except Exception:
            return None

    async def set(self, key: str, value: dict, ttl: int = 86400) -> None:
        """Write JSON payload to cache and ignore write failures."""

        if self._redis is None:
            return
        try:
            await self._redis.setex(key, ttl, json.dumps(value))
        except Exception as exc:
            logger.warning("Cache set failed for key %s: %s", key, exc)
