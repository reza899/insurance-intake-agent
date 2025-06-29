import json
from typing import Any, Optional

import redis.asyncio as redis

from config.settings import settings
from src.core.exceptions import DatabaseError
from src.core.logging import get_logger

logger = get_logger(__name__)


class RedisDB:
    """Redis connection manager."""

    def __init__(self) -> None:
        self.client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            logger.info(f"Connecting to Redis at {settings.redis_url}")
            self.client = redis.from_url(settings.redis_url, decode_responses=True)

            # Test connection
            await self.client.ping()
            logger.info("Successfully connected to Redis")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise DatabaseError(f"Redis connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.client is not None:
            await self.client.close()
            logger.info("Disconnected from Redis")

    async def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            if self.client is None:
                return False
            await self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a key-value pair in Redis."""
        if self.client is None:
            raise DatabaseError("Redis not connected")

        try:
            serialized_value = (
                json.dumps(value) if not isinstance(value, str) else value
            )
            await self.client.set(key, serialized_value, ex=ttl or settings.redis_ttl)
        except Exception as e:
            logger.error(f"Failed to set key {key}: {e}")
            raise DatabaseError(f"Redis set operation failed: {e}")

    async def get(self, key: str) -> Optional[Any]:
        """Get a value by key from Redis."""
        if self.client is None:
            raise DatabaseError("Redis not connected")

        try:
            value = await self.client.get(key)
            if value is None:
                return None

            # Try to deserialize JSON, fallback to string
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        except Exception as e:
            logger.error(f"Failed to get key {key}: {e}")
            raise DatabaseError(f"Redis get operation failed: {e}")

    async def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        if self.client is None:
            raise DatabaseError("Redis not connected")

        try:
            result = await self.client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            raise DatabaseError(f"Redis delete operation failed: {e}")

    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        if self.client is None:
            raise DatabaseError("Redis not connected")

        try:
            result = await self.client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to check existence of key {key}: {e}")
            raise DatabaseError(f"Redis exists operation failed: {e}")


# Global Redis instance
redis_db = RedisDB()
