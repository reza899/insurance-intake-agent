from typing import Optional

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)

from config.settings import settings
from src.utils.exceptions import DatabaseError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MongoDB:
    """MongoDB connection manager."""

    def __init__(self) -> None:
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None

    async def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            logger.info(f"Connecting to MongoDB at {settings.mongodb_url}")
            self.client = AsyncIOMotorClient(settings.mongodb_url)
            self.database = self.client[settings.mongodb_database]

            # Test connection
            await self.client.admin.command("ping")
            logger.info("Successfully connected to MongoDB")

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise DatabaseError(f"MongoDB connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self.client is not None:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    async def health_check(self) -> bool:
        """Check MongoDB connection health."""
        try:
            if self.client is None:
                return False
            await self.client.admin.command("ping")
            return True
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return False

    def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """Get a collection from the database."""
        if self.database is None:
            raise DatabaseError("Database not connected")
        return self.database[collection_name]


# Global MongoDB instance
mongodb = MongoDB()
