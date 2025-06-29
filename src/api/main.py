from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.core.logging import get_logger, setup_logging
from src.database.mongodb import mongodb
from src.database.redis import redis_db
from src.models.registration import RegistrationRequest, RegistrationResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    setup_logging()
    logger.info("Starting Insurance Intake Agent API")

    try:
        await mongodb.connect()
        await redis_db.connect()
        logger.info("Database connections established successfully")
    except Exception as e:
        logger.error(f"Failed to connect to databases: {e}")
        logger.warning(
            "Running without database connections - some features will be unavailable"
        )

    yield

    # Shutdown
    try:
        await mongodb.disconnect()
        await redis_db.disconnect()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Insurance Intake Agent API",
        description="AI-powered conversational agent for car insurance registration",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        """Basic health check endpoint."""
        return JSONResponse(
            {"status": "healthy", "service": "Insurance Intake Agent API"}
        )

    @app.get("/health/db")
    async def health_db():
        """Database health check endpoint."""
        try:
            mongodb_health = await mongodb.health_check()
            redis_health = await redis_db.health_check()

            return JSONResponse(
                {
                    "status": "healthy"
                    if mongodb_health and redis_health
                    else "degraded",
                    "mongodb": "connected" if mongodb_health else "disconnected",
                    "redis": "connected" if redis_health else "disconnected",
                }
            )
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            raise HTTPException(
                status_code=500, detail={"status": "unhealthy", "error": str(e)}
            )

    # API routes
    @app.post("/api/registrations", response_model=RegistrationResponse)
    async def create_registration(registration: RegistrationRequest):
        """Create a new insurance registration."""
        # TODO: Implement in Phase 2 with actual database operations
        return JSONResponse(
            status_code=501,
            content={"message": "Registration endpoint - Phase 2 implementation"},
        )

    @app.get("/api/registrations/{registration_id}")
    async def get_registration(registration_id: str):
        """Get registration by ID."""
        # TODO: Implement in Phase 2
        return JSONResponse(
            status_code=501,
            content={"message": "Get registration endpoint - Phase 2 implementation"},
        )

    @app.post("/api/conversations")
    async def create_conversation():
        """Create a new conversation session."""
        # TODO: Implement in Phase 2
        return JSONResponse(
            status_code=501,
            content={"message": "Conversation endpoint - Phase 2 implementation"},
        )

    @app.get("/api/conversations/{session_id}")
    async def get_conversation(session_id: str):
        """Get conversation session."""
        # TODO: Implement in Phase 2
        return JSONResponse(
            status_code=501,
            content={"message": "Get conversation endpoint - Phase 2 implementation"},
        )

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
