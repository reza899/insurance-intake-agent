from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from config.settings import settings
from src.agent.orchestrator import InsuranceAgent
from src.database.mongodb import mongodb
from src.models.api import ChatRequest, ChatResponse
from src.models.insurance import RegistrationResponse
from src.utils.exceptions import (
    ConfigurationError,
    ConversationError,
    DataExtractionError,
    DuplicateDetectionError,
    RegistrationError,
    RegistrationNotFoundError,
)
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events."""
    # Startup
    setup_logging()
    logger.info("Starting Insurance Intake Agent API")

    try:
        await mongodb.connect()
        logger.info("Database connection established successfully")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        logger.warning("Running without database connection - some features will be unavailable")

    yield

    # Shutdown
    try:
        await mongodb.disconnect()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app_config = settings._app_config.get("application", {})
    app = FastAPI(
        title=f"{app_config['name']} API",
        description=app_config["description"],
        version=app_config["version"],
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        """Basic health check endpoint."""
        return JSONResponse({"status": "healthy", "service": "Insurance Intake Agent API"})

    @app.get("/health/db")
    async def health_db() -> JSONResponse:
        """Database health check endpoint."""
        try:
            mongodb_health = await mongodb.health_check()

            return JSONResponse(
                {
                    "status": "healthy" if mongodb_health else "degraded",
                    "mongodb": "connected" if mongodb_health else "disconnected",
                }
            )
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            raise HTTPException(status_code=500, detail={"status": "unhealthy", "error": str(e)})

    # Initialize agent
    agent = InsuranceAgent()

    # API routes
    @app.get("/registrations/{registration_id}", response_model=RegistrationResponse)
    async def get_registration(registration_id: str) -> RegistrationResponse:
        """Get registration by ID."""
        try:
            registration = await agent.get_registration(registration_id)
            if not registration:
                raise HTTPException(status_code=404, detail="Registration not found")
            return registration
        except RegistrationNotFoundError:
            raise HTTPException(status_code=404, detail="Registration not found")
        except RegistrationError as e:
            logger.error(f"Registration error for {registration_id}: {e}")
            raise HTTPException(status_code=500, detail="Registration service error")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting registration {registration_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/chat", response_model=ChatResponse)
    async def chat_with_agent(request: ChatRequest) -> ChatResponse:
        """Chat with the insurance agent."""
        try:
            result = await agent.process_message(request.message, request.conversation_history)

            return ChatResponse(**result)

        except DataExtractionError as e:
            logger.error(f"Data extraction error: {e}")
            raise HTTPException(status_code=422, detail="Failed to extract data from message")
        except DuplicateDetectionError as e:
            logger.error(f"Duplicate detection error: {e}")
            raise HTTPException(status_code=500, detail="Duplicate detection service error")
        except ConversationError as e:
            logger.error(f"Conversation error: {e}")
            raise HTTPException(status_code=400, detail="Conversation flow error")
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            raise HTTPException(status_code=500, detail="Service configuration error")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
