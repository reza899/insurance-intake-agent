from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.agent.orchestrator import InsuranceAgent
from src.database.mongodb import mongodb
from src.models.api import ChatRequest, ChatResponse
from src.models.insurance import RegistrationResponse
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    app = FastAPI(
        title="Insurance Intake Agent API",
        description="AI-powered conversational agent for car insurance registration",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        """Basic health check endpoint."""
        return JSONResponse({"status": "healthy", "service": "Insurance Intake Agent API"})

    @app.get("/health/db")
    async def health_db():
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
    async def get_registration(registration_id: str):
        """Get registration by ID."""
        try:
            registration = await InsuranceAgent.get_registration(registration_id)
            if not registration:
                raise HTTPException(status_code=404, detail="Registration not found")
            return registration
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting registration {registration_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/chat", response_model=ChatResponse)
    async def chat_with_agent(request: ChatRequest):
        """Chat with the insurance agent."""
        try:
            # Convert Pydantic models to dicts for agent
            conversation_history = [item.model_dump() for item in request.conversation_history]

            result = await agent.process_message(request.message, conversation_history)

            # Return as ChatResponse model
            return ChatResponse(**result)

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

    from config.settings import settings

    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
