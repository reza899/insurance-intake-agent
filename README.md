# Insurance Intake Agent

A conversational AI system for collecting car insurance information through natural language interactions.

## Overview

This application implements a conversational agent that collects required car insurance data through chat interactions. Users can provide their information naturally, and the system extracts structured data, validates inputs, and detects potential duplicate registrations.

**Architecture**: Web UI → REST API → Agent Orchestrator → LLM Router → Data Processing → Database Storage

## Key Features

- 🤖 **Conversational Interface**: Natural language data collection using AI
- 🧠 **LLM Integration**: LiteLLM-powered with multi-provider support
- 🔍 **Smart Data Extraction**: Automatically extract required insurance fields
- 📊 **Duplicate Detection**: Fuzzy matching to identify existing customers
- 🌐 **REST API**: FastAPI-based backend with automatic documentation
- 💬 **Web UI**: Gradio-powered conversational interface
- 🗄️ **Database Integration**: MongoDB with health monitoring
- ⚡ **Async Architecture**: High-performance async/await throughout
- 🔧 **Production Ready**: Professional logging, configuration, and error handling

## Technology Stack

- **Backend**: FastAPI + Python 3.12+
- **Frontend**: Gradio
- **Database**: MongoDB
- **AI/LLM**: LiteLLM with multi-provider support
- **Package Management**: Poetry

## Prerequisites

- **Python 3.12+** - [Download](https://www.python.org/downloads/)
- **Poetry** - [Install](https://python-poetry.org/docs/#installation)
- **Make** - Pre-installed on macOS/Linux, for Windows: [Install](https://gnuwin32.sourceforge.net/packages/make.htm)
- **Docker** (optional) - [Install](https://docs.docker.com/get-docker/)

## Quick Start

**Option 1: Docker (Recommended)**
```bash
cp .env.example .env  # Configure environment
# Edit .env with your OpenAI API key
make docker-up        # Full containerized setup (takes 5+ minutes first time)
```

**Option 2: Local Development**
1. **Install dependencies**
   ```bash
   make install  # or: poetry install
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start the application**
   ```bash
   # Using Make commands (recommended)
   make run-api    # Start API (Terminal 1)
   make run-ui     # Start UI (Terminal 2)
   
   # Or manually
   poetry run uvicorn src.api.main:app --reload
   poetry run python src/ui/gradio_app.py
   ```

**Access the application:**
- Web UI: http://localhost:8501
- API Docs (Swagger): http://localhost:8000/docs

## Usage Example

**Option 1: Gradio Web UI**
- Open http://localhost:8501
- Type: "I need car insurance"
- Follow the conversation flow naturally

**Option 2: API (Postman/curl)**
```bash
# Start conversation
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "I need car insurance"}'

# Response
{
  "response": "I'd be happy to help you with car insurance! Let's start with your vehicle. What make and model of car do you drive?",
  "extracted_data": {},
  "missing_fields": ["car_type", "manufacturer", "year", "license_plate", "customer_name", "birth_date"],
  "status": "processing"
}
```

**Flow**: User provides info → Agent extracts data → Asks for missing fields → Detects duplicates → Completes registration.

## Configuration

**LLM Configuration:**
```bash
LLM_PRIMARY_MODEL=ollama/gemma3:4b
LLM_FALLBACK_MODELS=gpt-4o-mini,huggingface/google/gemma-3-4b-it
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096
```

## Testing

Run the test suite:
```bash
make test
```
