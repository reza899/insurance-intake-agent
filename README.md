# Insurance Intake Agent

An AI-powered conversational agent for car insurance registration with natural language processing capabilities.

## Overview

This project provides a modern, scalable solution for collecting car insurance information through natural conversation. Built with FastAPI and Gradio, it offers both REST API and web interface access with comprehensive data validation and database integration.

## Key Features

- 🤖 **Conversational Interface**: Natural language data collection
- 🔍 **Smart Validation**: Comprehensive input validation with Pydantic
- 🌐 **REST API**: FastAPI-based backend with automatic documentation
- 💬 **Web UI**: Gradio-powered conversational interface
- 🗄️ **Database Integration**: MongoDB and Redis with health monitoring
- ⚡ **Async Architecture**: High-performance async/await throughout
- 🔧 **Production Ready**: Professional logging, configuration, and error handling

## Technology Stack

- **Backend**: FastAPI + Python 3.12+
- **Frontend**: Gradio
- **Databases**: MongoDB + Redis
- **AI Framework**: LangChain
- **Package Management**: Poetry

## Quick Start

1. **Install dependencies**
   ```bash
   poetry install
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start the application**
   ```bash
   # Start API (Terminal 1)
   poetry run uvicorn src.api.main:app --reload
   
   # Start UI (Terminal 2)
   poetry run python src/ui/gradio_app.py
   ```

4. **Access the application**
   - Web UI: http://localhost:8501
   - API Docs: http://localhost:8000/docs
