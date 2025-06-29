.PHONY: help install clean lint format run-api run-ui docker-up docker-down

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	poetry install

clean: ## Clean cache files
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

lint: ## Run code quality checks
	poetry run ruff check src/
	poetry run mypy src/

format: ## Format code
	poetry run ruff format src/

run-api: ## Start API server
	poetry run uvicorn src.api.main:app --reload

run-ui: ## Start UI server
	poetry run python src/ui/gradio_app.py

docker-up: ## Start with Docker
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down