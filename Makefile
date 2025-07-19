# SignalBot Makefile

.PHONY: help install dev test run clean lint format

# Default target
help:
	@echo "SignalBot - Professional Crypto Trading Bot"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make dev        - Install dev dependencies"
	@echo "  make test       - Run tests"
	@echo "  make run        - Run the bot"
	@echo "  make clean      - Clean cache files"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"

# Install production dependencies
install:
	pip install -r requirements.txt
	cp .env.example .env
	@echo "✅ Dependencies installed. Edit .env file with your API keys!"

# Install development dependencies
dev: install
	pip install -r requirements-dev.txt
	pre-commit install

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html

# Run the bot
run:
	python src/main.py

# Run in signal-only mode
run-signal:
	python src/main.py --mode signal

# Run in paper trading mode
run-paper:
	python src/main.py --mode paper

# Clean cache and temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Run linters
lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Docker commands
docker-build:
	docker build -t signalbot:latest .

docker-run:
	docker run -d --name signalbot --env-file .env signalbot:latest

docker-stop:
	docker stop signalbot
	docker rm signalbot

# Database commands
db-init:
	alembic init alembic
	alembic revision --autogenerate -m "Initial migration"
	alembic upgrade head

db-migrate:
	alembic revision --autogenerate -m "$(message)"
	alembic upgrade head

db-rollback:
	alembic downgrade -1
