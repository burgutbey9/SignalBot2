# SignalBot Makefile
# Qulay komandalar uchun

.PHONY: help install test run clean setup lint format

# Default target
help:
	@echo "SignalBot - Professional Crypto Trading Bot"
	@echo ""
	@echo "Mavjud komandalar:"
	@echo "  make install    - Barcha dependencies o'rnatish"
	@echo "  make setup      - Muhit sozlash (venv, deps, folders)"
	@echo "  make test       - Testlarni ishga tushirish"
	@echo "  make run        - Botni signal rejimda ishga tushirish"
	@echo "  make run-paper  - Botni paper trading rejimda ishga tushirish"
	@echo "  make lint       - Kodni tekshirish"
	@echo "  make format     - Kodni formatlash"
	@echo "  make clean      - Cache va temp fayllarni tozalash"

# Virtual environment yaratish
venv:
	python3 -m venv venv
	@echo "Virtual environment yaratildi. Aktivlashtirish: source venv/bin/activate"

# Dependencies o'rnatish
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# To'liq setup
setup: venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	mkdir -p logs data/cache data/models data/backtest
	@echo "Setup tugadi! .env faylini sozlashni unutmang."

# Testlar
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Bot ishga tushirish
run:
	python src/main.py --mode signal

run-paper:
	python src/main.py --mode paper

run-live:
	@echo "OGOHLANTIRISH: Live trading rejimi!"
	@echo "Ishonchingiz komilmi? (yes/no)"
	@read confirm && [ "$$confirm" = "yes" ] && python src/main.py --mode live

# Kod sifati
lint:
	flake8 src/ tests/ --max-line-length=120
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

# Tozalash
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf data/cache/*
	@echo "Tozalash tugadi!"

# Database
db-init:
	python -m alembic init alembic

db-migrate:
	python -m alembic revision --autogenerate -m "$(msg)"

db-upgrade:
	python -m alembic upgrade head

db-downgrade:
	python -m alembic downgrade -1

# Docker
docker-build:
	docker build -t signalbot:latest .

docker-run:
	docker run -d --name signalbot --env-file .env signalbot:latest

docker-stop:
	docker stop signalbot
	docker rm signalbot

# Monitoring
logs:
	tail -f logs/bot.log

logs-error:
	tail -f logs/error.log

# Development
dev:
	python src/main.py --mode signal --debug
