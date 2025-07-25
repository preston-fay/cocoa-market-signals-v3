# Makefile for Cocoa Market Signals v3

.PHONY: help install dev-install test lint format run-dashboard fetch-data validate-data clean

help:
	@echo "Cocoa Market Signals v3 - Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make dev-install   - Install development dependencies"
	@echo "  make test         - Run all tests with coverage"
	@echo "  make lint         - Run code linting"
	@echo "  make format       - Format code with black"
	@echo "  make fetch-data   - Fetch latest price data"
	@echo "  make validate-data - Validate all data integrity"
	@echo "  make run-dashboard - Start the dashboard server"
	@echo "  make clean        - Clean cache and temporary files"

install:
	pip install -r requirements.txt

dev-install: install
	pip install pytest pytest-cov pytest-asyncio black flake8 mypy pre-commit
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ --line-length=100

fetch-data:
	python -m src.data_pipeline.daily_price_fetcher

validate-data:
	python -m src.validation.check_data_integrity

run-dashboard:
	python run_dashboard.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf logs/*.log

# Development shortcuts
dev: format lint test

# Check everything before commit
pre-commit: format lint test validate-data
	@echo "✅ All checks passed! Ready to commit."

# Initialize project
init: dev-install
	mkdir -p logs
	mkdir -p data/cache
	cp .env.example .env
	@echo "⚠️  Remember to edit .env with your API keys!"
	@echo "✅ Project initialized!"