# Makefile for AI Predictive Maintenance System

.PHONY: install test run docker-build docker-run clean help

help:
	@echo "Usage:"
	@echo "  make install      Install dependencies"
	@echo "  make test         Run tests"
	@echo "  make run          Run the FastAPI application"
	@echo "  make docker-build Build Docker images"
	@echo "  make docker-run   Run the system using docker-compose"
	@echo "  make clean        Remove temporary files"

install:
	pip install -r requirements.txt

test:
	pytest tests/

run:
	uvicorn src.api.main:app --reload

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
