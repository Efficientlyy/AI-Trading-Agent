# AI Crypto Trading System Makefile

.PHONY: run run-dev run-test run-prod setup test lint clean help keys dashboard

# Default environment
ENV ?= development

help:
	@echo "AI Crypto Trading System"
	@echo ""
	@echo "Usage:"
	@echo "  make run           Run the system in development mode (default)"
	@echo "  make run-dev       Run the system in development mode"
	@echo "  make run-test      Run the system in testing mode"
	@echo "  make run-prod      Run the system in production mode"
	@echo "  make dashboard     Run the dashboard"
	@echo "  make keys          Set up exchange API keys"
	@echo "  make setup         Install dependencies"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linters"
	@echo "  make clean         Clean temporary files"
	@echo ""

# Run with different environments
run: run-dev

run-dev:
	python run_trading_system.py --env development

run-test:
	python run_trading_system.py --env testing --validate-keys

run-prod:
	python run_trading_system.py --env production --validate-keys

# Run with specific components
run-sentiment:
	python run_trading_system.py --env $(ENV) --component analysis_agents --component data_collection

run-backtest:
	python run_trading_system.py --env testing --dry-run

# Dashboard
dashboard:
	python run_dashboard.py

# API keys setup
keys:
	@echo "Available commands:"
	@echo "  python setup_exchange_credentials.py add --exchange <exchange>"
	@echo "  python setup_exchange_credentials.py list [--show-keys]"
	@echo "  python setup_exchange_credentials.py validate [--exchange <exchange>]"
	@echo "  python setup_exchange_credentials.py delete --exchange <exchange>"

setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "Setting up Rust components..."
	@if [ -d "rust" ]; then \
		cd rust && cargo build --release; \
	fi

test:
	python -m pytest

lint:
	flake8 src
	pylint src
	mypy src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/