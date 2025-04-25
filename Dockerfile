# Multi-stage Dockerfile for AI Trading Agent

# ============== Builder Stage ==============
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    build-essential \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust for building the Rust extension
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create and set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry==1.7.1

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not use virtualenvs (we're in a container)
RUN poetry config virtualenvs.create false

# Install dependencies without development dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy Rust extension source code
COPY rust_backtesting_extension /app/rust_backtesting_extension

# Build the Rust extension
RUN cd /app/rust_backtesting_extension && \
    cargo build --release && \
    cp target/release/librust_backtesting.so /app/ai_trading_agent/

# ============== Production Stage ==============
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy Rust extension library
COPY --from=builder /app/ai_trading_agent/librust_backtesting.so /app/ai_trading_agent/

# Copy application code
COPY . /app/

# Fix permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port the server will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ============== Development Stage ==============
FROM builder AS development

# Install development dependencies
RUN poetry install --no-interaction --no-ansi

# Copy application code
COPY . /app/

# Expose the port the server will run on
EXPOSE 8000

# Command to run the application in development mode with hot reload
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]