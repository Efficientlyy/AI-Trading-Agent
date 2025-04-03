FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files for dashboard dependencies first (better layer caching)
COPY requirements.txt .

# Install minimal required dependencies for modular dashboard
RUN pip install --no-cache-dir \
    flask==2.0.1 \
    flask-socketio==5.3.0 \
    dash==2.6.0 \
    dash-bootstrap-components==1.2.0 \
    plotly==5.9.0 \
    pandas==1.4.3 \
    numpy==1.22.4 \
    psutil==5.9.1 \
    structlog==22.1.0

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p templates static logs \
    && mkdir -p src/dashboard/components \
    && mkdir -p src/dashboard/utils

# Set environment variables for feature flags and flask configuration
ENV FLASK_ENV=development
ENV PYTHONPATH=/app
ENV NEXT_PUBLIC_USE_MOCK_DATA=true
ENV FLASK_SECRET_KEY=ai-trading-dashboard-secret

# Expose the ports
EXPOSE 5000 8050

# Health check file for Docker health checks
RUN echo "OK" > /app/health.txt

# Run the modular dashboard with explicit port and host settings
CMD ["python", "run_modular_dashboard.py", "--host", "0.0.0.0", "--port", "5000"]
