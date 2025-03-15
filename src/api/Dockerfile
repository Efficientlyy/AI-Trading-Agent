FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code
COPY . .

# Create directory for visualizations
RUN mkdir -p ./visualizations

# Expose the port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "regime_detection_api:app", "--host", "0.0.0.0", "--port", "8000"] 