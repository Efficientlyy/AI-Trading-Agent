# Docker Setup for Modular Dashboard

This guide explains how to use Docker to run the modular dashboard for the AI Trading Agent project.

## Overview

The Docker configuration has been updated to support the new modular dashboard architecture. The setup includes:

- A main container for the dashboard application
- Volume mounts for real-time development
- Health checks for container monitoring
- Environment variable configuration

## Prerequisites

- Docker and Docker Compose installed on your machine
- Git repository cloned locally

## Building and Running

### Option 1: Using Docker Compose (Recommended)

The simplest way to start the entire system including the dashboard and its dependencies:

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode (background)
docker-compose up -d --build

# View logs
docker-compose logs -f dashboard
```

### Option 2: Using Docker Directly

If you want to run just the dashboard container without other services:

```bash
# Build the image
docker build -t ai-trading-dashboard .

# Run the container
docker run -p 5000:5000 -p 8050:8050 \
  -e FLASK_ENV=development \
  -e PYTHONPATH=/app \
  -e NEXT_PUBLIC_USE_MOCK_DATA=true \
  -e FLASK_SECRET_KEY=ai-trading-dashboard-secret \
  ai-trading-dashboard
```

## Environment Variables

The following environment variables can be configured:

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Flask environment | `development` |
| `PYTHONPATH` | Python module path | `/app` |
| `NEXT_PUBLIC_USE_MOCK_DATA` | Use mock data flag | `true` |
| `FLASK_SECRET_KEY` | Secret key for Flask sessions | `ai-trading-dashboard-secret` |

## Volume Mounts

The Docker Compose configuration includes the following volume mounts for development:

- `./src/dashboard:/app/src/dashboard` - Dashboard source code
- `./templates:/app/templates` - HTML templates
- `./static:/app/static` - Static assets
- `./logs:/app/logs` - Log files

This allows you to make changes to these files on your host machine and see them reflected in the container without rebuilding the image.

## Health Checks

The container includes a health check endpoint at `/health` that returns a 200 OK response when the service is running properly. Docker Compose is configured to monitor this endpoint and restart the container if it becomes unhealthy.

## Development Workflow

For the best development experience:

1. Make changes to the dashboard files on your local machine
2. The changes will be reflected in the container thanks to the volume mounts
3. If you need to restart the application without rebuilding, use:
   ```bash
   docker-compose restart dashboard
   ```

## Troubleshooting

### Container Fails to Start

If the container fails to start, check the logs:

```bash
docker-compose logs dashboard
```

Common issues include:

- Port conflicts: Change the exposed ports in docker-compose.yml
- Missing directories: Ensure the required directories exist in your project
- Permission issues: Check file permissions on volume-mounted directories

### Application Errors

If the container starts but the application shows errors:

1. Check the application logs in the `logs` directory
2. Ensure all dependencies are correctly installed (check Dockerfile)
3. Verify the environment variables are set correctly

## Building for Production

For production deployment, consider making these adjustments:

1. Set `FLASK_ENV=production` in docker-compose.yml
2. Use a strong, randomly generated `FLASK_SECRET_KEY`
3. Consider using Docker Swarm or Kubernetes for orchestration
4. Set up proper SSL termination with a reverse proxy like Nginx
