#!/bin/bash
# Script to deploy the Market Regime Detection API

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Market Regime Detection API Deployment${NC}"
echo "========================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker before continuing."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed.${NC}"
    echo "Please install Docker Compose before continuing."
    exit 1
fi

# Create visualizations directory if it doesn't exist
if [ ! -d "visualizations" ]; then
    echo -e "${YELLOW}Creating visualizations directory...${NC}"
    mkdir -p visualizations
fi

# Build and start the containers
echo -e "${YELLOW}Building and starting containers...${NC}"
docker-compose up --build -d

# Check if the containers are running
if [ $? -eq 0 ]; then
    echo -e "${GREEN}API service is now running!${NC}"
    echo "You can access the API at: http://localhost:8000"
    echo "API documentation is available at: http://localhost:8000/docs"
    
    # Run the test script
    echo -e "${YELLOW}Running API tests...${NC}"
    python test_api.py
    
    echo -e "${GREEN}Deployment complete!${NC}"
else
    echo -e "${RED}Error: Failed to start the API service.${NC}"
    echo "Check the logs with: docker-compose logs"
    exit 1
fi

echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "- View logs: docker-compose logs -f"
echo "- Stop service: docker-compose down"
echo "- Restart service: docker-compose restart"
echo "- Run client example: python client_example.py" 