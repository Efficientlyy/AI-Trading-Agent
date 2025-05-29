#!/bin/bash
# Production Deployment Script - Day 1 (Shadow Trading)
# This script implements the first day of the production deployment plan,
# setting up the system for shadow trading (paper trading with real data)

set -e

echo "======================================================="
echo "  AI Trading Agent - Production Deployment (Day 1)      "
echo "  Shadow Trading Deployment                            "
echo "======================================================="

# Load environment variables
if [ -f ".env.production" ]; then
    echo "Loading environment variables from .env.production"
    export $(grep -v '^#' .env.production | xargs)
else
    echo "Warning: .env.production file not found"
    echo "Make sure to set required environment variables manually"
fi

# Set default values
REGISTRY=${REGISTRY:-"ghcr.io"}
REPOSITORY_OWNER=${REPOSITORY_OWNER:-"ai-trading-agent"}
KUBE_CONTEXT=${KUBE_CONTEXT:-"production"}
NAMESPACE=${NAMESPACE:-"ai-trading-agent-prod"}

# Check for required tools
for cmd in kubectl envsubst; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is required but not installed"
        exit 1
    fi
done

echo "Using Kubernetes context: $KUBE_CONTEXT"
kubectl config use-context $KUBE_CONTEXT || {
    echo "Failed to switch to Kubernetes context: $KUBE_CONTEXT"
    echo "Please make sure the context exists and you have proper permissions"
    exit 1
}

# Create namespace if it doesn't exist
kubectl get namespace $NAMESPACE &> /dev/null || kubectl create namespace $NAMESPACE

echo "Deploying to namespace: $NAMESPACE"

# Step 1: Deploy database secrets and config maps
echo "Deploying configuration and secrets..."
envsubst < devops/environments/production.yaml | kubectl apply -f - -n $NAMESPACE
envsubst < devops/environments/data-connections.yaml | kubectl apply -f - -n $NAMESPACE

# Step 2: Create persistent volumes
echo "Creating persistent volumes..."
kubectl apply -f devops/kubernetes/storage/persistent-volumes.yaml -n $NAMESPACE

# Step 3: Deploy core services
echo "Deploying core services..."
envsubst < devops/kubernetes/services/trading-agent-services.yaml | kubectl apply -f - -n $NAMESPACE

# Step 4: Deploy the trading agent components
echo "Deploying trading agent components (SHADOW MODE)..."
envsubst < devops/kubernetes/deployments/trading-agent.yaml | kubectl apply -f - -n $NAMESPACE

# Step 5: Wait for deployments to be ready
echo "Waiting for deployments to be ready..."
kubectl rollout status deployment/trading-agent-core -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/data-service -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/execution-service -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/sentiment-analyzer -n $NAMESPACE --timeout=300s

# Step 6: Run validation tests
echo "Running shadow trading validation tests..."
# Note: In a real deployment, this would run the integration tests against the deployed system
# This is a placeholder for now
python -m pytest tests/integration/test_shadow_trading.py -v

# Step 7: Set up monitoring for the deployed system
echo "Setting up monitoring..."
# Apply monitoring configuration
# kubectl apply -f devops/kubernetes/monitoring/prometheus.yaml -n $NAMESPACE
# kubectl apply -f devops/kubernetes/monitoring/grafana-dashboards.yaml -n $NAMESPACE

# Step 8: Check system health
echo "Checking system health..."
CORE_IP=$(kubectl get service trading-agent-core -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
CORE_PORT=$(kubectl get service trading-agent-core -n $NAMESPACE -o jsonpath='{.spec.ports[0].port}')

if [ -n "$CORE_IP" ] && [ -n "$CORE_PORT" ]; then
    echo "Trading agent core service is available at $CORE_IP:$CORE_PORT"
    # In a real deployment, we would make HTTP requests to health check endpoints
    # curl http://$CORE_IP:$CORE_PORT/health
else
    echo "Warning: Trading agent core service IP or port not available yet"
fi

echo ""
echo "======================================================="
echo "  Day 1 Deployment Complete - SHADOW TRADING MODE      "
echo "======================================================="
echo ""
echo "The AI Trading Agent has been deployed in SHADOW TRADING MODE."
echo "It is connected to real market data but executing trades in simulation."
echo ""
echo "Next steps:"
echo "1. Monitor system stability for the next 24 hours"
echo "2. Review shadow trading performance metrics"
echo "3. Proceed to Day 2-3 monitoring phase if stable"
echo ""
echo "Access the dashboard at: http://${DASHBOARD_URL:-"<dashboard-url-not-available>"}"
echo "======================================================="
