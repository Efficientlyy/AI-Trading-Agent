#!/bin/bash
# Automated deployment script for AI Trading Agent
# This script handles deployments to different environments with different strategies
# Usage: ./deploy.sh <environment> <deployment_type>

set -e

# Configuration
ENVIRONMENTS=("development" "staging" "production")
DEPLOYMENT_TYPES=("standard" "canary" "blue-green")
NAMESPACE_PREFIX="ai-trading-agent"
COMPONENTS=("agent" "data-service" "execution-service" "oversight-service" "frontend")
REGISTRY="ghcr.io"
IMAGE_PREFIX="ai-trading-agent"

# Parse arguments
ENVIRONMENT=$1
DEPLOYMENT_TYPE=$2

# Validate environment
if [[ ! " ${ENVIRONMENTS[@]} " =~ " ${ENVIRONMENT} " ]]; then
    echo "Error: Invalid environment. Must be one of: ${ENVIRONMENTS[*]}"
    exit 1
fi

# Validate deployment type
if [[ ! " ${DEPLOYMENT_TYPES[@]} " =~ " ${DEPLOYMENT_TYPE} " ]]; then
    echo "Error: Invalid deployment type. Must be one of: ${DEPLOYMENT_TYPES[*]}"
    exit 1
fi

# Set namespace based on environment
NAMESPACE="${NAMESPACE_PREFIX}-${ENVIRONMENT}"

echo "=== Starting deployment ==="
echo "Environment: $ENVIRONMENT"
echo "Deployment Type: $DEPLOYMENT_TYPE"
echo "Namespace: $NAMESPACE"

# Ensure namespace exists
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply common resources
echo "Applying common resources..."
kubectl apply -f kubernetes/common/configmaps.yaml -n $NAMESPACE
kubectl apply -f kubernetes/common/secrets.yaml -n $NAMESPACE

# Load environment-specific configurations
echo "Applying environment-specific configurations..."
if [ -f "kubernetes/environments/$ENVIRONMENT/config.yaml" ]; then
    kubectl apply -f kubernetes/environments/$ENVIRONMENT/config.yaml -n $NAMESPACE
fi

if [ -f "kubernetes/environments/$ENVIRONMENT/secrets.yaml" ]; then
    kubectl apply -f kubernetes/environments/$ENVIRONMENT/secrets.yaml -n $NAMESPACE
fi

# Function to deploy a component using the standard strategy
deploy_standard() {
    component=$1
    echo "Deploying $component using standard strategy..."
    kubectl apply -f kubernetes/$component/deployment.yaml -n $NAMESPACE
    kubectl apply -f kubernetes/$component/service.yaml -n $NAMESPACE
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/$component -n $NAMESPACE --timeout=300s
}

# Function to deploy a component using the canary strategy
deploy_canary() {
    component=$1
    echo "Deploying $component using canary strategy..."
    
    # Check if we have a existing deployment
    if kubectl get deployment/$component -n $NAMESPACE &>/dev/null; then
        # Apply canary configuration
        kubectl apply -f kubernetes/$component/canary-deployment.yaml -n $NAMESPACE
        
        # Initial traffic split: 90% to stable, 10% to canary
        echo "Initial traffic split: 90% stable, 10% canary"
        kubectl apply -f kubernetes/$component/traffic-split-90-10.yaml -n $NAMESPACE
        
        # Wait for canary deployment to be ready
        kubectl rollout status deployment/$component-canary -n $NAMESPACE --timeout=300s
        
        # Prompt for canary promotion or automatic based on environment
        if [ "$ENVIRONMENT" = "production" ]; then
            read -p "Do you want to continue with canary promotion? (y/n): " promote
            if [ "$promote" != "y" ]; then
                echo "Canary promotion aborted, rolling back..."
                kubectl delete -f kubernetes/$component/canary-deployment.yaml -n $NAMESPACE
                kubectl apply -f kubernetes/$component/traffic-split-100-0.yaml -n $NAMESPACE
                return
            fi
        fi
        
        # Gradually shift traffic
        echo "Shifting traffic: 50% stable, 50% canary"
        kubectl apply -f kubernetes/$component/traffic-split-50-50.yaml -n $NAMESPACE
        sleep 30
        
        echo "Shifting traffic: 0% stable, 100% canary"
        kubectl apply -f kubernetes/$component/traffic-split-0-100.yaml -n $NAMESPACE
        sleep 30
        
        # Promote canary to stable
        echo "Promoting canary to stable..."
        kubectl apply -f kubernetes/$component/deployment.yaml -n $NAMESPACE
        kubectl rollout status deployment/$component -n $NAMESPACE --timeout=300s
        
        # Clean up canary resources
        kubectl delete -f kubernetes/$component/canary-deployment.yaml -n $NAMESPACE
        kubectl apply -f kubernetes/$component/traffic-split-100-0.yaml -n $NAMESPACE
    else
        # Initial deployment
        echo "No existing deployment, deploying initial version..."
        kubectl apply -f kubernetes/$component/deployment.yaml -n $NAMESPACE
        kubectl apply -f kubernetes/$component/service.yaml -n $NAMESPACE
        kubectl rollout status deployment/$component -n $NAMESPACE --timeout=300s
    fi
}

# Function to deploy a component using the blue-green strategy
deploy_blue_green() {
    component=$1
    echo "Deploying $component using blue-green strategy..."
    
    # Determine current active color (blue or green)
    current_color="blue"
    if kubectl get service/$component -n $NAMESPACE -o jsonpath='{.spec.selector.color}' 2>/dev/null | grep -q "blue"; then
        current_color="blue"
        new_color="green"
    else
        current_color="green"
        new_color="blue"
    fi
    
    echo "Current active color: $current_color, deploying to $new_color"
    
    # Deploy new version with the new color
    sed "s/COLOR/$new_color/g" kubernetes/$component/deployment-template.yaml > kubernetes/$component/$new_color-deployment.yaml
    kubectl apply -f kubernetes/$component/$new_color-deployment.yaml -n $NAMESPACE
    
    # Wait for new deployment to be ready
    kubectl rollout status deployment/$component-$new_color -n $NAMESPACE --timeout=300s
    
    # Prompt for production switchover
    if [ "$ENVIRONMENT" = "production" ]; then
        read -p "Do you want to switch traffic to the new $new_color deployment? (y/n): " switch
        if [ "$switch" != "y" ]; then
            echo "Traffic switch aborted, keeping $current_color active..."
            rm kubernetes/$component/$new_color-deployment.yaml
            return
        fi
    fi
    
    # Switch traffic to new color
    echo "Switching traffic to $new_color deployment..."
    sed "s/COLOR/$new_color/g" kubernetes/$component/service-template.yaml > kubernetes/$component/service.yaml
    kubectl apply -f kubernetes/$component/service.yaml -n $NAMESPACE
    
    # Wait for a while to ensure everything is working
    echo "Waiting 60 seconds to verify new deployment stability..."
    sleep 60
    
    # If all is well, remove the old deployment
    echo "Removing old $current_color deployment..."
    kubectl delete deployment/$component-$current_color -n $NAMESPACE
    
    # Clean up
    rm kubernetes/$component/$new_color-deployment.yaml
}

# Deploy each component based on deployment type
for component in "${COMPONENTS[@]}"; do
    echo "=== Deploying $component ==="
    
    case $DEPLOYMENT_TYPE in
        "standard")
            deploy_standard $component
            ;;
        "canary")
            deploy_canary $component
            ;;
        "blue-green")
            deploy_blue_green $component
            ;;
    esac
    
    echo "$component deployment completed"
    echo ""
done

# Apply ingress resources
echo "Applying ingress resources..."
kubectl apply -f kubernetes/ingress/$ENVIRONMENT.yaml -n $NAMESPACE

echo "=== Deployment complete ==="
echo "Verifying all deployments..."

# List all deployments and their status
kubectl get deployments -n $NAMESPACE

# Create a simple test to verify the deployment
echo "Running deployment tests..."
./devops/test_deployment.sh $ENVIRONMENT

echo "Deployment to $ENVIRONMENT environment completed successfully!"
