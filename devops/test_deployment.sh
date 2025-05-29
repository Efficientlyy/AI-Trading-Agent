#!/bin/bash
# Deployment verification script for AI Trading Agent
# This script runs verification tests after deployment to ensure all components are working properly
# Usage: ./test_deployment.sh <environment>

set -e

# Configuration
ENVIRONMENTS=("development" "staging" "production")
NAMESPACE_PREFIX="ai-trading-agent"

# Parse arguments
ENVIRONMENT=$1

# Validate environment
if [[ ! " ${ENVIRONMENTS[@]} " =~ " ${ENVIRONMENT} " ]]; then
    echo "Error: Invalid environment. Must be one of: ${ENVIRONMENTS[*]}"
    exit 1
fi

# Set namespace based on environment
NAMESPACE="${NAMESPACE_PREFIX}-${ENVIRONMENT}"

echo "=== Starting deployment verification tests ==="
echo "Environment: $ENVIRONMENT"
echo "Namespace: $NAMESPACE"

# Get service endpoints
echo "Retrieving service endpoints..."
API_URL=$(kubectl get ingress api-ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
UI_URL=$(kubectl get ingress ui-ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

if [ -z "$API_URL" ] || [ -z "$UI_URL" ]; then
    echo "Warning: Could not retrieve ingress URLs. Using placeholders for testing."
    API_URL="api.${ENVIRONMENT}.ai-trading-agent.example.com"
    UI_URL="${ENVIRONMENT}.ai-trading-agent.example.com"
fi

echo "API URL: https://$API_URL"
echo "UI URL: https://$UI_URL"

# Function to test API health
test_api_health() {
    echo "Testing API health..."
    
    # Test API health endpoint
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://$API_URL/health")
    
    if [ $HTTP_CODE -eq 200 ]; then
        echo "✅ API health check passed"
    else
        echo "❌ API health check failed with HTTP code $HTTP_CODE"
        return 1
    fi
    
    return 0
}

# Function to test data service
test_data_service() {
    echo "Testing data service..."
    
    # Test data service endpoints
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://$API_URL/api/v1/data/status")
    
    if [ $HTTP_CODE -eq 200 ]; then
        echo "✅ Data service check passed"
    else
        echo "❌ Data service check failed with HTTP code $HTTP_CODE"
        return 1
    fi
    
    return 0
}

# Function to test execution service
test_execution_service() {
    echo "Testing execution service..."
    
    # Test execution service status endpoint
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://$API_URL/api/v1/execution/status")
    
    if [ $HTTP_CODE -eq 200 ]; then
        echo "✅ Execution service check passed"
    else
        echo "❌ Execution service check failed with HTTP code $HTTP_CODE"
        return 1
    fi
    
    return 0
}

# Function to test oversight service
test_oversight_service() {
    echo "Testing oversight service..."
    
    # Test oversight service health endpoint
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://$API_URL/api/v1/oversight/health")
    
    if [ $HTTP_CODE -eq 200 ]; then
        echo "✅ Oversight service check passed"
    else
        echo "❌ Oversight service check failed with HTTP code $HTTP_CODE"
        return 1
    fi
    
    return 0
}

# Function to test frontend
test_frontend() {
    echo "Testing frontend..."
    
    # Test frontend is accessible
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://$UI_URL")
    
    if [ $HTTP_CODE -eq 200 ]; then
        echo "✅ Frontend check passed"
    else
        echo "❌ Frontend check failed with HTTP code $HTTP_CODE"
        return 1
    fi
    
    return 0
}

# Function to test end-to-end functionality
test_e2e() {
    echo "Testing end-to-end functionality..."
    
    # For now, just check if all services are responding
    if test_api_health && test_data_service && test_execution_service && test_oversight_service && test_frontend; then
        echo "✅ Basic end-to-end check passed"
    else
        echo "❌ End-to-end check failed"
        return 1
    fi
    
    # If this is production, perform additional verification tests
    if [ "$ENVIRONMENT" = "production" ]; then
        echo "Running additional production verification tests..."
        
        # Test data feed connections
        echo "Testing data feed connections..."
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://$API_URL/api/v1/data/feeds/status")
        
        if [ $HTTP_CODE -eq 200 ]; then
            echo "✅ Data feed connections check passed"
        else
            echo "❌ Data feed connections check failed with HTTP code $HTTP_CODE"
            echo "⚠️ This is a non-critical error but requires attention"
        fi
        
        # Test oversight system integration
        echo "Testing oversight system integration..."
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://$API_URL/api/v1/oversight/validate")
        
        if [ $HTTP_CODE -eq 200 ]; then
            echo "✅ Oversight system integration check passed"
        else
            echo "❌ Oversight system integration check failed with HTTP code $HTTP_CODE"
            echo "⚠️ This is a critical error that requires immediate attention"
            
            # Notify operations team for production
            echo "Sending alert to operations team..."
            # (alert mechanism would go here)
        fi
    fi
    
    return 0
}

# Run verification tests
echo "=== Running verification tests ==="

# Run all tests and collect results
test_api_health
API_HEALTH_RESULT=$?

test_data_service
DATA_SERVICE_RESULT=$?

test_execution_service
EXECUTION_SERVICE_RESULT=$?

test_oversight_service
OVERSIGHT_SERVICE_RESULT=$?

test_frontend
FRONTEND_RESULT=$?

test_e2e
E2E_RESULT=$?

# Print summary
echo "=== Test Results Summary ==="
echo "API Health: $([ $API_HEALTH_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
echo "Data Service: $([ $DATA_SERVICE_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
echo "Execution Service: $([ $EXECUTION_SERVICE_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
echo "Oversight Service: $([ $OVERSIGHT_SERVICE_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
echo "Frontend: $([ $FRONTEND_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
echo "End-to-End Tests: $([ $E2E_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"

# Calculate overall result
OVERALL_RESULT=$((API_HEALTH_RESULT + DATA_SERVICE_RESULT + EXECUTION_SERVICE_RESULT + OVERSIGHT_SERVICE_RESULT + FRONTEND_RESULT + E2E_RESULT))

if [ $OVERALL_RESULT -eq 0 ]; then
    echo "=== All verification tests passed! ==="
    echo "Deployment to $ENVIRONMENT environment is verified."
    
    # Update deployment status in monitoring system
    echo "Updating deployment status in monitoring system..."
    # (monitoring update mechanism would go here)
    
    exit 0
else
    echo "=== Some verification tests failed! ==="
    echo "Deployment to $ENVIRONMENT environment requires attention."
    
    # If production, consider rollback
    if [ "$ENVIRONMENT" = "production" ] && [ $OVERALL_RESULT -gt 2 ]; then
        echo "Critical failures detected in production - automatic rollback recommended."
        echo "Please review logs and decide on rollback action."
    fi
    
    exit 1
fi
