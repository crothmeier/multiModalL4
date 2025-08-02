#!/bin/bash
# LLaVA FP8 Deployment Automation Script
# Zero-downtime deployment with automatic rollback capability

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_NAME="llava-fp8"
ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://phx-ai20:8888}"
MODELS_PATH="/mnt/models"
ENGINES_PATH="/mnt/engines"
LOG_DIR="/var/log/llava-deployment"
BACKUP_DIR="/mnt/backups"

# Deployment stages
STAGE=""
ROLLBACK_POINT=""

# Logging
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/deployment_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

# Trap errors and perform rollback
trap 'error_handler $? $LINENO' ERR

error_handler() {
    log_error "Deployment failed at line $2 with exit code $1"
    log_error "Current stage: $STAGE"

    if [[ -n "$ROLLBACK_POINT" ]]; then
        log_warn "Initiating rollback to: $ROLLBACK_POINT"
        rollback
    fi

    exit 1
}

# Pre-deployment checks
pre_deployment_checks() {
    STAGE="pre-deployment-checks"
    log_info "Running pre-deployment checks..."

    # Check GPU availability
    if ! nvidia-smi &>/dev/null; then
        log_error "No NVIDIA GPU detected"
        return 1
    fi

    # Check GPU memory
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [[ $GPU_MEMORY -lt 20000 ]]; then
        log_error "Insufficient GPU memory: ${GPU_MEMORY}MB < 20000MB"
        return 1
    fi

    # Check disk space
    DISK_SPACE=$(df "$ENGINES_PATH" | awk 'NR==2 {print $4}')
    if [[ $DISK_SPACE -lt 50000000 ]]; then
        log_error "Insufficient disk space for engines"
        return 1
    fi

    # Check Docker
    if ! docker info &>/dev/null; then
        log_error "Docker is not running"
        return 1
    fi

    # Check model files
    if [[ ! -d "$MODELS_PATH/llava-7b" ]]; then
        log_error "LLaVA model not found at $MODELS_PATH/llava-7b"
        return 1
    fi

    log_info "Pre-deployment checks passed"
    return 0
}

# Backup current deployment
backup_current() {
    STAGE="backup"
    log_info "Backing up current deployment..."

    BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

    # Backup docker-compose files
    if [[ -f "docker-compose.yml" ]]; then
        cp docker-compose.yml "$BACKUP_DIR/$BACKUP_NAME/"
    fi

    # Backup configuration
    if [[ -d "triton/models" ]]; then
        cp -r triton/models "$BACKUP_DIR/$BACKUP_NAME/"
    fi

    # Save current service state
    docker compose ps --format json > "$BACKUP_DIR/$BACKUP_NAME/service_state.json"

    ROLLBACK_POINT="$BACKUP_DIR/$BACKUP_NAME"
    log_info "Backup completed: $ROLLBACK_POINT"
}

# Run FP8 quantization
run_quantization() {
    STAGE="quantization"
    log_info "Starting FP8 quantization..."

    # Check if engines already exist
    if [[ -f "$ENGINES_PATH/llava-fp8/language_model_fp8.plan" ]]; then
        log_warn "FP8 engines already exist. Skipping quantization."
        return 0
    fi

    # Run quantization container
    docker compose --profile quantization up llava-quantizer

    # Verify quantization output
    if [[ ! -f "$ENGINES_PATH/llava-fp8/language_model_fp8.plan" ]]; then
        log_error "Quantization failed: engine file not created"
        return 1
    fi

    # Check quantization report
    if [[ -f "$ENGINES_PATH/llava-fp8/quantization_report.json" ]]; then
        ACCURACY=$(jq -r '.accuracy' "$ENGINES_PATH/llava-fp8/quantization_report.json")
        PASSED=$(jq -r '.passed' "$ENGINES_PATH/llava-fp8/quantization_report.json")

        if [[ "$PASSED" != "true" ]]; then
            log_error "Quantization failed accuracy check: $ACCURACY"
            return 1
        fi

        log_info "Quantization successful with accuracy: $ACCURACY"
    fi

    return 0
}

# Deploy Triton server
deploy_triton() {
    STAGE="triton-deployment"
    log_info "Deploying Triton inference server..."

    # Stop existing Triton if running
    docker compose stop triton-server-fp8 2>/dev/null || true

    # Start Triton with FP8 model
    docker compose up -d triton-server-fp8

    # Wait for Triton to be ready
    log_info "Waiting for Triton to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/v2/health/ready | grep -q "true"; then
            log_info "Triton is ready"
            return 0
        fi
        sleep 2
    done

    log_error "Triton failed to start"
    return 1
}

# Deploy API gateway
deploy_gateway() {
    STAGE="gateway-deployment"
    log_info "Deploying API gateway..."

    # Deploy gateway and dependencies
    docker compose up -d redis-cache api-gateway

    # Wait for gateway to be ready
    for i in {1..20}; do
        if curl -s http://localhost:8888/health | grep -q "healthy"; then
            log_info "API gateway is ready"
            return 0
        fi
        sleep 2
    done

    log_error "API gateway failed to start"
    return 1
}

# Deploy monitoring stack
deploy_monitoring() {
    STAGE="monitoring-deployment"
    log_info "Deploying monitoring stack..."

    # Deploy monitoring services
    docker compose up -d prometheus grafana health-monitor

    # Verify monitoring is running
    sleep 5
    if curl -s http://localhost:9090/-/ready | grep -q "Prometheus"; then
        log_info "Monitoring stack deployed"
    else
        log_warn "Monitoring stack may not be fully ready"
    fi

    return 0
}

# Register with orchestrator
register_service() {
    STAGE="orchestrator-registration"
    log_info "Registering with orchestrator..."

    # Run orchestrator hook
    docker compose run --rm orchestrator-hook python3 /scripts/orchestrator_hook.py

    # Verify registration
    RESPONSE=$(curl -s -H "Authorization: Bearer $JWT_TOKEN" \
        "$ORCHESTRATOR_URL/v1/services/$DEPLOYMENT_NAME")

    if echo "$RESPONSE" | grep -q "$DEPLOYMENT_NAME"; then
        log_info "Service registered with orchestrator"
        return 0
    else
        log_error "Failed to register with orchestrator"
        return 1
    fi
}

# Run smoke tests
run_smoke_tests() {
    STAGE="smoke-tests"
    log_info "Running smoke tests..."

    # Test inference endpoint
    TEST_IMAGE="http://images.cocodataset.org/val2017/000000397133.jpg"

    RESPONSE=$(curl -s -X POST http://localhost:8888/v1/predict \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $JWT_TOKEN" \
        -d "{\"image_url\": \"$TEST_IMAGE\", \"prompt\": \"What is in this image?\"}")

    if echo "$RESPONSE" | grep -q "generated_text"; then
        log_info "Inference test passed"
    else
        log_error "Inference test failed"
        return 1
    fi

    # Test metrics endpoint
    if curl -s http://localhost:8002/metrics | grep -q "nv_inference"; then
        log_info "Metrics endpoint test passed"
    else
        log_error "Metrics endpoint test failed"
        return 1
    fi

    return 0
}

# Enable A/B testing
enable_ab_testing() {
    STAGE="ab-testing"
    log_info "Enabling A/B testing..."

    # Deploy FP16 service for comparison
    docker compose --profile ab-testing up -d triton-server-fp16

    # Start A/B testing manager
    docker compose run -d ab-test-manager python3 /scripts/ab_testing.py

    log_info "A/B testing enabled with 10% traffic to FP16"
    return 0
}

# Health check loop
monitor_health() {
    STAGE="health-monitoring"
    log_info "Starting health monitoring..."

    # Run health check for 5 minutes
    END_TIME=$(($(date +%s) + 300))

    while [[ $(date +%s) -lt $END_TIME ]]; do
        if ! curl -s http://localhost:8888/health | grep -q "healthy"; then
            log_error "Health check failed"
            return 1
        fi
        sleep 10
    done

    log_info "Health monitoring passed"
    return 0
}

# Rollback function
rollback() {
    log_warn "Starting rollback procedure..."

    if [[ -z "$ROLLBACK_POINT" ]]; then
        log_error "No rollback point available"
        return 1
    fi

    # Stop all services
    docker compose down

    # Restore backup
    if [[ -f "$ROLLBACK_POINT/docker-compose.yml" ]]; then
        cp "$ROLLBACK_POINT/docker-compose.yml" .
    fi

    if [[ -d "$ROLLBACK_POINT/models" ]]; then
        rm -rf triton/models
        cp -r "$ROLLBACK_POINT/models" triton/
    fi

    # Restart services with backup configuration
    docker compose up -d

    log_info "Rollback completed"

    # Notify orchestrator of rollback
    curl -X POST "$ORCHESTRATOR_URL/v1/events" \
        -H "Content-Type: application/json" \
        -d "{\"event\": \"rollback\", \"service\": \"$DEPLOYMENT_NAME\", \"timestamp\": \"$(date -Iseconds)\"}"
}

# Main deployment flow
main() {
    log_info "Starting LLaVA FP8 deployment"
    log_info "Deployment name: $DEPLOYMENT_NAME"
    log_info "Orchestrator URL: $ORCHESTRATOR_URL"

    # Run deployment stages
    pre_deployment_checks
    backup_current

    # Quantization (skip if already done)
    if [[ "${SKIP_QUANTIZATION:-false}" != "true" ]]; then
        run_quantization
    fi

    # Deploy services
    deploy_triton
    deploy_gateway
    deploy_monitoring

    # Register and test
    register_service
    run_smoke_tests

    # Optional: Enable A/B testing
    if [[ "${ENABLE_AB_TESTING:-false}" == "true" ]]; then
        enable_ab_testing
    fi

    # Monitor health
    monitor_health

    log_info "Deployment completed successfully!"
    log_info "Access the service at: http://localhost:8888"
    log_info "Metrics available at: http://localhost:8002/metrics"
    log_info "Grafana dashboard at: http://localhost:3000"

    # Clear rollback point on success
    ROLLBACK_POINT=""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-quantization)
            SKIP_QUANTIZATION=true
            shift
            ;;
        --enable-ab-testing)
            ENABLE_AB_TESTING=true
            shift
            ;;
        --jwt-token)
            JWT_TOKEN="$2"
            shift 2
            ;;
        --rollback)
            # Manual rollback
            ROLLBACK_POINT=$(ls -t "$BACKUP_DIR" | head -1)
            if [[ -n "$ROLLBACK_POINT" ]]; then
                ROLLBACK_POINT="$BACKUP_DIR/$ROLLBACK_POINT"
                rollback
            else
                log_error "No backup found for rollback"
            fi
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main deployment
main
