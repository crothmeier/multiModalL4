#!/bin/bash
# Automatic rollback script for FP8 to FP16 fallback

set -euo pipefail

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "ALERT: Initiating automatic rollback to FP16 model"

# Update orchestrator routing to 100% FP16
curl -X PUT http://phx-ai20:8888/v1/services/llava/config \
    -H "Content-Type: application/json" \
    -d '{
        "primary_variant": "fp16",
        "routing": {"fp8": 0.0, "fp16": 1.0},
        "reason": "automatic_rollback",
        "timestamp": "'$(date -Iseconds)'"
    }'

# Scale down FP8 service
docker compose scale triton-server-fp8=0

# Scale up FP16 service
docker compose --profile ab-testing scale triton-server-fp16=2

# Update gateway configuration
docker compose exec api-gateway sh -c 'echo "FP16_PRIMARY=true" >> /etc/environment'
docker compose restart api-gateway

# Send alert
if [[ -n "${ALERT_WEBHOOK_URL:-}" ]]; then
    curl -X POST "$ALERT_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d '{
            "severity": "critical",
            "event": "fp8_rollback",
            "message": "Automatically rolled back to FP16 due to performance issues",
            "timestamp": "'$(date -Iseconds)'"
        }'
fi

log "Rollback to FP16 completed successfully"
