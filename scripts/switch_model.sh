#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MODEL=$1

show_usage() {
    echo "Usage: $0 <model>"
    echo ""
    echo "Available models:"
    echo "  mistral   - Mistral-7B AWQ (7GB VRAM, port 8001)"
    echo "  coder     - DeepSeek Coder GPTQ (11GB VRAM, port 8002)"
    echo "  llava     - LLaVA Multimodal (20GB VRAM, port 8003)"
    echo "  stop      - Stop all running models"
    echo ""
    echo "Example:"
    echo "  $0 mistral"
    echo "  $0 stop"
}

stop_all_models() {
    echo -e "${YELLOW}Stopping all models...${NC}"
    docker compose down 2>/dev/null || true
    sleep 2
    echo -e "${GREEN}✓ All models stopped${NC}"
}

check_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${BLUE}GPU Memory Status:${NC}"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
            awk '{printf "  Used: %dMB / %dMB (%.1f%%)\n", $1, $3, ($1/$3)*100}'
    fi
}

start_model() {
    local profile=$1
    local port=$2
    local model_name=$3

    echo -e "${BLUE}Starting $model_name...${NC}"

    stop_all_models

    check_gpu_memory

    echo -e "${YELLOW}Loading model with profile: $profile${NC}"
    docker compose --profile "$profile" up -d

    echo -e "${YELLOW}Waiting for model to be ready...${NC}"
    local max_attempts=60
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:"$port"/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $model_name is ready!${NC}"
            echo -e "${GREEN}API endpoint: http://localhost:$port${NC}"
            check_gpu_memory
            return 0
        fi

        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "\n${RED}✗ Model failed to start. Check logs:${NC}"
    echo "  docker compose logs --tail 50"
    return 1
}

if [ -z "$MODEL" ]; then
    show_usage
    exit 1
fi

case "$MODEL" in
    mistral)
        start_model "mistral" "8001" "Mistral-7B AWQ"
        ;;
    coder)
        start_model "coder" "8002" "DeepSeek Coder GPTQ"
        ;;
    llava)
        start_model "llava" "8003" "LLaVA Multimodal"
        ;;
    stop)
        stop_all_models
        ;;
    *)
        echo -e "${RED}Error: Unknown model '$MODEL'${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac
