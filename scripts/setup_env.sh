#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Multimodal LLM Stack Environment Setup ===${NC}"

if [ -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file already exists${NC}"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env file"
        exit 0
    fi
fi

if [ ! -f ".env.example" ]; then
    cat > .env.example << 'EOF'
# Multimodal LLM Stack Configuration

# Required: HuggingFace token for model downloads
HF_TOKEN=hf_your_token_here

# Optional: API key for vLLM endpoints (default: default-key)
VLLM_API_KEY=your-secure-api-key

# Model storage directory (default: /mnt/models)
MODEL_DIR=/mnt/models

# Log directory (default: ./logs)
LOG_DIR=./logs
EOF
    echo -e "${GREEN}✓ Created .env.example${NC}"
fi

cp .env.example .env
echo -e "${GREEN}✓ Created .env from .env.example${NC}"

echo -e "\n${YELLOW}Please edit .env and add your HF_TOKEN:${NC}"
echo "  1. Get your token from https://huggingface.co/settings/tokens"
echo "  2. Edit .env and replace 'hf_your_token_here' with your actual token"
echo ""

read -r -p "Do you have your HF_TOKEN ready? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your HF_TOKEN: " token
    if [ -n "$token" ]; then
        sed -i "s/hf_your_token_here/$token/" .env
        echo -e "${GREEN}✓ HF_TOKEN configured${NC}"
    fi
fi

if [ ! -d "/mnt/models" ]; then
    echo -e "\n${YELLOW}Note: /mnt/models directory does not exist${NC}"
    echo "You may need to create it with appropriate permissions:"
    echo "  sudo mkdir -p /mnt/models"
    echo "  sudo chown $USER:$USER /mnt/models"
fi

echo -e "\n${BLUE}Verifying environment...${NC}"
# shellcheck source=/dev/null
source .env

if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" == "hf_your_token_here" ]; then
    echo -e "${RED}✗ HF_TOKEN is not configured${NC}"
    echo "Please edit .env and add your HuggingFace token"
else
    echo -e "${GREEN}✓ HF_TOKEN is configured${NC}"
fi

if [ -n "$VLLM_API_KEY" ] && [ "$VLLM_API_KEY" != "your-secure-api-key" ]; then
    echo -e "${GREEN}✓ VLLM_API_KEY is configured${NC}"
else
    echo -e "${YELLOW}! VLLM_API_KEY using default (consider setting a secure key)${NC}"
fi

echo -e "\n${GREEN}Environment setup complete!${NC}"
echo "Next steps:"
echo "  1. Ensure HF_TOKEN is configured in .env"
echo "  2. Run: ./scripts/download_models.sh"
echo "  3. Start a model: docker compose --profile mistral up -d"
