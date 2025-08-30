#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Validating multimodal LLM stack directory structure..."

REQUIRED_DIRS=(
    "/mnt/models"
    "/mnt/models/mistral-7b-awq"
    "/mnt/models/deepseek-coder-gptq"
    "/mnt/models/llava"
    "./logs"
    "./scripts"
)

ERRORS=0

for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ "$dir" == /mnt/* ]]; then
        if [ ! -d "$dir" ]; then
            echo -e "${YELLOW}Warning: $dir does not exist (will be created during setup)${NC}"
        else
            echo -e "${GREEN}✓ $dir exists${NC}"
        fi
    else
        if [ ! -d "$dir" ]; then
            echo -e "${RED}✗ $dir is missing${NC}"
            ERRORS=$((ERRORS + 1))
        else
            echo -e "${GREEN}✓ $dir exists${NC}"
        fi
    fi
done

if [ -f ".env" ]; then
    if grep -q "HF_TOKEN" .env; then
        echo -e "${GREEN}✓ HF_TOKEN found in .env${NC}"
    else
        echo -e "${YELLOW}Warning: HF_TOKEN not found in .env${NC}"
    fi
else
    echo -e "${YELLOW}Warning: .env file not found${NC}"
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓ Docker is installed${NC}"
fi

if ! docker compose version &> /dev/null; then
    echo -e "${RED}✗ Docker Compose is not available${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓ Docker Compose is available${NC}"
fi

if [ $ERRORS -eq 0 ]; then
    echo -e "\n${GREEN}All validations passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Validation failed with $ERRORS errors${NC}"
    exit 1
fi
