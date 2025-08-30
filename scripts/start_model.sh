#!/bin/bash
# Helper script to manually start a specific model
MODEL=$1
if [ -z "$MODEL" ]; then
  echo "Usage: $0 <mistral|llava|coder>"
  exit 1
fi

echo "Starting $MODEL..."
docker compose up -d "${MODEL}-llm"
docker compose logs -f "${MODEL}-llm"
