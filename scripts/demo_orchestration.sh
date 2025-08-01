#!/bin/bash
# Demo script for model orchestration

echo "=== Dynamic Model Orchestrator Demo ==="
echo

# Function to test model
test_model() {
    local model=$1
    local prompt=$2
    echo "Testing $model..."
    echo "Prompt: $prompt"

    # Direct test without orchestrator for now
    response=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$model\",
        \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
        \"max_tokens\": 50
      }" | jq -r '.choices[0].message.content // .error.message // "Error"')

    echo "Response: $response"
    echo
}

# Check current status
echo "Current orchestrator status:"
curl -s http://localhost:8888/status | jq
echo

# Test with current model
echo "=== Testing with Mistral (currently running) ==="
test_model "mistral" "What are you and what can you do in one sentence?"

# Check GPU usage
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo

# Manual swap demonstration
echo "=== Demonstrating Manual Model Swap ==="
echo "Stopping Mistral..."
docker stop multimodal-stack-mistral-llm-1 >/dev/null 2>&1
sleep 3

echo "Starting Coder model..."
docker start multimodal-stack-coder-llm-1 2>/dev/null || docker compose up -d coder-llm
echo "Waiting for model to load..."
sleep 20

echo "Testing Coder model..."
test_model "deepseek" "Write a Python hello world function"

echo "GPU Memory after swap:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo

echo "=== Demo Complete ==="
echo "The orchestrator automatically handles these swaps when integrated with the API gateway."
