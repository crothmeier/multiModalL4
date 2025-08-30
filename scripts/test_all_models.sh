#!/bin/bash
# Test all models with orchestration

echo "Testing Model Orchestration"
echo "=========================="

for model in mistral coder llava; do
  echo -e "\nTesting $model..."
  START=$(date +%s)

  curl -s -X POST http://localhost:8080/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$model\",
        \"messages\": [{\"role\": \"user\", \"content\": \"What model are you?\"}],
        \"max_tokens\": 20
      }" | jq -r '.choices[0].message.content'

  END=$(date +%s)
  echo "Time: $((END - START))s"
done

echo -e "\nFinal status:"
curl -s http://localhost:8888/status | jq
