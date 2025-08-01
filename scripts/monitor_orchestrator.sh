#!/bin/bash
# Real-time orchestrator monitoring
watch -n 2 '
echo "=== Orchestrator Status ==="
curl -s http://localhost:8888/status | jq
echo -e "\n=== GPU Usage ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo -e "\n=== Container Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}" | grep llm
'
