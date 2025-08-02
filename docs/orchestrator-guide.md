# Dynamic Model Orchestrator Guide

## Overview

The orchestrator automatically swaps LLM models based on incoming requests, managing GPU memory constraints by ensuring only one model is loaded at a time.

## Architecture

```
Client Request → API Gateway → Model Orchestrator → vLLM Container
                     ↓              ↓
                Check Model    Swap if needed
                     ↓              ↓
                Proxy Request   Stop old/Start new
```

## API Endpoints

### Gateway (port 8080)

- `POST /chat/completions` - Standard completion endpoint
- `GET /health` - Combined health status
- `GET /status` - Current model status

### Orchestrator (port 8888)

- `POST /ensure_model` - Ensure specific model is loaded
- `GET /status` - Current state and metrics
- `GET /health` - Orchestrator health

## Model Swap Process

1. Request arrives with model name
2. Orchestrator checks if model is loaded
3. If different model needed:
   - Stop current model container
   - Wait for GPU memory to clear
   - Start requested model container
   - Wait for health check
4. Request proceeds to model

## Performance Characteristics

- First request after swap: +5-25s (model load time)
- Subsequent requests: Normal latency
- Swap times:
  - Mistral: ~15s
  - Coder: ~20s
  - LLaVA: ~25s

## Manual Operations

### Check current model

```bash
curl http://localhost:8888/status | jq
```

### Force model swap

```bash
curl -X POST http://localhost:8888/ensure_model \
  -H "Content-Type: application/json" \
  -d '{"model": "llava"}'
```

### View logs

```bash
# Orchestrator logs
docker logs -f multimodal-stack-orchestrator

# Current model logs
docker compose logs -f

# All model containers
docker ps -a | grep llm
```

## Monitoring Metrics

- `orchestrator_model_swaps_total` - Total number of swaps
- `orchestrator_swap_duration_seconds` - Swap duration histogram
- `orchestrator_current_model` - Currently loaded model (gauge)

## Troubleshooting

### Model fails to start

1. Check GPU memory: `nvidia-smi`
2. Check container logs: `docker logs <container>`
3. Manually stop all models: `docker compose stop mistral-llm llava-llm coder-llm`
4. Restart orchestrator: `docker compose restart model-orchestrator`

### Slow swaps

- Increase swap timeout in orchestrator
- Pre-pull model images
- Consider reducing model sizes

### Concurrent request handling

- Orchestrator queues requests during swaps
- Use request timeout on client side
- Consider adding request queue metrics
