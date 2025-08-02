# multiModalL4

MultiModal L4 chatbot assistant

## Prerequisites

### Required Tools

- **Docker** ≥ 20.10 with Docker Compose v2
- **NVIDIA Container Toolkit** for GPU support
- **yq** ≥ 4.40 - YAML processor ([install instructions](https://github.com/mikefarah/yq#install))

  ```bash
  # macOS
  brew install yq

  # Linux (snap)
  snap install yq

  # Linux (binary)
  wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/local/bin/yq
  chmod +x /usr/local/bin/yq
  ```

### Environment Variables

Create a `.env` file with:

```bash
HF_TOKEN=your-huggingface-token  # Required for model downloads
JWT_SECRET=your-secret-key        # Required for API authentication
```

## Quick Start

1. **Validate Configuration**

   ```bash
   ./scripts/auto-patch-compose.sh
   ```

   This ensures `docker-compose.yml` has proper security settings.

2. **Download Models**

   ```bash
   ./scripts/auto-refresh-models.sh
   ```

3. **Start Services**
   ```bash
   docker-compose up -d
   ```

## Development

### Pre-commit Hooks

Install pre-commit hooks to enforce code quality:

```bash
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Integration tests with isolated environment
invoke test-integration

# Security tests only
invoke test-security

# All tests with coverage
invoke test-all
```

## Architecture

- **API Gateway**: JWT-authenticated proxy with path validation
- **Model Orchestrator**: Dynamic model lifecycle management
- **vLLM Services**:
  - Mistral 7B AWQ (general chat)
  - LLaVA 7B (vision/multimodal)
  - DeepSeek Coder GPTQ (code generation)

## Security

- JWT authentication required for all API calls
- Path traversal protection in API gateway
- Secure environment variable handling
- GPU memory limits to prevent OOM

## Scripts

| Script                   | Purpose                                  |
| ------------------------ | ---------------------------------------- |
| `auto-patch-compose.sh`  | Validates and patches docker-compose.yml |
| `auto-refresh-models.sh` | Downloads and validates model files      |
| `assert_models.sh`       | Verifies model directory structure       |

## Monitoring

Check GPU memory usage:

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Troubleshooting

- **yq not found**: Install yq v4.40+ (see Prerequisites)
- **HF_TOKEN error**: Set HF_TOKEN environment variable
- **GPU OOM**: Models are configured with conservative memory limits
