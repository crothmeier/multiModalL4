FROM vllm/vllm-openai:v0.6.3.post1
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8001/v1/models || exit 1
