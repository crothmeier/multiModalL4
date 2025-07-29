FROM vllm/vllm-openai:v0.6.3.post1
RUN pip install --no-cache-dir pillow==11.0.0 torchvision==0.21.0 transformers==4.42.0
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8001/v1/models || exit 1
