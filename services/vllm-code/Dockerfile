FROM vllm/vllm-openai:v0.6.3.post1

# Non-root user
RUN groupadd -g 1001 vllm && useradd -u 1001 -g vllm -m -s /bin/bash vllm

# Health check script
RUN printf '#!/bin/sh\ncurl -f http://localhost:8000/v1/models || exit 1\n' > /healthcheck.sh && \
    chmod +x /healthcheck.sh

USER 1001
STOPSIGNAL SIGINT

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD ["/healthcheck.sh"]

LABEL org.opencontainers.image.description="Requires runtime cap_drop=ALL cap_add=NET_BIND_SERVICE"

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
