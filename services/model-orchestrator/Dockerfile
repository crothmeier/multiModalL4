FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck and docker CLI
RUN apt-get update && \
    apt-get install -y curl ca-certificates gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator_cli.py .
COPY orchestrator_simple.py .

# Run as root to access Docker socket
USER root

# Set Docker host to use the socket
ENV DOCKER_HOST=unix:///var/run/docker.sock

CMD ["python", "orchestrator_simple.py"]
