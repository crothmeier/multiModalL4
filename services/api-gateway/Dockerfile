FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi==0.109.0 httpx==0.26.0 uvicorn==0.27.0

# Non-root user
RUN groupadd -g 1001 api && useradd -u 1001 -g api -m -s /bin/bash api

COPY --chown=1001:1001 app/ /app/

USER 1001

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
