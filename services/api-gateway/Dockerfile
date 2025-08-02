FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    httpx==0.26.0 \
    python-jose[cryptography]==3.3.0

COPY . .

CMD ["python", "main.py"]
