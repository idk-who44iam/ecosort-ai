
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/data/ollama \
    ECOSORT_INSTANCE_PATH=/data/instance

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl zstd \
    && rm -rf /var/lib/apt/lists/* \
    && bash -o pipefail -c "curl -fsSL https://ollama.com/install.sh | sh"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py classifier.py ./
COPY templates ./templates
COPY static ./static
COPY start.sh ./start.sh

RUN sed -i 's/\r$//' ./start.sh \
    && echo "===== VERIFY START.SH =====" \
    && file /app/start.sh \
    && head -n 1 /app/start.sh | od -An -tx1 \
    && echo "===== END VERIFICATION =====" \
    && chmod +x ./start.sh

EXPOSE 8080

CMD ["bash", "/app/start.sh"]
