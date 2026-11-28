
#!/usr/bin/env bash
set -Eeuo pipefail

DATA_DIR="${RAILWAY_VOLUME_MOUNT_PATH:-/data}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$DATA_DIR/ollama}"
export ECOSORT_INSTANCE_PATH="${ECOSORT_INSTANCE_PATH:-$DATA_DIR/instance}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export PORT="${PORT:-8080}"
MODEL_TAG="ahmadwaqar/smolvlm2-500m-video"

mkdir -p "$OLLAMA_MODELS" "$ECOSORT_INSTANCE_PATH"

ollama serve &
OLLAMA_PID=$!
PULL_PID=""
WEB_PID=""

cleanup() {
    for pid in "$PULL_PID" "$WEB_PID" "$OLLAMA_PID"; do
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

for attempt in $(seq 1 60); do
    if curl --silent --fail http://127.0.0.1:11434/api/version >/dev/null; then
        break
    fi

    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        wait "$OLLAMA_PID"
    fi

    sleep 1
done

if ! curl --silent --fail http://127.0.0.1:11434/api/version >/dev/null; then
    echo "Ollama did not become ready within 60 seconds." >&2
    exit 1
fi

(
    if ollama list | awk 'NR > 1 { print $1 }' | grep -Fxq "${MODEL_TAG}:latest"; then
        echo "Model ${MODEL_TAG}:latest is already available."
    else
        echo "Downloading ${MODEL_TAG} into persistent storage."
        ollama pull "$MODEL_TAG"
    fi
) &
PULL_PID=$!

gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers 1 \
    --threads 4 \
    --timeout 240 \
    --access-logfile - \
    --error-logfile - \
    app:app &
WEB_PID=$!

if wait -n "$OLLAMA_PID" "$WEB_PID"; then
    status=0
else
    status=$?
fi

exit "$status"

