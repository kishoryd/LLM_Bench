#!/bin/bash
# ─── Launch vLLM Server ──────────────────────────────────────────
#
# Usage:
#   ./scripts/start_vllm.sh /path/to/model
#   ./scripts/start_vllm.sh /path/to/model --tensor-parallel-size 2
#
# Defaults to port 8000. Override with VLLM_PORT env var.
# ──────────────────────────────────────────────────────────────────

set -e

MODEL_PATH="${1:?Usage: $0 <model_path> [extra vllm args...]}"
shift
PORT="${VLLM_PORT:-8000}"

echo "──────────────────────────────────────────"
echo "  Starting vLLM server"
echo "  Model : ${MODEL_PATH}"
echo "  Port  : ${PORT}"
echo "  Extra : $@"
echo "──────────────────────────────────────────"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    "$@"
