#!/bin/bash
# ─── Launch SGLang Server ─────────────────────────────────────────
#
# Usage:
#   ./scripts/start_sglang.sh /path/to/model
#   ./scripts/start_sglang.sh /path/to/model --tp 2
#
# Defaults to port 30000. Override with SGLANG_PORT env var.
# ──────────────────────────────────────────────────────────────────

set -e

MODEL_PATH="${1:?Usage: $0 <model_path> [extra sglang args...]}"
shift
PORT="${SGLANG_PORT:-30000}"

echo "──────────────────────────────────────────"
echo "  Starting SGLang server"
echo "  Model : ${MODEL_PATH}"
echo "  Port  : ${PORT}"
echo "  Extra : $@"
echo "──────────────────────────────────────────"

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --port "${PORT}" \
    --trust-remote-code \
    "$@"
