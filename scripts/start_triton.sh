#!/bin/bash
# ─── Launch Triton Inference Server ──────────────────────────────
#
# Usage:
#   ./scripts/start_triton.sh /path/to/triton_models
#
# Defaults to HTTP 8000, gRPC 8001. Override with env vars.
# ──────────────────────────────────────────────────────────────────

set -e

MODEL_REPO="${1:?Usage: $0 <model_repository_path>}"
HTTP_PORT="${TRITON_HTTP_PORT:-8000}"
GRPC_PORT="${TRITON_GRPC_PORT:-8001}"

echo "──────────────────────────────────────────"
echo "  Starting Triton Inference Server"
echo "  Model repo : ${MODEL_REPO}"
echo "  HTTP port  : ${HTTP_PORT}"
echo "  gRPC port  : ${GRPC_PORT}"
echo "──────────────────────────────────────────"

# Using Docker (recommended)
docker run --rm --gpus all \
    -p "${HTTP_PORT}:8000" \
    -p "${GRPC_PORT}:8001" \
    -v "${MODEL_REPO}:/models" \
    nvcr.io/nvidia/tritonserver:latest \
    tritonserver --model-repository=/models
