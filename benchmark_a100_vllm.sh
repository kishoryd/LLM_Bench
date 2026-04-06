#!/bin/bash -l
#SBATCH --job-name=vllm_benchmark
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --reservation=chatbot

# ========================
# ENVIRONMENT SETUP
# ========================
module load miniconda
conda activate vLLM_A100

# Temp directories
export TMPDIR="/home/kishoryd/LLM_Bench/tmp"
export RAY_TMPDIR="/home/kishoryd/LLM_Bench/tmp/ray"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

# Make sure the logs dir exists before SLURM tries to write to it
mkdir -p logs

# HuggingFace auth

# Local cache + offline mode
export HF_HOME="/home/kishoryd/LLM_Bench/hf_cache"

export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Networking / NCCL
export NCCL_DEBUG=INFO

# Required for vLLM multi-GPU
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ========================
# CONFIG
# ========================
MODEL_NAME="/home/kishoryd/LLM_Bench/data/Param2-17B"

NUM_WARMUP=3
NUM_ITERS=10
DOWNLOAD_DIR="/home/kishoryd/LLM_Bench/data"
MILU_SPLIT="test"

# ========================
# RUN BENCHMARK
# ========================
for tensor_parallel in 1 2; do
    for precision in float16; do
        for batch_size in 1 16 32 64; do
            for seq_len in 128 256 512 1024 2048 4096; do

                echo ""
                echo "============================================================"
                echo "  TP=$tensor_parallel | BS=$batch_size | SEQ=$seq_len | dtype=$precision"
                echo "============================================================"

                python3 benchmark_a100_vllm.py \
                    --model                  "$MODEL_NAME" \
                    --tensor-parallel-size   "$tensor_parallel" \
                    --input-len              "$seq_len" \
                    --output-len             "$seq_len" \
                    --batch-size             "$batch_size" \
                    --dtype                  "$precision" \
                    --num-iters-warmup       "$NUM_WARMUP" \
                    --num-iters              "$NUM_ITERS" \
                    --gpu-memory-utilization 0.9 \
                    --max-model-len          4096

                if [ $? -ne 0 ]; then
                    echo "ERROR: benchmark failed for TP=$tensor_parallel BS=$batch_size SEQ=$seq_len — aborting."
                    exit 1
                fi

            done
        done
    done
done

echo ""
echo "All benchmarks complete."
echo "Results saved to: LLM_Inference_Bench_vLLM_throughput.csv"
