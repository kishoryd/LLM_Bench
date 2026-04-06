import argparse
import csv
import os
import random
import time

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils.argparse_utils import FlexibleArgumentParser

SYSTEM_PROMPT = "You are a helpful assistant."

LANGUAGES_TO_SAMPLE = ["English", "Hindi"]

OPTION_LABELS = ["A", "B", "C", "D"]

def format_milu_prompt(row: dict) -> str:
    question = row["question"].strip()
    options  = row["options"]

    lines = [
        "The following is a multiple choice question.",
        question,
        "",
    ]

    for label, opt in zip(OPTION_LABELS, options):
        lines.append(f"{label}) {opt.strip()}")

    lines += ["", "Answer:"]
    return "\n".join(lines)


def load_milu_prompts(
    languages: list[str],
    num_samples: int,
    seed: int = 42,
) -> list[str]:

    random.seed(seed)
    per_lang = max(num_samples // len(languages), 1)

    all_prompts: list[str] = []

    for lang in languages:
        print(f"Loading LOCAL MILU / {lang} ...")

        dataset_path = f"/home/kishoryd/LLM_Bench/data/milu/{lang}"

        if not os.path.exists(dataset_path):
            print(f"[WARNING] Missing dataset: {dataset_path}")
            continue

        try:
            ds = load_from_disk(dataset_path)
        except Exception as exc:
            print(f"[WARNING] Could not load {lang}: {exc}")
            continue

        rows = list(ds)
        if not rows:
            print(f"[WARNING] Empty dataset: {lang}")
            continue

        # deterministic sampling for stable benchmarking
        sampled = rows[:per_lang]

        for row in sampled:
            try:
                all_prompts.append(format_milu_prompt(row))
            except Exception:
                continue

    if not all_prompts:
        raise RuntimeError("No prompts loaded from local dataset.")

    random.shuffle(all_prompts)
    print(f"Total prompts loaded: {len(all_prompts)}")

    return all_prompts


def build_inputs(
    tokenizer,
    raw_prompts: list[str],
    batch_size: int,
    input_len: int,
    seed: int = 42,
):

    random.seed(seed)
    selected = raw_prompts[:batch_size]

    inputs = []

    for text in selected:
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ]

        token_ids = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors=None,
        )

        if len(token_ids) >= input_len:
            token_ids = token_ids[-input_len:]
        else:
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            token_ids = [pad_id] * (input_len - len(token_ids)) + token_ids

        inputs.append({"prompt_token_ids": token_ids})

    return inputs



def build_llm(args):
    return LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )


def run_benchmark(llm, dummy_inputs, args):

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    # Benchmark
    latencies = []

    for _ in range(args.num_iters):
        t0 = time.perf_counter()
        llm.generate(dummy_inputs, sampling_params=sampling_params)
        latencies.append(time.perf_counter() - t0)

    mean_lat = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    throughput = args.batch_size * (args.input_len + args.output_len) / mean_lat

    return mean_lat, p50, p99, throughput


def write_csv(args, mean_lat, p50, p99, throughput):

    csv_file = "LLM_Inference_Bench_vLLM_throughput.csv"

    row = [
        args.tensor_parallel_size,
        args.input_len,
        args.batch_size,
        round(mean_lat, 4),
        round(p50, 4),
        round(p99, 4),
        round(throughput, 2),
    ]

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main(args):

    raw_prompts = load_milu_prompts(
        LANGUAGES_TO_SAMPLE,
        num_samples=max(args.batch_size * 10, 500),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    dummy_inputs = build_inputs(
        tokenizer,
        raw_prompts,
        args.batch_size,
        args.input_len,
    )

    llm = build_llm(args)

    mean_lat, p50, p99, throughput = run_benchmark(
        llm, dummy_inputs, args
    )

    write_csv(args, mean_lat, p50, p99, throughput)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--input-len", type=int)
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num-iters-warmup", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)

    args = parser.parse_args()
    main(args)