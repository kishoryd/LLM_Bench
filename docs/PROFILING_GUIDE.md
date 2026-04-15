# Profiling Guide

## PyTorch Profiler

Best for: operator-level breakdown, memory profiling, FLOPS counting.

### Quick start

```bash
python scripts/run_bench.py --backend native --model param2_17b --profiler pytorch
```

### What it captures

- CPU and CUDA operator timings
- Memory allocation/deallocation
- Call stacks
- FLOPS per operator
- Chrome trace (viewable in chrome://tracing)
- TensorBoard logs

### Output files

```
results/torch_profiler/
├── short_1_trace.json       # Chrome trace
├── medium_1_trace.json
└── ...
results/tb_logs/             # TensorBoard logs
```

### Viewing traces

Chrome trace:
1. Open Chrome → navigate to `chrome://tracing`
2. Click "Load" → select the `.json` trace file

TensorBoard:
```bash
pip install tensorboard torch-tb-profiler
tensorboard --logdir results/tb_logs/
```

### Config options

Edit `configs/profiles/pytorch_profiler.yaml`:

```yaml
schedule:
  wait: 1          # skip N steps before recording
  warmup: 1        # warmup steps (not recorded)
  active: 3        # steps to record
  repeat: 1

activities: ["cpu", "cuda"]
record_shapes: true
profile_memory: true
with_stack: true
with_flops: true
```

---

## NVIDIA Nsight Systems

Best for: GPU timeline, CUDA API calls, kernel launches, CPU-GPU sync.

### Quick start

```bash
# Generate the nsys command
python scripts/run_bench.py --backend native --model param2_17b --profiler nsight

# Or run directly
nsys profile \
  --trace=cuda,nvtx,osrt \
  --duration=60 \
  --output=results/nsight/my_profile \
  --stats=true \
  python scripts/run_bench.py --backend native --model param2_17b
```

### Viewing reports

```bash
# Open in Nsight Systems GUI
nsys-ui results/nsight/my_profile.nsys-rep

# CLI stats summary
nsys stats results/nsight/my_profile.nsys-rep
```

### What to look for

- **CUDA API timeline**: are kernels launching efficiently?
- **GPU idle gaps**: where is the GPU waiting?
- **Memory copies**: how much time in HtoD / DtoH transfers?
- **Kernel durations**: which kernels dominate?

---

## NVIDIA Nsight Compute

Best for: deep kernel-level analysis, occupancy, memory throughput.

### Quick start

```bash
ncu --output results/ncu/kernel_profile \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
  python scripts/run_bench.py --backend native --model param2_17b --batch-sizes 1
```

### Warning

Nsight Compute is very slow — it replays kernels multiple times. Use it selectively on specific batch sizes, not full sweeps.

### Viewing reports

```bash
ncu-ui results/ncu/kernel_profile.ncu-rep
```

---

## Tips

- Profile with `--batch-sizes 1` first to isolate single-request behaviour.
- PyTorch Profiler is lightweight — use it for every run.
- Nsight Systems is for timeline analysis — use when investigating latency.
- Nsight Compute is heavy — use only when optimising specific kernels.
- Compare profiles across backends to see where each spends time.
