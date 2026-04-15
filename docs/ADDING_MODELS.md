# Adding a New Model

## Steps

### 1. Create the config file

Create `configs/models/<your_model>.yaml`:

```yaml
name: "Your-Model-Name"
path: "/absolute/path/to/model/weights"
max_context: 8192
default_dtype: "bf16"
trust_remote_code: false

# Optional: model-specific prompt seeds
prompts:
  custom_seed: >
    Your custom seed text for benchmarking prompts.
```

### 2. Run the benchmark

```bash
python scripts/run_bench.py --model your_model --backend native
```

That's it. The model registry auto-discovers any `.yaml` file in `configs/models/`.

### 3. Verify it's detected

```bash
python scripts/run_bench.py --list-models
```

## Config fields

| Field              | Required | Description                                   |
|--------------------|----------|-----------------------------------------------|
| `name`             | yes      | Display name for reports                      |
| `path`             | yes      | Absolute path to model weights                |
| `max_context`      | yes      | Max context window in tokens                  |
| `default_dtype`    | no       | `bf16`, `fp16`, `fp32` (default: auto)        |
| `trust_remote_code`| no       | Allow custom model code (default: false)      |
| `ollama_name`      | no       | Model name for Ollama backend (if different)  |
| `prompts`          | no       | Custom seed texts for prompt construction     |

## Notes

- Model weights must be downloaded locally — the benchmark runs offline.
- For Ollama, pull the model first: `ollama pull <model_name>`
- For Triton, you'll also need to set up the model repository structure.
