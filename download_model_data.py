import os
from huggingface_hub import snapshot_download
from datasets import load_dataset

# =========================
# CONFIG
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN")  # must be set

MODEL_ID = "bharatgenai/Param2-17B-A2.4B-Thinking"
MODEL_DIR = "/home/kishoryd/LLM_Bench/data/Param2-17B"

DATASET_ID = "ai4bharat/MILU"
DATASET_DIR = "/home/kishoryd/LLM_Bench/data/milu"

LANGUAGES = [
    "English", "Hindi", "Bengali", "Tamil", "Telugu"
]

SPLIT = "test"   # or "validation"

# =========================
# CHECK TOKEN
# =========================
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not set. Export it before running.")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# =========================
# DOWNLOAD MODEL
# =========================
print("\n=== Downloading Model ===")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_DIR,
    token=HF_TOKEN,
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"Model saved to: {MODEL_DIR}")

# =========================
# DOWNLOAD DATASET
# =========================
print("\n=== Downloading Dataset ===")

for lang in LANGUAGES:
    print(f"\nDownloading {lang}...")

    ds = load_dataset(
        DATASET_ID,
        data_dir=lang,
        split=SPLIT,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    save_path = os.path.join(DATASET_DIR, lang)
    ds.save_to_disk(save_path)

    print(f"Saved: {save_path}")

print("\n✅ All downloads complete.")
