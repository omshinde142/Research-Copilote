import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        env_path = Path.home() / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break

    if not token:
        print("ERROR: HF_TOKEN missing.")
        sys.exit(1)

    api = HfApi(token=token)
    username = api.whoami()["name"]
    repo_id = f"{username}/research-copilot-training"

    files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))

    has_embedder = any("embedder-ft" in f for f in files)
    has_scibert = "scibert_metrics.json" in files
    has_gbdt = "gbdt_metrics.json" in files
    has_smi = "nvidia_smi.txt" in files

    print("=" * 65)
    print(f"HUGGING FACE REPO STATUS: {repo_id}")
    print("=" * 65)
    print(f"  [GPU Check]     nvidia_smi.txt     : {'UPLOADED (GPU Active)' if has_smi else 'Pending'}")
    print(f"  [Job 1/3]       embedder-ft        : {'FINISHED & UPLOADED' if has_embedder else 'In progress / Pending'}")
    print(f"  [Job 2/3]       scibert_metrics    : {'FINISHED & UPLOADED' if has_scibert else 'In progress / Pending'}")
    print(f"  [Job 3/3]       gbdt_metrics       : {'FINISHED & UPLOADED' if has_gbdt else 'In progress / Pending'}")
    print("-" * 65)

    if has_embedder and has_scibert and has_gbdt:
        print("ALL THREE JOBS FINISHED ON GPU!")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
