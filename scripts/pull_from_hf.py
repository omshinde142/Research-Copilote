import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

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
        print("ERROR: HF_TOKEN environment variable is missing.")
        sys.exit(1)

    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    repo_name = "research-copilot-training"
    repo_id = f"{username}/{repo_name}"

    print(f"Downloading trained models and results from private repo: {repo_id}...")
    local_dir = Path("downloaded_gpu_run")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token
    )

    # Sync into project structure
    if (local_dir / "embedder-ft").exists():
        import shutil
        target = Path("data/models/embedder-ft")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(local_dir / "embedder-ft", target)
        print(f"Synced embedder model to {target}")

    for fname in ["scibert_metrics.json", "gbdt_metrics.json"]:
        src = local_dir / fname
        if src.exists():
            import shutil
            shutil.copy(src, Path("data") / fname)
            print(f"Synced {fname} to data/{fname}")

    if (local_dir / "search_benchmark_gpu.md").exists():
        import shutil
        shutil.copy(local_dir / "search_benchmark_gpu.md", "results/search_benchmark.md")
        print("Synced search_benchmark_gpu.md to results/search_benchmark.md")

    if (local_dir / "nvidia_smi.txt").exists():
        import shutil
        shutil.copy(local_dir / "nvidia_smi.txt", "results/nvidia_smi.txt")
        print("Synced nvidia_smi.txt to results/nvidia_smi.txt")

    print("\nAll GPU training results successfully downloaded into project!")

if __name__ == "__main__":
    main()
