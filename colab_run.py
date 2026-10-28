import os
import subprocess
import sys
import time
from pathlib import Path

def run_cmd(cmd, desc):
    print(f"\n{'='*70}\n[STEP] {desc}\nCOMMAND: {cmd}\n{'='*70}")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, text=True)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"FAILED: {desc} exited with code {res.returncode} in {elapsed:.1f}s")
        sys.exit(res.returncode)
    print(f"COMPLETED: {desc} in {elapsed:.1f}s")

def upload_artifact(local_path, repo_path, token, repo_id):
    if not token or not repo_id:
        print(f"Skipping upload for {repo_path} (HF_TOKEN or HF_REPO_ID not set).")
        return
    p = Path(local_path)
    if not p.exists():
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        if p.is_dir():
            print(f"Uploading directory {p} -> {repo_path} in {repo_id}...")
            api.upload_folder(
                folder_path=str(p),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset"
            )
        else:
            print(f"Uploading file {p} -> {repo_path} in {repo_id}...")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset"
            )
        print(f"Successfully uploaded {repo_path} to {repo_id}.")
    except Exception as exc:
        print(f"Warning: Failed to upload {repo_path} to {repo_id}: {exc}")

def main():
    print("=" * 70)
    print("        RESEARCH COPILOT - RESUMABLE GPU TRAINING WORKFLOW")
    print("=" * 70)

    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")

    # 1. Check GPU & log status
    run_cmd("nvidia-smi > nvidia_smi.txt && cat nvidia_smi.txt", "Verify NVIDIA GPU & Drivers")
    upload_artifact("nvidia_smi.txt", "nvidia_smi.txt", token, repo_id)

    # 2. Install dependencies
    run_cmd(
        "pip install -q --upgrade pandas transformers datasets sentence-transformers accelerate lightgbm xgboost faiss-cpu scikit-learn huggingface_hub",
        "Install Required ML & Evaluation Packages"
    )

    # --------------------------------------------------------------------------
    # Job 1: Contrastive Embedder Fine-Tuning
    # --------------------------------------------------------------------------
    embedder_ckpt = Path("data/models/embedder-ft/model.safetensors")
    if embedder_ckpt.exists() and embedder_ckpt.stat().st_size > 10_000_000:
        print("\n[RESUME CHECK] Job 1 (Contrastive Embedder) already completed. Skipping training.")
    else:
        run_cmd("python train_embedder.py --mode full", "Job 1/3: Contrastive Embedder Fine-Tuning (--mode full)")
        print("\n[CHECKPOINT] Job 1 finished. Uploading immediately to Hugging Face...")
        upload_artifact("data/models/embedder-ft", "embedder-ft", token, repo_id)
        upload_artifact("results/search_benchmark.md", "search_benchmark_gpu.md", token, repo_id)

    # --------------------------------------------------------------------------
    # Job 2: SciBERT Sequence Classifier Fine-Tuning
    # --------------------------------------------------------------------------
    scibert_metrics = Path("data/scibert_metrics.json")
    if scibert_metrics.exists() and scibert_metrics.stat().st_size > 20:
        print("\n[RESUME CHECK] Job 2 (SciBERT Classifier) already completed. Skipping training.")
    else:
        run_cmd("python train_scibert.py --mode full --epochs 4", "Job 2/3: SciBERT Fine-Tuning with Early Stopping (--mode full)")
        print("\n[CHECKPOINT] Job 2 finished. Uploading immediately to Hugging Face...")
        upload_artifact("data/scibert_metrics.json", "scibert_metrics.json", token, repo_id)
        if Path("out_scibert").exists():
            upload_artifact("out_scibert", "scibert-model", token, repo_id)

    # --------------------------------------------------------------------------
    # Job 3: Native LightGBM and XGBoost Classifiers
    # --------------------------------------------------------------------------
    gbdt_metrics = Path("data/gbdt_metrics.json")
    if gbdt_metrics.exists() and gbdt_metrics.stat().st_size > 20:
        print("\n[RESUME CHECK] Job 3 (LightGBM & XGBoost) already completed. Skipping training.")
    else:
        run_cmd("python train_gbdt.py", "Job 3/3: LightGBM and XGBoost Classifiers on Full Splits")
        print("\n[CHECKPOINT] Job 3 finished. Uploading immediately to Hugging Face...")
        upload_artifact("data/gbdt_metrics.json", "gbdt_metrics.json", token, repo_id)

    print("\n" + "=" * 70)
    print("ALL THREE TRAINING RUNS COMPLETED & SYNCED TO HUGGING FACE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
