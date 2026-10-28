import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

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

    print(f"Authenticated as Hugging Face user: {username}")
    print(f"Creating/verifying private dataset repo: {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)

    files_to_upload = [
        "data/papers.pkl",
        "data/splits.json",
        "data/index.faiss",
        "prepare.py",
        "train_embedder.py",
        "train_scibert.py",
        "train_gbdt.py",
        "colab_run.py",
        "config.yaml",
        "requirements.txt"
    ]

    for fpath in files_to_upload:
        p = Path(fpath)
        if not p.exists():
            print(f"Warning: {fpath} does not exist, skipping.")
            continue
        print(f"Uploading {fpath} -> {fpath} in {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=fpath,
            repo_id=repo_id,
            repo_type="dataset"
        )

    # Upload scifact folder if exists
    scifact_dir = Path("data/scifact")
    if scifact_dir.exists():
        print(f"Uploading {scifact_dir} -> data/scifact in {repo_id}...")
        api.upload_folder(
            folder_path=str(scifact_dir),
            path_in_repo="data/scifact",
            repo_id=repo_id,
            repo_type="dataset"
        )

    print("\nDataset and scripts successfully pushed to private repo!")
    print(f"REPO_ID: {repo_id}")

if __name__ == "__main__":
    main()
