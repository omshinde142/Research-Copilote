import os
import sys
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    model_path = Path("data/models/embedder-ft")
    if not model_path.exists():
        print(f"ERROR: {model_path} does not exist.")
        sys.exit(1)

    papers_path = Path("data/papers.pkl")
    if not papers_path.exists():
        print(f"ERROR: {papers_path} does not exist.")
        sys.exit(1)

    df = pd.read_pickle(papers_path)
    print(f"Loaded {len(df)} papers from {papers_path}.")

    print(f"Loading fine-tuned model from {model_path}...")
    model = SentenceTransformer(str(model_path))

    text = (df.title + ". " + df.abstract).tolist()
    print(f"Encoding {len(text)} papers with fine-tuned model...")
    emb = model.encode(text, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    print("Building FAISS HNSW vector index for fine-tuned embedder...")
    index = faiss.IndexHNSWFlat(emb.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 100
    index.add(emb.astype("float32"))

    out_file = Path("data/index_ft.faiss")
    faiss.write_index(index, str(out_file))
    print(f"Successfully wrote {index.ntotal} vectors to {out_file} (size: {out_file.stat().st_size / 1e6:.1f} MB).")

if __name__ == "__main__":
    main()
