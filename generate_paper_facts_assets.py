#!/usr/bin/env python3
"""
generate_paper_facts_assets.py
Generates all charts, captures app screenshots via headless Chrome,
and copies all existing result assets into paper_facts/images/.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    base_dir = Path("/Users/durveshnarkhede/Desktop/COPILOTE")
    out_dir = base_dir / "paper_facts" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = base_dir / "data"

    print("Generating assets into:", out_dir)

    # 1. Copy existing confusion matrix and Colab screenshot
    cm_src = base_dir / "results" / "confusion_matrix.png"
    if cm_src.exists():
        shutil.copy(cm_src, out_dir / "confusion_matrix.png")
        print("Copied confusion_matrix.png")

    colab_src = Path("/Users/durveshnarkhede/.gemini/antigravity/brain/dde1e24e-ca80-411f-8260-5c3375051e65/colab_training_active.png")
    if colab_src.exists():
        shutil.copy(colab_src, out_dir / "colab_training_run.png")
        print("Copied colab_training_run.png")

    # 2. Generate Chart: Papers Per Month (2020–2026)
    rep = json.loads((data_dir / "data_report.json").read_text())
    m_counts = rep.get("monthly_counts_2023_to_2026", {})
    if m_counts:
        plt.figure(figsize=(12, 4.5), dpi=300)
        months = list(m_counts.keys())
        counts = list(m_counts.values())
        plt.bar(months, counts, color="#4f46e5", edgecolor="#312e81", alpha=0.85)
        plt.title("arXiv Academic Paper Sampling per Month (September 2023 – October 2026)", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Submission Month (YYYY-MM)", fontsize=10, labelpad=8)
        plt.ylabel("Number of Retained Papers", fontsize=10, labelpad=8)
        plt.xticks(rotation=65, ha="right", fontsize=8)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.ylim(0, 350)
        plt.axhline(300, color="red", linestyle=":", alpha=0.7, label="Target Ceiling (300/mo)")
        plt.legend(frameon=True, facecolor="#ffffff", loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / "papers_per_month.png")
        plt.close()
        print("Generated papers_per_month.png")

    # 3. Generate Chart: Class Distribution across 35 Categories
    df = pd.read_pickle(data_dir / "papers.pkl")
    cat_counts = df.primary_category.value_counts()
    plt.figure(figsize=(14, 5.5), dpi=300)
    plt.bar(cat_counts.index, cat_counts.values, color="#059669", edgecolor="#064e3b", alpha=0.85)
    plt.title("Distribution of 16,417 arXiv Papers Across 35 Computer Science Disciplines", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Primary Category (arXiv Taxonomy)", fontsize=10, labelpad=8)
    plt.ylabel("Paper Count", fontsize=10, labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png")
    plt.close()
    print("Generated class_distribution.png")

    # 4. Generate Chart: Model Ladder Comparison (Rolling vs Random vs Test)
    models = ["Naive Bayes", "Random Forest", "k-NN (k=15)", "PyTorch MLP", "Linear SVM", "Logistic Reg.", "Voting Ens."]
    rolling_acc = [43.43, 57.89, 63.91, 63.89, 66.19, 66.25, 66.78]
    random_acc = [45.99, 60.53, 67.20, 67.04, 68.46, 69.04, 69.17]
    test_acc = [38.54, 55.28, 61.24, 62.58, 63.60, 64.61, 65.62]

    x = np.arange(len(models))
    width = 0.26

    plt.figure(figsize=(13, 5), dpi=300)
    plt.bar(x - width, rolling_acc, width, label="Rolling-Origin CV (Historic)", color="#6366f1", alpha=0.85)
    plt.bar(x, random_acc, width, label="Random 5-Fold CV (Temporal Leakage)", color="#ef4444", alpha=0.85)
    plt.bar(x + width, test_acc, width, label="Test Holdout (Aug–Oct 2026)", color="#10b981", alpha=0.85)
    plt.title("Classification Accuracy Ladder: Impact of Temporal Data Leakage vs. Honest Test Holdout", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Model Architecture", fontsize=10, labelpad=8)
    plt.ylabel("Classification Accuracy (%)", fontsize=10, labelpad=8)
    plt.xticks(x, models, rotation=15, ha="right", fontsize=9)
    plt.ylim(30, 75)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="#ffffff", loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "classification_ladder_comparison.png")
    plt.close()
    print("Generated classification_ladder_comparison.png")

    # 5. Generate Chart: Search Benchmark Retrieval Quality (Base vs Fine-Tuned)
    bench_names = ["Validation (Recall@10)", "Validation (MRR@10)", "Validation (nDCG@10)",
                   "Test Holdout (Recall@10)", "Test Holdout (MRR@10)", "Test Holdout (nDCG@10)",
                   "BEIR SciFact (Recall@10)", "BEIR SciFact (MRR@10)", "BEIR SciFact (nDCG@10)"]
    base_scores = [95.77, 87.37, 89.41, 93.93, 82.99, 85.64, 78.33, 60.47, 64.51]
    ft_scores = [98.78, 95.01, 95.93, 98.31, 93.38, 94.60, 77.83, 59.45, 63.59]

    x = np.arange(len(bench_names))
    width = 0.35

    plt.figure(figsize=(15, 5.5), dpi=300)
    plt.bar(x - width/2, base_scores, width, label="Pre-Trained Base (all-MiniLM-L6-v2)", color="#94a3b8", alpha=0.85)
    plt.bar(x + width/2, ft_scores, width, label="Fine-Tuned Embedder (embedder-ft)", color="#2563eb", alpha=0.85)
    plt.title("Information Retrieval Quality: In-Domain Contrastive Fine-Tuning Gains and SciFact Generalization", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Score (%) / Metric x 100", fontsize=10, labelpad=8)
    plt.xticks(x, bench_names, rotation=35, ha="right", fontsize=8.5)
    plt.ylim(50, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.axvline(2.5, color="#cbd5e1", linestyle="--")
    plt.axvline(5.5, color="#cbd5e1", linestyle="--")
    plt.text(1, 101, "Validation Set", ha="center", fontweight="bold", color="#334155")
    plt.text(4, 101, "Test Holdout", ha="center", fontweight="bold", color="#334155")
    plt.text(7, 101, "BEIR SciFact (Zero-Shot)", ha="center", fontweight="bold", color="#334155")
    plt.legend(frameon=True, facecolor="#ffffff", loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "search_benchmark_comparison.png")
    plt.close()
    print("Generated search_benchmark_comparison.png")

    # 6. Capture Screenshots of Running App via Headless Chrome
    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if os.path.exists(chrome_bin):
        print("Capturing live app screenshots from http://127.0.0.1:8000...")
        # 1. Main Map Overview
        subprocess.run([
            chrome_bin,
            "--headless",
            "--disable-gpu",
            "--window-size=1440,900",
            "--virtual-time-budget=4000",
            f"--screenshot={out_dir / 'app_map_overview.png'}",
            "http://127.0.0.1:8000"
        ], capture_output=True)
        print("Captured app_map_overview.png")

        # Create a small HTML driver to display search results, ask citations, and model comparison panel
        driver_html = base_dir / "test_driver.html"
        driver_html.write_text("""<!DOCTYPE html>
<html>
<head><meta http-equiv="refresh" content="0; url=http://127.0.0.1:8000"></head>
<body></body>
</html>""")

if __name__ == "__main__":
    main()
