# Research Copilot - Search Model Fine-Tuning & Retrieval Benchmark

This benchmark measures the retrieval quality of **our fine-tuned search model** compared against the original pre-trained `all-MiniLM-L6-v2` checkpoint.

### Fine-Tuning Setup:
- **Base Architecture**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Training Pairs**: 14629 (Title $\rightarrow$ Abstract) pairs strictly from `splits['train']`
- **Objective Function**: `MultipleNegativesRankingLoss` (MNRL) with in-batch negatives + mined hard negatives
- **Hard Negative Strategy**: Drawn strictly from the train-split within the same topic cluster, excluding the **top 5 nearest neighbors** to avoid false negatives
- **Training Time**: 812.0s (3 epoch(s))
- **Artifacts Saved**: [`data/models/embedder-ft`](file://data/models/embedder-ft)

---

## 1. Corpus Retrieval Benchmark (Title $\rightarrow$ Abstract)

Evaluated on 5,022 corpus abstracts across the **Validation Set (May–Jul 2026)** and **Unseen Holdout Test Set (Aug–Oct 2026)**:

| Evaluation Split | Checkpoint | Recall@10 (95% CI) | MRR@10 (95% CI) | nDCG@10 (95% CI) |
|:---|:---|:---:|:---:|:---:|
| **Validation Set** (898 queries) | Pre-trained Base | 95.77% [94.4%, 97.0%] | 0.8737 [0.8542, 0.8922] | 0.8941 [0.8774, 0.9102] |
| **Validation Set** (898 queries) | Model (full) | 98.78% [98.0%, 99.4%] | 0.9501 [0.9377, 0.9619] | 0.9593 [0.9486, 0.9692] |
| **Test Holdout** (890 queries) | Pre-trained Base | 93.93% [92.4%, 95.4%] | 0.8299 [0.8077, 0.8511] | 0.8564 [0.8366, 0.8750] |
| **Test Holdout** (890 queries) | Model (full) | 98.31% [97.4%, 99.1%] | 0.9338 [0.9195, 0.9473] | 0.9460 [0.9336, 0.9574] |

---

## 2. External Out-of-Corpus Benchmark: BEIR SciFact TEST Split

To verify whether representation quality changes outside our arXiv corpus, we evaluated on the official **BEIR SciFact TEST split** (300 test claims against 5183 scientific abstracts with `title + text`):

| Benchmark | Checkpoint | Recall@10 (95% CI) | MRR@10 (95% CI) | nDCG@10 (95% CI) |
|:---|:---|:---:|:---:|:---:|
| **SciFact (BEIR Test Split)** | Pre-trained Base | 78.33% [73.6%, 82.9%] | 0.6047 [0.5557, 0.6531] | 0.6451 [0.5995, 0.6907] |
| **SciFact (BEIR Test Split)** | Model (full) | 77.83% [73.1%, 82.5%] | 0.5945 [0.5456, 0.6425] | 0.6359 [0.5899, 0.6813] |

> [!NOTE]
> **BEIR SciFact Baseline Verification**: Pre-trained `all-MiniLM-L6-v2` achieves **0.6451 nDCG@10** on the full 300 test queries, which directly reproduces the standard published BEIR benchmark result (~0.645–0.650 nDCG@10). An earlier scratch test showed ~0.70 because it evaluated only an initial 100-query subset; full evaluation on all 300 test queries with `title + text` resolves this discrepancy.

---

## 3. Paired Bootstrap Hypothesis Tests (Per-Query Differences)

We performed paired bootstrap hypothesis testing (10,000 iterations) directly on per-query differences (Delta_q = Metric_AFTER(q) - Metric_BEFORE(q)):

| Benchmark Dataset | Metric | Mean Difference (After - Before) | 95% Bootstrap CI of Difference | Bootstrap p-value | Significance Verdict |
|:---|:---|:---:|:---:|:---:|:---|
| **Validation Set** | Recall@10 | +3.01% | [1.89%, 4.23%] | 0.0000 | **Significant (p < 0.05)** |
| **Validation Set** | MRR@10 | +0.0764 | [0.0610, 0.0926] | 0.0000 | **Significant (p < 0.05)** |
| **Validation Set** | nDCG@10 | +0.0652 | [0.0523, 0.0789] | 0.0000 | **Significant (p < 0.05)** |
| **Test Holdout** | Recall@10 | +4.38% | [2.92%, 5.84%] | 0.0000 | **Significant (p < 0.05)** |
| **Test Holdout** | MRR@10 | +0.1039 | [0.0873, 0.1218] | 0.0000 | **Significant (p < 0.05)** |
| **Test Holdout** | nDCG@10 | +0.0896 | [0.0749, 0.1051] | 0.0000 | **Significant (p < 0.05)** |
| **SciFact (BEIR)** | Recall@10 | -0.50% | [-3.30%, 2.30%] | 0.7238 | **Not Significant (p >= 0.05)** |
| **SciFact (BEIR)** | MRR@10 | -0.0102 | [-0.0321, 0.0110] | 0.3590 | **Not Significant (p >= 0.05)** |
| **SciFact (BEIR)** | nDCG@10 | -0.0092 | [-0.0286, 0.0093] | 0.3290 | **Not Significant (p >= 0.05)** |

---

## 4. Deployment & Fallback Protocol
- **Model Directory**: [`data/models/embedder-ft`](file://data/models/embedder-ft)
- **Fallback Toggle**: In `config.yaml`, `active_model` specifies `"base"` or `"fine_tuned"`. If the fine-tuned checkpoint is missing, `app.py` falls back seamlessly to the pre-trained `all-MiniLM-L6-v2`.
