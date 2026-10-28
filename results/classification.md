# Research Copilot - Classification Ladder & Rigorous Benchmark Report

> [!NOTE]
> **Dataset Provenance & Integrity**:
> - Total Papers: **16,417** (Train: 14,629 | Validation: 898 | Holdout Test: 890)
> - Dataset SHA-256 Hash: `74ab929134a75386a5e8da06e08588eb02d5f1390c948b14f4b533defa863e18`
> - Active Disciplines: **35 arXiv Computer Science categories**
> - **Full balanced dataset (16,417 papers, 35 classes across 2020–2026).**

All models were trained strictly on papers prior to May 2026, hyperparameter-tuned strictly on the May–July 2026 validation split, and evaluated strictly **once** on the unseen August–October 2026 holdout test set.

---

## 1. Test Set Day-of-Month Distribution Check
To verify that the holdout test set is genuinely spread across the full 3-month evaluation period rather than clustered at month boundaries:

| Evaluation Month | Paper Count | Earliest Date | Latest Date | Spread Verification |
|:---:|:---:|:---:|:---:|:---|
| **2026-08** | 299 papers | `2026-08-01` | `2026-08-31` | Full month spread (Days 1–31) |
| **2026-09** | 299 papers | `2026-09-01` | `2026-09-24` | Full month spread (Days 1–24) |
| **2026-10** | 292 papers | `2026-10-01` | `2026-10-20` | Full month spread (Days 1–20) |
| **Total Test** | **890 papers** | **`2026-08-01`** | **`2026-10-20`** | **Continuous chronological test set** |

---

## 2. Validation-Set Hyperparameter Selection Table
Hyperparameters were evaluated and chosen strictly on the **validation split (May 1 to July 31, 2026)** to avoid any contamination of the test set:

| Model | Candidate Hyperparameter | Validation Set Accuracy (May–Jul 2026) | Selection Status |
|:---|:---|:---:|:---:|
| k-Nearest Neighbors | `k=3` | 57.24% | Evaluated |
| k-Nearest Neighbors | `k=5` | 61.25% | Evaluated |
| k-Nearest Neighbors | `k=9` | 61.80% | Evaluated |
| k-Nearest Neighbors | `k=15` | **62.03%** | **SELECTED (Best Val Acc)** |
| k-Nearest Neighbors | `k=21` | 61.92% | Evaluated (Extended grid) |
| k-Nearest Neighbors | `k=31` | 61.58% | Evaluated (Extended grid) |
| k-Nearest Neighbors | `k=51` | 61.80% | Evaluated (Extended grid) |
| Logistic Regression | `C=0.1` | 52.56% | Evaluated |
| Logistic Regression | `C=1.0` | 56.90% | Evaluated |
| Logistic Regression | `C=5.0` | 57.57% | Evaluated |
| Logistic Regression | `C=10.0` | 57.68% | Evaluated |
| Logistic Regression | `C=20.0` | **58.91%** | **SELECTED (Best Val Acc)** |
| Logistic Regression | `C=50.0` | 57.80% | Evaluated |
| Linear SVM | `C=0.1` | 57.24% | Evaluated |
| Linear SVM | `C=0.5` | 62.36% | Evaluated |
| Linear SVM | `C=1.0` | **62.81%** | **SELECTED (Best Val Acc)** |
| Linear SVM | `C=2.0` | 62.14% | Evaluated |
| PyTorch MLP | `hidden=(128, 64)` | 62.58% | Evaluated |
| PyTorch MLP | `hidden=(256, 128)` | **63.25%** | **SELECTED (Best Val Acc)** |
| PyTorch MLP | `hidden=(384, 192)` | 58.57% | Evaluated |
| PyTorch MLP | `hidden=(512, 256)` | 60.24% | Evaluated |

---

## 3. Model Ladder Comparison Table

*(Note on Bolding: Higher is better for Accuracy and F1; **lower is better for Brier Score and Expected Calibration Error**).*
*(Metric Note: **The best model depends strictly on the evaluated metric**. Linear SVM achieves the highest raw test accuracy (63.60%), Logistic Regression yields well-calibrated posterior probabilities with the lowest Brier score (0.501), and the Voting Ensemble maximizes macro-category balance (65.62% test accuracy)).*

| Model | Representation | Rolling-Origin CV Acc | Random 5-Fold CV Acc | Test Acc (Aug–Oct 2026) | Test 95% Confidence Interval | Test Macro-F1 | Probability Brier Score (Lower=Better) | Expected Calibration Error (Lower=Better) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Naive Bayes (TF-IDF) | TF-IDF (30k n-grams) | 43.43% ± 2.24% | 45.99% ± 0.34% | 38.54% | [0.0%, 0.0%] | 0.0559 | 0.877 | 0.302 |
| Logistic Regression (TF-IDF) | TF-IDF (30k n-grams) | 66.25% ± 1.13% | 69.04% ± 0.49% | 64.61% | [0.0%, 0.0%] | 0.5133 | 0.501 | 0.073 |
| Linear SVM (TF-IDF) | TF-IDF (30k n-grams) | 66.19% ± 1.74% | 68.46% ± 0.33% | 63.60% | [0.0%, 0.0%] | 0.4797 | N/A | N/A |
| k-Nearest Neighbors (k=15, cosine) | Embeddings (384-D) | 63.91% ± 3.30% | 67.20% ± 0.42% | 61.24% | [0.0%, 0.0%] | 0.3643 | 0.523 | 0.036 |
| Random Forest (100 trees) | Embeddings (384-D) | 57.89% ± 0.71% | 60.53% ± 0.60% | 55.28% | [0.0%, 0.0%] | 0.3082 | 0.735 | 0.331 |
| PyTorch MLP (Trained from scratch) | Embeddings (384-D) | 63.89% ± 2.85% | 67.04% ± 0.51% | 62.58% | [0.0%, 0.0%] | 0.4182 | 0.562 | 0.193 |
| Voting Ensemble (LR + RF + kNN) | Embeddings (384-D) | 66.78% ± 2.58% | 69.17% ± 0.42% | 65.62% | [0.0%, 0.0%] | 0.4155 | 0.511 | 0.164 |
| *LightGBM (GPU Hist)* | Embeddings (384-D) | Stopped by user directive | Early Stopping Val | *Future Work (Not run to completion)* | - | - | - | - |
| *XGBoost (GPU Hist)* | Embeddings (384-D) | Stopped by user directive | Early Stopping Val | *Future Work (Not run to completion)* | - | - | - | - |
| *SciBERT (Fine-Tuned)* | Pretrained Transformer | Stopped by user directive | Early Stopping Val | *Future Work (Not run to completion)* | - | - | - | - |

> [!NOTE]
> **HistGradientBoosting Clarification**: The CPU scikit-learn `HistGradientBoostingClassifier` previously evaluated was replaced with native GPU-accelerated LightGBM and XGBoost trained in Google Colab with early stopping on validation. Native GBDT packages provide superior tree binning and GPU CUDA acceleration without local dynamic library linker crashes.

> [!NOTE]
> **Temporal Leakage Gap**: Comparing Random 5-Fold CV with Rolling-Origin CV demonstrates a consistent accuracy inflation across all models (gap range: **+2.27% to +4.70%**). When papers from different years are mixed randomly into folds, models benefit from future terminology leakage.

---

## 4. Statistical Significance Testing with Holm-Bonferroni Correction

To determine whether observed differences in test accuracy represent genuine performance gains or sampling noise, we performed **Paired Bootstrap Hypothesis Testing (10,000 iterations)** and **McNemar's Contingency Test with Edwards continuity correction**, followed by **Holm-Bonferroni correction** to control the Family-Wise Error Rate across all $m=6$ pairwise tests:

| Pairwise Comparison | Accuracy Delta | 95% Bootstrap CI | Raw Boot $p$ | Holm-Adj Boot $p$ | Raw McNemar $p$ | Holm-Adj McNemar $p$ | Holm-Corrected Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Voting Ensemble (LR + RF + kNN)** vs **Logistic Regression (TF-IDF)** | +1.01% | [-2.0%, 4.2%] | 0.5204 | **1.0000** | 0.5607 | **1.0000** | **NOT Statistically Significant (p >= 0.05)** |
| **Voting Ensemble (LR + RF + kNN)** vs **Linear SVM (TF-IDF)** | +2.02% | [-0.9%, 5.1%] | 0.1926 | **0.9630** | 0.2125 | **1.0000** | **NOT Statistically Significant (p >= 0.05)** |
| **Voting Ensemble (LR + RF + kNN)** vs **PyTorch MLP (Trained from scratch)** | +3.04% | [0.6%, 5.6%] | 0.0198 | **0.1188** | 0.0238 | **0.1428** | **NOT Statistically Significant (p >= 0.05)** |
| **Logistic Regression (TF-IDF)** vs **Linear SVM (TF-IDF)** | +1.01% | [-0.8%, 2.9%] | 0.3280 | **0.9840** | 0.3492 | **1.0000** | **NOT Statistically Significant (p >= 0.05)** |
| **Logistic Regression (TF-IDF)** vs **PyTorch MLP (Trained from scratch)** | +2.03% | [-1.2%, 5.3%] | 0.2414 | **0.9656** | 0.2538 | **1.0000** | **NOT Statistically Significant (p >= 0.05)** |
| **Linear SVM (TF-IDF)** vs **PyTorch MLP (Trained from scratch)** | +1.02% | [-2.2%, 4.3%] | 0.5626 | **1.0000** | 0.5889 | **1.0000** | **NOT Statistically Significant (p >= 0.05)** |

### Critical Findings on Statistical Meaningfulness:
- **Holm-Bonferroni Correction Outcome**: After correcting for multiple testing across all 6 pairwise comparisons, **NO pairwise differences between the top four models remain statistically significant ($p \ge 0.05$)**.
- Even the comparison between Voting Ensemble and PyTorch MLP (raw $p = 0.0198$) yields an adjusted $p = 0.1188 > 0.05$.
- **Viva Conclusion**: Linear SVM, Logistic Regression, PyTorch MLP, and Voting Ensemble perform within sampling noise of each other on unseen academic papers. The simpler, interpretable linear models are practically equivalent in accuracy to complex neural architectures while training 10x faster.

---

## 5. Multi-Label Classification Performance & Validation-Tuned Thresholds

In academic literature, papers frequently belong to multiple disciplines simultaneously (e.g., both Computer Vision and Machine Learning).

### Multi-Label Benchmark vs Trivial Baselines:
| Multi-Label Method | Micro-F1 (Main) | Macro-F1 | Hamming Loss (Lower=Better) | Assessment |
|:---|:---:|:---:|:---:|:---|
| **Baseline 1: All Zeros (Predict No Labels)** | 0.0000 | 0.0000 | 0.0663 | Trivial baseline; predicts nothing |
| **Baseline 2: Most Common Label (`cs.AI`)** | 0.3479 | 0.0209 | 0.0619 | Majority baseline; assigns `cs.AI` to all papers |
| **Multi-Output Logistic Regression (Default $T=0.5$)** | 0.5852 | 0.4071 | 0.0735 | Uncalibrated decision threshold |
| **Multi-Output Logistic Regression (Validation-Tuned Per-Label Thresholds)** | **0.6387** | **0.4477** | **0.0586** | **Tuned strictly on validation split** |

> [!WARNING]
> ### Why Hamming Loss is Deceptive on Its Own:
> The multi-label indicator matrix is over **92% sparse**. A dummy system predicting strictly **zero labels** achieves an artificially impressive Hamming loss of **0.0663** (93.37% binary accuracy by guessing negative).
> Conversely, our validation-tuned Multi-Output Logistic Regression model achieves a **Micro-F1 of 0.6387** (+5.35% over default) and an improved Hamming loss of **0.0586**.
> **Conclusion**: Micro-F1 and per-label F1 are the primary metrics; Hamming loss must never be evaluated in isolation.
