# Research Copilot: 40 Essential Viva Questions & Concise Answers

A comprehensive viva examination guide covering core concepts, mathematical principles, empirical findings, and architectural decisions.

---

### Part 1: Information Retrieval & System Architecture

**Q1: What is the core problem Research Copilot solves?**
**A1:** It addresses the "semantic blindness" of keyword search (which fails when queries and papers use different vocabulary) and the "hallucination problem" of generative AI by using hybrid retrieval with verified citations.

**Q2: What is the difference between sparse and dense retrieval?**
**A2:** Sparse retrieval (BM25) matches exact keywords and lexical tokens based on term frequencies; dense retrieval (FAISS) maps texts to continuous vector embeddings to capture conceptual meaning regardless of exact wording.

**Q3: Why did you combine BM25 and dense embeddings into a hybrid search?**
**A3:** Dense embeddings excel at conceptual paraphrases but struggle with rare acronyms, author names, or mathematical notation; BM25 catches exact keyword matches. Combining them ensures neither semantic nuance nor technical precision is lost.

**Q4: How does Reciprocal Rank Fusion (RRF) work, and why not just add scores?**
**A4:** Raw BM25 scores are unbounded $[0, \infty)$ while cosine similarities are $[-1, 1]$. Adding them requires fragile heuristic weight tuning. RRF fuses them purely by their ordinal ranks: $RRF(d) = \sum \frac{1}{60 + \text{rank}(d)}$, creating a robust, scale-invariant ranking.

**Q5: What is the role of the Cross-Encoder re-ranker?**
**A5:** Dual-encoders embed queries and documents independently to allow fast indexing. A cross-encoder passes the query and candidate text together into full self-attention layers, computing fine-grained token-level cross-attention to produce a highly accurate final score.

**Q6: What is FAISS HNSW and why is it preferred over flat indexing?**
**A6:** Flat indexing performs an exhaustive $O(N)$ comparison across all vectors. HNSW (Hierarchical Navigable Small World) builds a multi-layer proximity graph that achieves $O(\log N)$ sub-millisecond approximate nearest neighbor retrieval with $>95\%$ recall.

**Q7: What is the cosine similarity metric, and why is inner product identical here?**
**A7:** Cosine similarity measures the angle between two vectors: $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$. Because all our embeddings are $L_2$-normalized to unit length ($\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1$), the denominator is 1, making cosine similarity mathematically identical to the inner product.

**Q8: What is extractive vs generative question answering in your app?**
**A8:** Extractive QA directly selects real, verbatim sentences from the retrieved paper abstracts using Maximal Marginal Relevance (MMR). Generative QA synthesizes an answer using an LLM while enforcing strict inline citations $[1], [2]$.

**Q9: How does the citation grounding verification work?**
**A9:** Each sentence in the generated answer containing a citation $[n]$ is embedded and compared against all sentences in abstract $n$. If semantic similarity is $\ge 0.45$, it receives a verified check $(\checkmark)$; otherwise, it is flagged with a warning $(\triangle)$.

**Q10: What is Maximal Marginal Relevance (MMR)?**
**A10:** MMR balances relevance to the query with diversity among chosen sentences to prevent repetitive answers: $MMR = \arg\max [\lambda \cdot \text{Sim}(s, Q) - (1 - \lambda) \cdot \max \text{Sim}(s, s_j)]$.

---

### Part 2: Machine Learning, Data Splits & Calibration

**Q11: Why did you use a date-boundary split instead of random k-fold cross-validation?**
**A11:** Random splits cause temporal data leakage: models trained on future papers memorize emerging vocabulary, artificially inflating accuracy by $2.3\text{–}4.7\%$. Date boundaries simulate true production deployment on unseen future papers.

**Q12: What are your dataset splits?**
**A12:** 16,417 total arXiv papers across 35 CS categories: Train ($N=14,629$, $\le$ April 2026), Validation ($N=898$, May–July 2026), and Holdout Test ($N=890$, August–October 2026).

**Q13: How did you verify that the test set is continuous and not clustered?**
**A13:** We ran a day-of-month distribution audit confirming papers are evenly distributed across all days of August (Days 1–31), September (Days 1–24), and October 2026 (Days 1–20).

**Q14: What is the Brier score and what does it measure?**
**A14:** The Brier score is the mean squared error between predicted class probability distributions and one-hot true labels: $\frac{1}{N} \sum \|\mathbf{p}_i - \mathbf{y}_i\|_2^2$. Lower is better; it measures probability calibration.

**Q15: What is Expected Calibration Error (ECE)?**
**A15:** ECE partitions predicted probabilities into bins, calculates the difference between average confidence and true accuracy in each bin, and computes the weighted average. Lower ECE means predicted probabilities reflect real-world confidence.

**Q16: Which model had the best calibration?**
**A16:** Logistic Regression achieved the lowest multi-class Brier score (0.501), and k-NN achieved the lowest Expected Calibration Error (0.036).

**Q17: Why does Linear SVM lack a standard Brier score?**
**A17:** SVM is a maximum-margin classifier that outputs uncalibrated geometric distances to the separating hyperplane rather than true posterior probabilities.

**Q18: What is the difference between Micro-F1 and Macro-F1?**
**A18:** Macro-F1 computes the metric independently for each category and averages them, giving equal weight to minority and majority classes. Micro-F1 aggregates all true positives, false positives, and false negatives globally, reflecting overall sample accuracy.

**Q19: Why was hyperparameter tuning performed strictly on the validation set?**
**A19:** Tuning on the test set contaminates the holdout with human and model selection bias. The test set was touched exactly once after all parameters were locked.

**Q20: What is Rolling-Origin Cross-Validation?**
**A20:** An expanding-window time-series CV that trains on historical blocks (e.g., 2020–2022) and tests on the immediate next block (2023), repeating across 4 temporal splits without ever training on future data.

---

### Part 3: Statistical Significance & Multi-Label Classification

**Q21: Why do we need statistical significance tests when comparing model accuracies?**
**A21:** A higher accuracy on an 890-paper test set could easily occur due to random sampling noise. Significance tests quantify whether the difference is statistically distinguishable from zero.

**Q22: What is McNemar's Test?**
**A22:** A paired non-parametric test on a $2 \times 2$ contingency table evaluating discordance: papers where Model A was right and Model B was wrong vs papers where Model B was right and Model A was wrong ($\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$).

**Q23: What is Paired Bootstrap Hypothesis Testing?**
**A23:** We resample the test set with replacement 10,000 times, calculate the accuracy difference $\Delta = \text{Acc}_A - \text{Acc}_B$ on each replicate, construct empirical 95% confidence intervals, and compute a two-sided $p$-value.

**Q24: What is Holm-Bonferroni correction, and why did you use it?**
**A24:** When conducting multiple pairwise comparisons ($m=6$), the chance of a false positive increases (family-wise error). Holm's method sorts $p$-values and tests them sequentially with adjusted thresholds ($(m - k + 1) \cdot p_{(k)}$) to prevent false discoveries.

**Q25: What was the outcome of Holm-Bonferroni correction on your models?**
**A25:** After Holm-Bonferroni correction, **no pairwise differences between the top four models remain statistically significant ($p \ge 0.05$)**. Voting Ensemble, Linear SVM, Logistic Regression, and PyTorch MLP perform equivalently within margin of error.

**Q26: What is the practical engineering implication of this lack of statistical significance?**
**A26:** Simpler, linear TF-IDF models and fast linear classifiers match complex neural models in production accuracy while training 10x faster and using far fewer compute resources.

**Q27: Why is Hamming loss misleading when evaluating multi-label text classification?**
**A27:** Academic category matrices are over 92% sparse. A dummy baseline that predicts zero labels for every paper achieves a deceptive 0.0663 Hamming loss despite having 0.0 Micro-F1.

**Q28: What is your multi-label threshold tuning strategy?**
**A28:** Instead of using an arbitrary default cutoff of 0.5 for all classes, we optimize individual category thresholds $T_c \in [0.2, 0.8]$ on the validation set to maximize each class's F1-score.

**Q29: How much did validation-tuned thresholds improve multi-label performance?**
**A29:** Micro-F1 improved from 0.5852 to **0.6387 (+5.35%)**, Macro-F1 improved from 0.4071 to **0.4477 (+4.06%)**, and Hamming loss improved to **0.0586**.

**Q30: Why does the Voting Ensemble outperform individual models on test accuracy?**
**A30:** It averages soft posterior probabilities across three fundamentally different inductive biases: linear hyperplanes (Logistic Regression), recursive partitioning (Random Forest), and local manifolds (k-NN).

---

### Part 4: Deep Learning, Transformers & Contrastive Embeddings

**Q31: What loss function is used for contrastive embedder fine-tuning?**
**A31:** Multiple Negatives Ranking Loss (MNRL), which uses cross-entropy over a softmax of cosine similarities between positive pairs and in-batch hard negative pairs.

**Q32: What is a "hard negative" in contrastive learning?**
**A32:** An abstract from the same high-level scientific domain (e.g., another machine learning paper) that is conceptually close but does not describe the specific paper's title.

**Q33: How did you prevent false negatives during hard negative mining?**
**A33:** When mining candidates from the same topic cluster, we explicitly excluded the **top 5 nearest neighbors** of the positive document to prevent treating a closely related paper as a negative.

**Q34: What is SciFact, and why did you evaluate on it?**
**A34:** SciFact is an official BEIR benchmark of 300 expert-verified scientific claim-verification queries. Evaluating on SciFact tests whether contrastive fine-tuning on arXiv caused catastrophic forgetting on general scientific retrieval.

**Q35: What baseline score does all-MiniLM-L6-v2 achieve on BEIR SciFact?**
**A35:** It achieves **0.6451 nDCG@10**, precisely reproducing published BEIR benchmark literature.

**Q36: What is Catastrophic Forgetting?**
**A36:** When a pretrained model fine-tuned on a narrow domain loses its generalized representations and degrades on external benchmarks.

**Q37: What is SciBERT and how does it differ from standard BERT?**
**A37:** SciBERT (`allenai/scibert_scivocab_uncased`) was pretrained from scratch on 1.14 million scientific papers with a custom scientific vocabulary (`scivocab`), enabling it to tokenize medical, CS, and mathematical terms without aggressive subword splitting.

**Q38: Why do GBDTs (LightGBM/XGBoost) require histogram binning?**
**A38:** Histogram-based gradient boosting sorts continuous feature values into discrete integer bins (typically 256), reducing split-finding complexity from $O(\text{data} \times \text{features})$ to $O(\text{bins} \times \text{features})$ and enabling GPU execution.

**Q39: If deep transformers have more parameters, why did Linear SVM achieve 63.60% accuracy?**
**A39:** Academic paper categorizations are strongly correlated with domain-specific terminology (e.g., *"convolutional"*, *"cryptographic"*). High-dimensional TF-IDF feature spaces (30,000 n-grams) are often linearly separable, allowing maximum-margin hyperplanes to perform exceptionally well.

**Q40: What are three concrete areas of future work?**
**A40:** 
1. Natural Language Inference (NLI) for automatic citation premise-hypothesis entailment checking.
2. Comparative study between neural topic modeling (BERTopic) and classical LDA.
3. Autoregressive temporal modeling for multi-year scientific topic trend forecasting.
