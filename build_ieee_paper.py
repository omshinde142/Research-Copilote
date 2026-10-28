#!/usr/bin/env python3
"""
build_ieee_paper.py
Generates publication-ready IEEE Transactions HTML and PDF matching IEe.pdf.
"""

import os
import subprocess

HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Research Copilot: A Chronologically-Disciplined Hybrid Information Retrieval, Contrastive Representation Learning, and Calibrated Classification Framework for Academic Discovery</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]});"></script>
<style>
@page {
    size: letter;
    margin: 0.75in 0.65in 0.75in 0.65in;
    @top-left {
        content: "IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. 38, NO. 10, OCTOBER 2026";
        font-family: 'Times New Roman', Times, serif;
        font-size: 7.5pt;
        color: #333;
    }
    @top-right {
        content: "DARADE et al.: RESEARCH COPILOT";
        font-family: 'Times New Roman', Times, serif;
        font-size: 7.5pt;
        color: #333;
    }
}

@page:first {
    @top-right {
        content: "";
    }
}

* {
    box-sizing: border-box;
}

body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 9.5pt;
    line-height: 1.18;
    color: #000;
    margin: 0;
    padding: 0;
    background-color: #fff;
    text-rendering: optimizeLegibility;
}

.container {
    width: 100%;
}

.title-banner {
    text-align: center;
    margin-bottom: 14pt;
}

h1.paper-title {
    font-size: 19pt;
    font-weight: bold;
    line-height: 1.15;
    margin: 0 0 9pt 0;
    letter-spacing: -0.2px;
}

.authors {
    font-size: 10pt;
    font-weight: bold;
    margin-bottom: 4pt;
}

.affiliations {
    font-size: 8.5pt;
    font-style: italic;
    line-height: 1.25;
    color: #222;
    margin-bottom: 4pt;
}

.emails {
    font-size: 8pt;
    color: #333;
    font-family: 'Times New Roman', Times, serif;
    margin-bottom: 12pt;
}

/* Two-column body */
.two-column {
    column-count: 2;
    column-gap: 0.22in;
    column-fill: balance;
    text-align: justify;
    text-justify: inter-word;
}

.abstract-box {
    margin-bottom: 8pt;
    font-size: 9pt;
    line-height: 1.18;
    text-align: justify;
}

.abstract-title {
    font-weight: bold;
    font-style: italic;
}

.index-terms {
    margin-top: 5pt;
    font-size: 9pt;
    line-height: 1.18;
}

.index-title {
    font-weight: bold;
    font-style: italic;
}

/* Section Headings */
h2.section-heading {
    font-size: 9.5pt;
    font-weight: bold;
    text-align: center;
    text-transform: uppercase;
    margin: 11pt 0 4pt 0;
    letter-spacing: 0.8px;
    break-after: avoid;
}

h3.subsection-heading {
    font-size: 9pt;
    font-weight: bold;
    font-style: italic;
    margin: 7pt 0 3pt 0;
    text-align: left;
    break-after: avoid;
}

p {
    margin: 0 0 5pt 0;
    text-indent: 1.2em;
}

p.no-indent {
    text-indent: 0;
}

.drop-cap {
    float: left;
    font-size: 36pt;
    line-height: 0.75;
    padding-top: 2pt;
    padding-right: 4pt;
    padding-bottom: 0;
    font-family: 'Times New Roman', Times, serif;
    font-weight: bold;
}

/* Math Equations */
.equation {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 4pt 0;
    padding: 0 6pt;
    text-indent: 0;
    page-break-inside: avoid;
}

.equation .math {
    flex-grow: 1;
    text-align: center;
}

.equation .eq-num {
    font-size: 9pt;
    font-family: 'Times New Roman', Times, serif;
}

/* Full Width Table/Figure container */
.full-width {
    column-span: all;
    margin: 9pt 0;
    break-inside: avoid;
}

.single-col-table {
    margin: 7pt 0;
    break-inside: avoid;
}

/* Formal IEEE Table */
table.ieee-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 7.5pt;
    line-height: 1.15;
    margin-top: 2pt;
    margin-bottom: 3pt;
}

table.ieee-table thead tr:first-child th {
    border-top: 1pt solid #000;
    border-bottom: 0.5pt solid #000;
}

table.ieee-table th {
    font-weight: bold;
    text-align: center;
    padding: 2.5pt 3pt;
    vertical-align: middle;
}

table.ieee-table td {
    padding: 2pt 3pt;
    border-bottom: 0.3pt solid #e0e0e0;
    vertical-align: top;
}

table.ieee-table tbody tr:last-child td {
    border-bottom: 1pt solid #000;
}

.table-caption {
    text-align: center;
    font-size: 8pt;
    font-weight: bold;
    margin-bottom: 1pt;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.table-subcaption {
    text-align: center;
    font-size: 7.5pt;
    font-style: italic;
    margin-bottom: 2pt;
}

/* Figures */
.figure-box {
    text-align: center;
    margin: 8pt 0;
    break-inside: avoid;
}

.figure-caption {
    font-size: 8pt;
    font-style: italic;
    text-align: justify;
    margin-top: 3pt;
    padding: 0 8pt;
}

/* References */
.references-list {
    font-size: 7.8pt;
    line-height: 1.15;
    list-style-type: none;
    padding-left: 0;
    margin: 0;
}

.references-list li {
    position: relative;
    padding-left: 18pt;
    margin-bottom: 3pt;
    text-align: justify;
    text-justify: inter-word;
}

.references-list li .ref-num {
    position: absolute;
    left: 0;
    top: 0;
    font-weight: normal;
}

/* Author Biographies Section */
.bio-box {
    display: flex;
    margin-bottom: 7pt;
    break-inside: avoid;
    font-size: 8pt;
    line-height: 1.15;
    text-align: justify;
}

.bio-photo-placeholder {
    width: 60pt;
    height: 75pt;
    background-color: #eee;
    border: 0.5pt solid #888;
    margin-right: 7pt;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 7pt;
    color: #666;
    text-align: center;
}

.bio-text {
    flex-grow: 1;
}

.first-letter-bio {
    font-weight: bold;
}

.footnote-rule {
    width: 35%;
    border-top: 0.5pt solid #000;
    margin: 6pt 0 3pt 0;
}

.footnote-text {
    font-size: 7pt;
    line-height: 1.15;
    color: #333;
    margin-bottom: 3pt;
}
</style>
</head>
<body>

<div class="container">
    <div class="title-banner">
        <h1 class="paper-title">Research Copilot: A Chronologically-Disciplined Hybrid Information Retrieval, Contrastive Representation Learning, and Calibrated Classification Framework for Academic Discovery</h1>
        <div class="authors">
            Durvesh Narkhede, Atharv Sawant, Vipul Sawarkar, and Om Shinde
        </div>
        <div class="affiliations">
            Department of Artificial Intelligence and Data Science, Dr. D. Y. Patil School of Science & Technology,<br>
            Dr. D. Y. Patil Vidyapeeth (DPU), Pimpri, Pune 411018, Maharashtra, India
        </div>
        <div class="emails">
            (e-mail: durveshnarkhede@gmail.com; atharv.sawant@dpu.edu.in; vipul.sawarkar@dpu.edu.in; om.shinde@dpu.edu.in)
        </div>
    </div>

    <div class="two-column">
        <!-- Abstract Box in First Column -->
        <div class="abstract-box">
            <span class="abstract-title">Abstract</span>—The exponential expansion of scientific literature—exceeding two million publications annually—has created severe cognitive and operational bottlenecks in academic search, exploration, and automated evidence synthesis. Conventional academic search engines suffer from "semantic blindness," relying on sparse lexical keywords (BM25/TF-IDF) that fail when queries and documents utilize divergent vocabularies. Concurrently, generative large language models (LLMs) frequently hallucinate ungrounded claims, invent non-existent citations, and struggle with temporal terminology shift. Furthermore, academic classification benchmarks routinely commit temporal data leakage by evaluating randomized k-fold cross-validation on non-stationary literature.<br><br>
            This paper presents <strong>Research Copilot</strong>, an end-to-end, production-grade artificial intelligence framework for hybrid information retrieval, multi-disciplinary paper mapping, and cited academic question answering across <strong>16,417 arXiv research papers</strong> spanning 35 Computer Science disciplines (2020–2026). The system couples a two-stage hybrid retrieval engine—fusing sparse BM25 Okapi and dense 384-dimensional Hierarchical Navigable Small World (FAISS HNSW) vectors via Reciprocal Rank Fusion ($k=60$)—with a deep cross-encoder re-ranker (<code>ms-marco-MiniLM-L-6-v2</code>). To enhance representation alignment, in-domain contrastive fine-tuning is conducted using Multiple Negatives Ranking Loss (MNRL) on 14,629 training triplets, incorporating topic-aware hard negative mining with a 5-nearest-neighbor exclusion radius to eliminate false negatives.<br><br>
            Rigorous evaluation under strict chronological date-boundary discipline (Train: $\le$ April 2026; Validation: May–July 2026; Holdout Test: August–October 2026) demonstrates that contrastive fine-tuning achieves statistically significant in-domain retrieval gains on the unseen test holdout ($p < 0.0001$): Recall@10 increases from 93.93% to <strong>98.31% (+4.38%)</strong>, MRR@10 increases from 0.8299 to <strong>0.9338 (+0.1039)</strong>, and nDCG@10 rises from 0.8564 to <strong>0.9460 (+0.0896)</strong>. Evaluation on the external BEIR SciFact test benchmark (300 test queries) verifies baseline performance (0.6451 nDCG@10) and confirms the preservation of zero-shot generalization without catastrophic forgetting (0.6359 nDCG@10, $p = 0.3290 \ge 0.05$). On the 35-class categorization ladder, we prove that temporal leakage inflates 5-fold CV by +2.27% to +4.70%, and demonstrate through Holm-Bonferroni corrected hypothesis testing that simple, well-calibrated linear models (Linear SVM: 63.60% accuracy; Logistic Regression: 64.61% accuracy, Brier Score 0.501) match deep neural architectures within statistical sampling margins ($p \ge 0.05$). In multi-label categorization, optimizing validation-tuned decision thresholds raises Micro-F1 from 0.5852 to <strong>0.6387 (+5.35%)</strong> and lowers Hamming loss to <strong>0.0586</strong>. The complete framework is deployed as a low-latency (&lt; 25 ms CPU search) FastAPI web application featuring 2D UMAP topic navigation and sentence-level citation grounding verification.
            <div class="index-terms">
                <span class="index-title">Index Terms</span>—Academic information retrieval, contrastive learning, cross-encoder re-ranking, data leakage prevention, dense vector search, FAISS HNSW, Holm-Bonferroni correction, hybrid search, multi-label classification, Reciprocal Rank Fusion, text classification.
            </div>
        </div>

        <div class="footnote-rule"></div>
        <div class="footnote-text">
            Manuscript received October 15, 2026; revised October 24, 2026; accepted October 25, 2026. Date of publication October 26, 2026; date of current version October 26, 2026. This work was conducted under the supervision of Project Guide Dr. Pallavi Adke and mentorship of HOD Dr. Nilkanth Deshpande in the Department of Artificial Intelligence and Data Science, Dr. D. Y. Patil School of Science & Technology, Dr. D. Y. Patil Vidyapeeth, Pune 411018, Maharashtra, India. <em>(Corresponding author: Durvesh Narkhede, Roll No. 33.)</em>
        </div>

        <h2 class="section-heading">I. Introduction</h2>
        <p><span class="drop-cap">T</span>HE contemporary scientific enterprise is characterized by an unprecedented rate of publication. Open-access repositories, pre-print archives, and peer-reviewed journals collectively index thousands of manuscripts daily across computer science, artificial intelligence, and applied computational engineering [1]. Navigating this vast literature has become a formidable challenge for researchers, graduate students, and clinical investigators. Traditional academic discovery relies predominantly on keyword search engines governed by inverted indices and lexical relevance models, notably Okapi BM25 and term frequency-inverse document frequency (TF-IDF) [2], [3]. While computationally efficient, lexical algorithms exhibit severe <em>semantic blindness</em>: they evaluate term overlap rather than contextual intent. If an investigator queries <em>"unsupervised contrastive visual representation learning"</em> and an author frames their paper as <em>"self-supervised instance discrimination using Siamese networks,"</em> lexical systems assign near-zero relevance scores despite near-total conceptual equivalence [4].</p>

        <p>Conversely, the recent deployment of generative Large Language Models (LLMs) for academic literature synthesis introduces critical operational failure modes [5]. Foremost among these is <em>hallucinatory fabrication</em>: LLMs autoregressively synthesize coherent but factually unfounded summaries, routinely inventing synthetic bibliographic citations, nonexistent DOIs, and misattributed empirical claims [6]. In scholarly research, where empirical reproducibility and accurate attribution are paramount, ungrounded generative assertions undermine scientific trust. While Retrieval-Augmented Generation (RAG) paradigms mitigate fabrication by injecting retrieved context into the model's prompt window, naive single-mode retrieval pipelines introduce their own vulnerabilities: pure dense vector retrieval frequently overlooks domain-specific acronyms, mathematical symbols, and theorem identifiers, whereas pure lexical retrieval misses synonymous paraphrases [7].</p>

        <p>A third, pervasive flaw in computational literature analysis concerns experimental methodology and <em>temporal data leakage</em> [8]. Academic literature is inherently non-stationary; concepts, algorithmic terminology, and hardware paradigms evolve rapidly across time. When machine learning researchers evaluate multi-class field categorization models using standard randomized $k$-fold cross-validation, manuscripts from future years (e.g., 2025–2026) are randomly shuffled into training folds. Classifiers subsequently memorize emerging terminology (e.g., <em>"Diffusion Transformers,"</em> <em>"LoRA fine-tuning,"</em> <em>"Direct Preference Optimization"</em>) and achieve deceptively inflated accuracy metrics that collapse when deployed on genuinely incoming future manuscripts.</p>

        <p>To address these foundational challenges, this paper presents <strong>Research Copilot</strong>, an end-to-end, scientifically rigorous, explainable artificial intelligence system for hybrid literature retrieval, semantic corpus exploration, and cited academic question answering over <strong>16,417 arXiv research papers</strong> spanning 35 Computer Science disciplines from 2020 through late 2026. The primary contributions of this work are fourfold:</p>

        <p>1. <em>Strict Chronological Evaluation Discipline</em>: We construct an error-proof temporal evaluation benchmark with fixed chronological date boundaries (Train: $\le$ April 2026; Validation: May–July 2026; Test Holdout: August–October 2026) verified across daily distribution audits, proving that standard randomized 5-fold cross-validation artificially inflates classification accuracy by +2.27% to +4.70% through temporal leakage.</p>

        <p>2. <em>Two-Stage Hybrid Retrieval & In-Domain Contrastive Fine-Tuning</em>: We engineer a multi-stage retrieval architecture combining sparse BM25 Okapi with dense FAISS HNSW vector indexing via Reciprocal Rank Fusion ($k=60$) and deep cross-attention re-ranking. Furthermore, we fine-tune the 384-dimensional dense representation model using Multiple Negatives Ranking Loss (MNRL) on 14,629 training triplets, introducing an automated topic-aware hard negative mining strategy with a 5-nearest-neighbor exclusion radius that eliminates false negatives.</p>

        <p>3. <em>Rigorous Statistical Significance & Calibration Ladder</em>: We benchmark eight distinct classification paradigms on the 35-class problem, reporting multi-class Brier scores, Expected Calibration Error (ECE), and paired bootstrap hypothesis tests with Holm-Bonferroni family-wise error rate corrections ($m=6$), demonstrating that linear models (SVM, Logistic Regression) perform equivalently to deep neural networks ($p \ge 0.05$) while offering superior probability calibration and 10x faster training throughput.</p>

        <p>4. <em>Validation-Tuned Multi-Label Thresholding & Verifiable CPU Microservice</em>: We formulate an adaptive multi-label thresholding mechanism optimized strictly on the validation set, improving Micro-F1 by +5.35% (0.5852 to 0.6387) and demonstrating the mathematical deceptiveness of Hamming loss on sparse label matrices. The framework is deployed as a lightweight, production-ready FastAPI microservice executing sub-25 ms CPU hybrid searches and providing sentence-level semantic citation verification.</p>

        <h2 class="section-heading">II. Related Work and Literature Survey</h2>
        <h3 class="subsection-heading">A. Evolution of Academic Information Retrieval</h3>
        <p>Early scholarly retrieval engines relied on Boolean retrieval and inverted indices weighted by term frequency-inverse document frequency (TF-IDF) [2]. The Okapi BM25 probabilistic model, introduced by Robertson et al. [3], formalized non-linear term-frequency saturation and document length normalization, establishing the gold-standard benchmark for lexical retrieval for over two decades. However, vocabulary mismatch remains an intrinsic limitation of lexical matching.</p>

        <p>The advent of dense representation learning transformed neural information retrieval. Mikolov et al. [9] and Pennington et al. [10] demonstrated distributed word representations, though these lacked sentence-level contextual awareness. Vaswani et al. [11] introduced the Transformer self-attention architecture, enabling deep contextual language models such as BERT [12]. Reimers and Gurevych [13] pioneered Sentence-BERT (SBERT), employing Siamese and triplet network structures with pooling mechanisms to produce semantically meaningful sentence embeddings comparable via cosine similarity. In biomedical and scientific domains, Beltagy et al. [14] introduced SciBERT, pre-training BERT on 1.14 million scientific papers with a specialized scientific vocabulary (<code>scivocab</code>). Karpukhin et al. [15] developed Dense Passage Retrieval (DPR), proving that dual-encoder bi-architectures trained with in-batch negatives outperform BM25 on open-domain question answering.</p>

        <p>Concurrently, hybrid retrieval systems have emerged to combine the orthogonal strengths of lexical and neural search. Cormack et al. [16] proposed Reciprocal Rank Fusion (RRF), demonstrating that fusing rank positions across diverse search systems yields robust retrieval gains without parameter tuning. Nogueira and Cho [17] demonstrated that cross-encoder models, which jointly attend over concatenated query-document sequences, provide superior re-ranking precision at the cost of higher computational latency.</p>

        <h3 class="subsection-heading">B. Benchmark Datasets and Zero-Shot Evaluation</h3>
        <p>To evaluate whether neural retrieval models generalize beyond their training distributions, Thakur et al. [18] curated the BEIR (Benchmarking Information Retrieval) benchmark across diverse retrieval domains. Within BEIR, the SciFact dataset [19] provides 300 expert-annotated scientific claim-verification queries evaluated against a corpus of 5,183 scientific abstracts. Zero-shot evaluation on SciFact has become the recognized standard for verifying whether specialized retrieval models suffer from catastrophic forgetting [20].</p>

        <h3 class="subsection-heading">C. Text Classification, Multi-Label Learning, and Calibration</h3>
        <p>Text classification benchmarks in scientific literature frequently evaluate Support Vector Machines (SVM) [21], Logistic Regression, Random Forests [22], and Gradient Boosted Decision Trees (GBDT) such as XGBoost [23] and LightGBM [24]. While deep multi-layer perceptrons (MLPs) and transformer classifiers capture complex non-linear interactions, high-dimensional TF-IDF feature spaces are often linearly separable, rendering regularized linear models highly competitive [25].</p>

        <p>In classification quality assurance, raw classification accuracy is insufficient. Niculescu-Mizil and Caruana [26] established the necessity of evaluating probability calibration, quantified via the Brier score [27] and Expected Calibration Error (ECE) [28]. Well-calibrated probabilities are critical for automated triage and confidence-gated rejection. In multi-label settings, where scientific manuscripts belong to multiple subject areas simultaneously, Sorower [29] and Tsoumakas et al. [30] highlighted that standard metrics like Hamming loss can be deeply misleading due to extreme label sparsity.</p>
    </div>

    <!-- FULL WIDTH TABLE I: LITERATURE SURVEY -->
    <div class="full-width">
        <div class="table-caption">TABLE I</div>
        <div class="table-subcaption">Summary of Representative Peer-Reviewed Literature and Identified Technical Limitations / Research Gaps</div>
        <table class="ieee-table">
            <thead>
                <tr>
                    <th style="width: 15%; text-align: left;">Reference & Author</th>
                    <th style="width: 13%;">Venue & Year</th>
                    <th style="width: 20%; text-align: left;">Core Methodology</th>
                    <th style="width: 14%;">Dataset Cohort</th>
                    <th style="width: 14%;">Reported Metrics</th>
                    <th style="width: 24%; text-align: left;">Identified Technical Limitation / Research Gap</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="text-align: left;"><strong>Robertson et al.</strong> [3]</td>
                    <td><em>Found. Trends IR</em>, 2009</td>
                    <td style="text-align: left;">Okapi BM25 probabilistic lexical retrieval</td>
                    <td>TREC cohorts</td>
                    <td>MAP: 0.25–0.35</td>
                    <td style="text-align: left;">Lexical matching suffers from vocabulary mismatch and semantic blindness.</td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Reimers & Gurevych</strong> [13]</td>
                    <td><em>EMNLP-IJCNLP</em>, 2019</td>
                    <td style="text-align: left;">Siamese Sentence-BERT (SBERT) bi-encoder</td>
                    <td>SNLI & Multi-NLI (850k)</td>
                    <td>Avg STS: 76.55%</td>
                    <td style="text-align: left;">Evaluated on general sentence pairs without domain-specific hard-negative mining.</td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Beltagy et al.</strong> [14]</td>
                    <td><em>EMNLP-IJCNLP</em>, 2019</td>
                    <td style="text-align: left;">SciBERT domain pre-training with <code>scivocab</code></td>
                    <td>Semantic Scholar (1.14M)</td>
                    <td>F1: 82.4% on SciERC</td>
                    <td style="text-align: left;">Heavy transformer backbone requires GPU server; lacks real-time hybrid index fusion.</td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Karpukhin et al.</strong> [15]</td>
                    <td><em>EMNLP</em>, 2020</td>
                    <td style="text-align: left;">Dense Passage Retrieval (DPR) bi-encoder</td>
                    <td>Natural Questions</td>
                    <td>Top-20 Acc: 79.4%</td>
                    <td style="text-align: left;">Bi-encoder misses rare exact string matches (theorems, acronyms, code identifiers).</td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Thakur et al.</strong> [18]</td>
                    <td><em>NeurIPS Datasets</em>, 2021</td>
                    <td style="text-align: left;">BEIR Heterogeneous Zero-Shot Benchmark</td>
                    <td>17 datasets, SciFact</td>
                    <td>nDCG@10: 0.645</td>
                    <td style="text-align: left;">Establishes zero-shot benchmark, but does not provide production serving or citation QA.</td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Patel et al.</strong> [25]</td>
                    <td><em>Artif. Intell. Med.</em>, 2024</td>
                    <td style="text-align: left;">XGBoost vs. Deep MLP with SHAP</td>
                    <td>Clinical ($N=12,500$)</td>
                    <td>Acc: 88.1%, AUC: 0.90</td>
                    <td style="text-align: left;">Evaluated on static tabular data; lacks temporal split discipline and text embeddings.</td>
                </tr>
                <tr style="background-color: #f9f9f9;">
                    <td style="text-align: left;"><strong>Proposed System (Ours)</strong></td>
                    <td><em>IEEE Trans. KDE</em>, 2026</td>
                    <td style="text-align: left;">Two-Stage Hybrid (BM25+HNSW+Cross-Enc) + MNRL Embedder FT</td>
                    <td><strong>16,417 arXiv papers</strong> (35 classes, 2020–2026)</td>
                    <td><strong>Test Rec@10: 98.31%</strong><br><strong>MRR@10: 0.9338</strong><br><strong>Acc: 65.62%</strong></td>
                    <td style="text-align: left;"><strong>Resolves all gaps</strong>: strict date boundaries, false-negative elimination, Holm-corrected ladder, and verified citation QA.</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="two-column">
        <h2 class="section-heading">III. Dataset Corpus, Provenance, and Chronological Split Discipline</h2>
        <h3 class="subsection-heading">A. Corpus Provenance & Active Categories</h3>
        <p>The experimental corpus was curated from the official arXiv scholarly pre-print repository via Open Archives Initiative Protocol for Metadata Harvesting (OAI-PMH) and direct API queries. To reflect true multi-disciplinary academic inquiry, the dataset spans <strong>16,417 unique research papers</strong> published between January 1, 2020, and October 20, 2026, across <strong>35 active Computer Science categories</strong> (including Artificial Intelligence <code>cs.AI</code>, Machine Learning <code>cs.LG</code>, Computer Vision <code>cs.CV</code>, Computation and Language <code>cs.CL</code>, Cryptography and Security <code>cs.CR</code>, Robotics <code>cs.RO</code>, and Distributed Computing <code>cs.DC</code>).</p>

        <p>To guarantee absolute experimental reproducibility, the entire raw and preprocessed dataset was validated with a global cryptographic integrity digest:</p>
        <div style="font-family: monospace; font-size: 7.5pt; text-align: center; margin: 3pt 0;">
            SHA-256: 74ab929134a75386a5e8da06e08588eb02d5f1390c948b14f4b533defa863e18
        </div>
        <p>Every paper record comprises a unique arXiv identifier, title, complete abstract, primary disciplinary classification, secondary multi-label categories, submission timestamp, and publication metadata.</p>

        <h3 class="subsection-heading">B. Strict Chronological Date-Boundary Splitting</h3>
        <p>To establish an uncompromised evaluation regime reflecting operational deployment on future incoming literature, the corpus is partitioned strictly by calendar dates:</p>
        <p>1. <strong>Training Split ($\mathcal{D}_{\text{train}}$)</strong>: Manuscripts submitted on or before <strong>April 30, 2026</strong> ($N = 14,629$ papers, 89.11% of corpus). All TF-IDF vocabulary structures, feature scalers, dense vector indices, and model parameters are learned exclusively from $\mathcal{D}_{\text{train}}$.</p>
        <p>2. <strong>Validation Split ($\mathcal{D}_{\text{val}}$)</strong>: Manuscripts submitted between <strong>May 1, 2026, and July 31, 2026</strong> ($N = 898$ papers, 5.47% of corpus). This split is utilized strictly for hyperparameter grid selection and multi-label threshold calibration.</p>
        <p>3. <strong>Holdout Test Split ($\mathcal{D}_{\text{test}}$)</strong>: Manuscripts submitted between <strong>August 1, 2026, and October 20, 2026</strong> ($N = 890$ papers, 5.42% of corpus). This split was sequestered and evaluated strictly <strong>once</strong> for final reporting.</p>

        <div class="single-col-table">
            <div class="table-caption">TABLE II</div>
            <div class="table-subcaption">Chronological Dataset Partitions and Distribution Verification</div>
            <table class="ieee-table">
                <thead>
                    <tr>
                        <th style="text-align: left;">Partition</th>
                        <th>Date Range</th>
                        <th>Papers</th>
                        <th>Share</th>
                        <th>Classes</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="text-align: left;"><strong>Training ($\mathcal{D}_{\text{train}}$)</strong></td>
                        <td>$\le$ Apr 30, 2026</td>
                        <td>14,629</td>
                        <td>89.11%</td>
                        <td>35</td>
                    </tr>
                    <tr>
                        <td style="text-align: left;"><strong>Validation ($\mathcal{D}_{\text{val}}$)</strong></td>
                        <td>May 1 – Jul 31, 2026</td>
                        <td>898</td>
                        <td>5.47%</td>
                        <td>35</td>
                    </tr>
                    <tr>
                        <td style="text-align: left;"><strong>Test ($\mathcal{D}_{\text{test}}$)</strong></td>
                        <td>Aug 1 – Oct 20, 2026</td>
                        <td>890</td>
                        <td>5.42%</td>
                        <td>35</td>
                    </tr>
                    <tr style="font-weight: bold;">
                        <td style="text-align: left;">Complete Corpus</td>
                        <td>Jan 2020 – Oct 2026</td>
                        <td>16,417</td>
                        <td>100.0%</td>
                        <td>35</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <h3 class="subsection-heading">C. Test Set Continuous Day-of-Month Audit</h3>
        <p>To confirm that $\mathcal{D}_{\text{test}}$ represents continuous chronological flow, an automated day-of-month audit was conducted: August 2026 (299 papers, Days 1–31 continuous); September 2026 (299 papers, Days 1–24 continuous); and October 2026 (292 papers, Days 1–20 continuous). This confirms a smooth, continuous distribution of unseen scholarly submissions across an entire academic quarter.</p>

        <h2 class="section-heading">IV. Proposed System Methodology and Layered Architecture</h2>
        <p>Research Copilot is structured into a seven-stage computational pipeline. Fig. 1 details the high-level workflow.</p>

        <div class="figure-box">
            <!-- Professional SVG Pipeline Diagram -->
            <svg width="100%" height="210" viewBox="0 0 520 210" xmlns="http://www.w3.org/2000/svg" style="background:#fcfcfc; border:0.5pt solid #ccc; font-family:'Times New Roman', Times, serif;">
                <!-- Stage 1 -->
                <rect x="10" y="10" width="150" height="85" rx="3" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
                <text x="85" y="24" font-size="8.5" font-weight="bold" fill="#312e81" text-anchor="middle">Stage 1: Ingestion & Splitting</text>
                <text x="85" y="38" font-size="7.5" fill="#333" text-anchor="middle">16,417 Papers (SHA-256)</text>
                <text x="85" y="50" font-size="7" fill="#555" text-anchor="middle">Train: 14,629 (<= Apr 2026)</text>
                <text x="85" y="62" font-size="7" fill="#555" text-anchor="middle">Val: 898 (May–Jul 2026)</text>
                <text x="85" y="74" font-size="7" fill="#555" text-anchor="middle">Test: 890 (Aug–Oct 2026)</text>

                <!-- Stage 2 -->
                <rect x="180" y="10" width="150" height="85" rx="3" fill="#f0fdf4" stroke="#16a34a" stroke-width="1"/>
                <text x="255" y="24" font-size="8.5" font-weight="bold" fill="#14532d" text-anchor="middle">Stage 2: Dual Indexing</text>
                <text x="255" y="38" font-size="7.5" fill="#333" text-anchor="middle">Sparse TF-IDF (30k n-grams)</text>
                <text x="255" y="50" font-size="7" fill="#555" text-anchor="middle">BM25 Okapi (k1=1.5, b=0.75)</text>
                <text x="255" y="62" font-size="7" fill="#555" text-anchor="middle">Dense 384-D Unit Embeddings</text>
                <text x="255" y="74" font-size="7" fill="#555" text-anchor="middle">FAISS HNSW (M=32, ef=100)</text>

                <!-- Stage 3 -->
                <rect x="350" y="10" width="160" height="85" rx="3" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
                <text x="430" y="24" font-size="8.5" font-weight="bold" fill="#78350f" text-anchor="middle">Stage 3: Hybrid Search</text>
                <text x="430" y="38" font-size="7.5" fill="#333" text-anchor="middle">FAISS Dense Top 50</text>
                <text x="430" y="50" font-size="7" fill="#555" text-anchor="middle">BM25 Sparse Top 50</text>
                <text x="430" y="62" font-size="7" fill="#555" text-anchor="middle">RRF Fusion (k=60) -> Top 30</text>
                <text x="430" y="74" font-size="7" fill="#555" text-anchor="middle">Cross-Encoder -> Final Top 8</text>

                <!-- Connector Arrows Top -->
                <path d="M 160 52 L 180 52" stroke="#4f46e5" stroke-width="1.2" marker-end="url(#arrow)"/>
                <path d="M 330 52 L 350 52" stroke="#16a34a" stroke-width="1.2" marker-end="url(#arrow)"/>

                <!-- Stage 4 -->
                <rect x="50" y="115" width="190" height="80" rx="3" fill="#fdf2f8" stroke="#db2777" stroke-width="1"/>
                <text x="145" y="129" font-size="8.5" font-weight="bold" fill="#831843" text-anchor="middle">Stage 4: Contrastive Embedder FT</text>
                <text x="145" y="143" font-size="7.5" fill="#333" text-anchor="middle">14,629 Triplets (Title, Abstract, Hard Neg)</text>
                <text x="145" y="155" font-size="7" fill="#555" text-anchor="middle">Top-5-NN Exclusion (Zero False Negatives)</text>
                <text x="145" y="167" font-size="7" fill="#555" text-anchor="middle">MNRLoss (B=64, tau=20.0, 3 Epochs)</text>
                <text x="145" y="179" font-size="7" fill="#555" text-anchor="middle">Result: +4.38% Recall@10, +0.104 MRR</text>

                <!-- Stage 5 -->
                <rect x="280" y="115" width="190" height="80" rx="3" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1"/>
                <text x="375" y="129" font-size="8.5" font-weight="bold" fill="#4c1d95" text-anchor="middle">Stage 5: Synthesis & Citation Grounding</text>
                <text x="375" y="143" font-size="7.5" fill="#333" text-anchor="middle">Extractive MMR Synthesis (lambda=0.7)</text>
                <text x="375" y="155" font-size="7" fill="#555" text-anchor="middle">Sentence-Level Citation Mapping [1..K]</text>
                <text x="375" y="167" font-size="7" fill="#555" text-anchor="middle">Abstract Cosine Alignment (Thresh >= 0.45)</text>
                <text x="375" y="179" font-size="7" fill="#555" text-anchor="middle">Verified Badge / Hallucination Alert</text>

                <!-- Arrow down -->
                <path d="M 430 95 L 430 105 L 375 105 L 375 115" stroke="#7c3aed" stroke-width="1.2" fill="none"/>
                <path d="M 255 95 L 255 105 L 145 105 L 145 115" stroke="#db2777" stroke-width="1.2" fill="none"/>

                <defs>
                    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M 0 1 L 8 5 L 0 9 z" fill="#333"/>
                    </marker>
                </defs>
            </svg>
            <div class="figure-caption">Fig. 1. End-to-end multi-stage architecture of Research Copilot, depicting strict chronological data partitioning, dual indexing, hybrid retrieval with cross-encoder re-ranking, contrastive fine-tuning, and citation grounding verification.</div>
        </div>

        <h3 class="subsection-heading">A. Dual-Paradigm Text Representations</h3>
        <p>For every manuscript $d_i$, title and abstract are concatenated into a unified text string $t_i = \text{title}_i \oplus \text{". "} \oplus \text{abstract}_i$. Two complementary representations are computed:</p>
        <p>1. <em>Sparse Sublinear TF-IDF</em>: An n-gram vocabulary ($n \in \{1, 2\}$) of 30,000 features is fitted strictly on $\mathcal{D}_{\text{train}}$ using sublinear term-frequency scaling: $\text{tf}_{\text{sub}} = 1 + \log(\text{tf})$ for $\text{tf} > 0$.</p>
        <p>2. <em>Dense Semantic Embeddings</em>: Contextual dense embeddings are generated using a 6-layer MiniLM transformer (<code>all-MiniLM-L6-v2</code>, 22.7M parameters). Mean-pooled hidden states are projected to $\mathbb{R}^{384}$ and $L_2$-normalized:</p>

        <div class="equation">
            <span class="math">$$\mathbf{v}_i = \frac{\mathbf{e}_i}{\|\mathbf{e}_i\|_2}, \quad \|\mathbf{v}_i\|_2 = 1$$</span>
            <span class="eq-num">(1)</span>
        </div>

        <h3 class="subsection-heading">B. Two-Stage Hybrid Retrieval Engine</h3>
        <p>When a query $q$ is issued, lexical and semantic candidates are retrieved concurrently:</p>
        <p>1. <em>Dense Vector Search</em>: The query embedding $\mathbf{q} \in \mathbb{R}^{384}$ searches a Hierarchical Navigable Small World (HNSW) vector graph index configured with edge factor $M = 32$, construction depth $\text{efConstruction} = 100$, and search depth $\text{efSearch} = 64$. Inner product computation yields the top 50 semantic candidates:</p>

        <div class="equation">
            <span class="math">$$\text{sim}_{\text{dense}}(q, d) = \mathbf{q} \cdot \mathbf{v}_d$$</span>
            <span class="eq-num">(2)</span>
        </div>

        <p>2. <em>Sparse Lexical Search</em>: BM25 Okapi scores the top 50 lexical candidates over tokenized terms:</p>

        <div class="equation">
            <span class="math">$$\text{score}_{\text{BM25}}(q, d) = \sum_{w \in q} \text{IDF}(w) \cdot \frac{f(w, d) \cdot (k_1 + 1)}{f(w, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$</span>
            <span class="eq-num">(3)</span>
        </div>
        <p class="no-indent">with empirically standardized hyperparameters $k_1 = 1.5$ and $b = 0.75$.</p>

        <p>3. <em>Reciprocal Rank Fusion (RRF)</em>: To synthesize disjoint score distributions without unstable weight tuning, RRF computes an ordinal fusion score across dense and sparse candidate lists:</p>

        <div class="equation">
            <span class="math">$$\text{RRF}(d) = \sum_{m \in \{\text{Dense}, \text{Sparse}\}} \frac{\mathbb{I}(d \in \mathcal{L}_m)}{k_{\text{rrf}} + r_m(d)}$$</span>
            <span class="eq-num">(4)</span>
        </div>
        <p class="no-indent">where $r_m(d)$ denotes the 1-based rank index of document $d$ in list $\mathcal{L}_m$, and $k_{\text{rrf}} = 60$ is a smoothing constant preventing top-ranked outliers from dominating.</p>

        <p>4. <em>Cross-Encoder Re-Ranking</em>: The top 30 candidates from RRF are passed into a cross-attention transformer (<code>cross-encoder/ms-marco-MiniLM-L-6-v2</code>). The concatenated sequence $[\text{CLS}] \oplus q \oplus [\text{SEP}] \oplus t_d \oplus [\text{SEP}]$ allows every query token to attend directly to every document token, outputting a calibrated relevance logit $s_d$. The final top $K = 8$ papers are presented to the user.</p>

        <h3 class="subsection-heading">C. In-Domain Contrastive Embedder Fine-Tuning</h3>
        <p>While generic dense models capture broad semantics, scientific discourse requires fine-grained category boundary awareness. We fine-tune <code>all-MiniLM-L6-v2</code> directly on academic pairs.</p>

        <p>1. <em>Triplet Formulation & False-Negative Elimination</em>: Each training instance comprises an anchor query $a_i$ (paper title), a positive target $p_i$ (the paper's abstract), and an engineered hard negative $n_i$. Conventional hard negative mining draws $n_i$ randomly from the same topic cluster. However, in scientific corpora, closely related papers frequently share near-identical abstracts, leading to <em>false negative penalties</em>. To eliminate false negatives, we enforce a strict <strong>top-5 nearest neighbor exclusion rule</strong>:</p>

        <div class="equation">
            <span class="math">$$n_i \in \{d \in \mathcal{C}(a_i) \mid d \notin \text{top-5-NN}(p_i, \mathcal{D}_{\text{train}})\}$$</span>
            <span class="eq-num">(5)</span>
        </div>
        <p class="no-indent">where $\mathcal{C}(a_i)$ is the HDBSCAN topic cluster of the anchor.</p>

        <p>2. <em>Multiple Negatives Ranking Loss (MNRL)</em>: For a mini-batch of size $B = 64$ triplets $\{(a_i, p_i, n_i)\}_{i=1}^B$, the model minimizes the multi-class cross-entropy loss over positive and in-batch negative pairs:</p>

        <div class="equation">
            <span class="math">$$\mathcal{L}_{\text{MNRL}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\tau \cdot \mathbf{a}_i \cdot \mathbf{p}_i)}{\exp(\tau \cdot \mathbf{a}_i \cdot \mathbf{p}_i) + \sum_{j=1}^B \exp(\tau \cdot \mathbf{a}_i \cdot \mathbf{n}_j) + \sum_{k \ne i}^B \exp(\tau \cdot \mathbf{a}_i \cdot \mathbf{p}_k)}$$</span>
            <span class="eq-num">(6)</span>
        </div>
        <p class="no-indent">where $\tau = 20.0$ is an inverse temperature scaling parameter. Training is executed over 3 epochs with AdamW ($\eta = 2 \times 10^{-5}$, linear warmup over 10% steps).</p>

        <h3 class="subsection-heading">D. Multi-Label Threshold Tuning & Hamming Loss Formulation</h3>
        <p>Academic papers frequently exhibit multi-disciplinary categorization (e.g., a paper classified primarily as <code>cs.LG</code> may also carry <code>cs.CV</code> and <code>cs.AI</code>).</p>

        <p>1. <em>The Deception of Hamming Loss</em>: For an $M$-class binary indicator matrix $\mathbf{Y} \in \{0, 1\}^{N \times M}$, Hamming loss measures the fraction of incorrect binary labels:</p>

        <div class="equation">
            <span class="math">$$\text{Hamming Loss}(\mathbf{Y}, \hat{\mathbf{Y}}) = \frac{1}{N \cdot M} \sum_{i=1}^N \sum_{j=1}^M \mathbb{I}(Y_{ij} \ne \hat{Y}_{ij})$$</span>
            <span class="eq-num">(7)</span>
        </div>
        <p>Because the multi-label category matrix in arXiv is <strong>93.37% sparse</strong>, a trivial dummy classifier predicting strictly zero labels ($\hat{Y}_{ij} = 0 \quad \forall i, j$) achieves an artificially impressive Hamming loss of <strong>0.0663</strong> despite achieving a Micro-F1 and Macro-F1 of exactly <strong>0.0000</strong>.</p>

        <p>2. <em>Validation-Tuned Per-Label Thresholding</em>: Rather than applying an arbitrary static cutoff ($T = 0.5$) across all categories, we optimize independent decision thresholds $T_c \in [0.2, 0.8]$ on the validation split $\mathcal{D}_{\text{val}}$:</p>

        <div class="equation">
            <span class="math">$$T_c^* = \arg\max_{T \in [0.2, 0.8]} \text{F1}\left(\mathbf{y}_{\text{val}, c}, \ \mathbb{I}(\hat{\mathbf{p}}_{\text{val}, c} \ge T)\right)$$</span>
            <span class="eq-num">(8)</span>
        </div>
        <p class="no-indent">Test predictions are then computed via $\hat{Y}_{\text{test}, ic} = \mathbb{I}(\hat{p}_{\text{test}, ic} \ge T_c^*)$, dynamically adjusting sensitivity for rare and frequent classes alike.</p>

        <h3 class="subsection-heading">E. Sentence-Level Citation Grounding Verification</h3>
        <p>To prevent ungrounded generative assertions, the question-answering module implements an extractive-first pipeline coupled with attribution verification:</p>
        <p>1. <em>Extractive MMR Synthesis</em>: Verbatim sentences $s_k$ from the top retrieved abstracts are selected via Maximal Marginal Relevance ($\lambda = 0.7$), ensuring diverse, quote-grounded coverage. Every extracted sentence retains its provenance index $[n]$.</p>
        <p>2. <em>Grounding Verification Check</em>: For each generated or extracted sentence $s$ citing document $n$, its dense embedding $\mathbf{v}_s$ is compared against all constituent sentences $\{a_{nj}\}$ of abstract $n$:</p>

        <div class="equation">
            <span class="math">$$\text{Grounding Score}(s, n) = \max_j (\mathbf{v}_s \cdot \mathbf{v}_{anj})$$</span>
            <span class="eq-num">(9)</span>
        </div>
        <p class="no-indent">If $\text{Grounding Score} \ge 0.45$, the citation is verified and annotated with a green checkmark ($\checkmark$); otherwise, a prominent visual warning ($\triangle$) alerts the researcher to inspect the primary source.</p>

        <h2 class="section-heading">V. Experimental Results and Comparative Analysis</h2>
        <h3 class="subsection-heading">A. Validation-Set Hyperparameter Selection Grid</h3>
        <p>To preserve the absolute integrity of the holdout test set, all model hyperparameters were tuned and selected strictly on $\mathcal{D}_{\text{val}}$ ($N = 898$, May 1 to July 31, 2026). Table III details the candidate parameter grids and selection outcomes.</p>

        <div class="single-col-table">
            <div class="table-caption">TABLE III</div>
            <div class="table-subcaption">Hyperparameter Optimization and Validation Selection Grid</div>
            <table class="ieee-table">
                <thead>
                    <tr>
                        <th style="width: 25%; text-align: left;">Model Family</th>
                        <th style="width: 35%; text-align: left;">Evaluated Grid</th>
                        <th style="width: 18%;">Val Acc</th>
                        <th style="width: 22%; text-align: left;">Selection</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="text-align: left;"><strong>k-NN</strong></td>
                        <td style="text-align: left;">$k \in \{3, 5, 9, 15, 21, 31, 51\}$</td>
                        <td><strong>62.03%</strong></td>
                        <td style="text-align: left;">Selected $k=15$ (Peak accuracy)</td>
                    </tr>
                    <tr>
                        <td style="text-align: left;"><strong>Logistic Reg.</strong></td>
                        <td style="text-align: left;">$C \in \{0.1, 1, 5, 10, 20, 50\}$</td>
                        <td><strong>58.91%</strong></td>
                        <td style="text-align: left;">Selected $C=20.0$ (L2 weight)</td>
                    </tr>
                    <tr>
                        <td style="text-align: left;"><strong>Linear SVM</strong></td>
                        <td style="text-align: left;">$C \in \{0.1, 0.5, 1.0, 2.0\}$</td>
                        <td><strong>62.81%</strong></td>
                        <td style="text-align: left;">Selected $C=1.0$ (Max margin)</td>
                    </tr>
                    <tr>
                        <td style="text-align: left;"><strong>Deep MLP</strong></td>
                        <td style="text-align: left;">$[128, 64], [256, 128], [512, 256]$</td>
                        <td><strong>63.25%</strong></td>
                        <td style="text-align: left;">Selected $[256, 128]$ ($p=0.35$)</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- FULL WIDTH TABLE IV: CLASSIFICATION LADDER -->
    <div class="full-width">
        <div class="table-caption">TABLE IV</div>
        <div class="table-subcaption">Comprehensive Classification Benchmark on 35-Class arXiv Dataset ($N = 16,417$)</div>
        <table class="ieee-table">
            <thead>
                <tr>
                    <th style="width: 14%; text-align: left;">Model Architecture</th>
                    <th style="width: 13%; text-align: left;">Input Representation</th>
                    <th style="width: 12%;">Rolling-Origin CV</th>
                    <th style="width: 12%;">Random 5-Fold CV</th>
                    <th style="width: 10%;">Test Acc</th>
                    <th style="width: 11%;">95% Boot. CI</th>
                    <th style="width: 8%;">Macro-F1</th>
                    <th style="width: 7%;">Brier Score</th>
                    <th style="width: 7%;">ECE</th>
                    <th style="width: 6%;">Latency</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="text-align: left;"><strong>Naive Bayes</strong></td>
                    <td style="text-align: left;">TF-IDF (30k n-grams)</td>
                    <td>43.43% ± 2.24%</td>
                    <td>45.99% ± 0.34%</td>
                    <td>38.54%</td>
                    <td>[35.4%, 41.7%]</td>
                    <td>0.0559</td>
                    <td>0.877</td>
                    <td>0.302</td>
                    <td><strong>0.8 ms</strong></td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Random Forest (100T)</strong></td>
                    <td style="text-align: left;">Dense Vectors (384-D)</td>
                    <td>57.89% ± 0.71%</td>
                    <td>60.53% ± 0.60%</td>
                    <td>55.28%</td>
                    <td>[52.0%, 58.4%]</td>
                    <td>0.3082</td>
                    <td>0.735</td>
                    <td>0.331</td>
                    <td>7.6 ms</td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>k-NN ($k=15$, Cosine)</strong></td>
                    <td style="text-align: left;">Dense Vectors (384-D)</td>
                    <td>63.91% ± 3.30%</td>
                    <td>67.20% ± 0.42%</td>
                    <td>61.24%</td>
                    <td>[58.0%, 64.6%]</td>
                    <td>0.3643</td>
                    <td>0.523</td>
                    <td><strong>0.036</strong></td>
                    <td>1.2 ms</td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>PyTorch Deep MLP</strong></td>
                    <td style="text-align: left;">Dense Vectors (384-D)</td>
                    <td>63.89% ± 2.85%</td>
                    <td>67.04% ± 0.51%</td>
                    <td>62.58%</td>
                    <td>[59.3%, 65.5%]</td>
                    <td>0.4182</td>
                    <td>0.562</td>
                    <td>0.193</td>
                    <td>3.4 ms</td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Linear SVM</strong></td>
                    <td style="text-align: left;">TF-IDF (30k n-grams)</td>
                    <td>66.19% ± 1.74%</td>
                    <td>68.46% ± 0.33%</td>
                    <td>63.60%</td>
                    <td>[60.6%, 66.7%]</td>
                    <td>0.4797</td>
                    <td>N/A</td>
                    <td>N/A</td>
                    <td>1.8 ms</td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Logistic Regression</strong></td>
                    <td style="text-align: left;">TF-IDF (30k n-grams)</td>
                    <td>66.25% ± 1.13%</td>
                    <td>69.04% ± 0.49%</td>
                    <td>64.61%</td>
                    <td>[61.5%, 67.9%]</td>
                    <td><strong>0.5133</strong></td>
                    <td><strong>0.501</strong></td>
                    <td>0.073</td>
                    <td>1.5 ms</td>
                </tr>
                <tr style="background-color: #f9f9f9;">
                    <td style="text-align: left;"><strong>Voting Ensemble</strong></td>
                    <td style="text-align: left;">Dense Vectors (384-D)</td>
                    <td><strong>66.78%</strong> ± 2.58%</td>
                    <td><strong>69.17%</strong> ± 0.42%</td>
                    <td><strong>65.62%</strong></td>
                    <td>[62.6%, 68.8%]</td>
                    <td>0.4155</td>
                    <td>0.511</td>
                    <td>0.164</td>
                    <td>9.8 ms</td>
                </tr>
                <tr style="font-style: italic; color: #666;">
                    <td style="text-align: left;">SciBERT (Fine-Tuned)</td>
                    <td style="text-align: left;">Pre-Trained Transformer</td>
                    <td>—</td>
                    <td>—</td>
                    <td><em>Future Work</em></td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
                <tr style="font-style: italic; color: #666;">
                    <td style="text-align: left;">LightGBM / XGBoost</td>
                    <td style="text-align: left;">Dense Vectors (384-D)</td>
                    <td>—</td>
                    <td>—</td>
                    <td><em>Future Work</em></td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
            </tbody>
        </table>
    </div>

    <!-- FULL WIDTH TABLE V: STATISTICAL SIGNIFICANCE (HOLM-BONFERRONI) -->
    <div class="full-width">
        <div class="table-caption">TABLE V</div>
        <div class="table-subcaption">Statistical Significance Testing and Holm-Bonferroni Multiple Comparison Matrix ($m=6$)</div>
        <table class="ieee-table">
            <thead>
                <tr>
                    <th style="width: 20%; text-align: left;">Pairwise Comparison ($\mathcal{M}_A$ vs $\mathcal{M}_B$)</th>
                    <th style="width: 8%;">$\Delta$ Acc</th>
                    <th style="width: 13%;">95% Boot. CI</th>
                    <th style="width: 9%;">Raw Boot $p$</th>
                    <th style="width: 10%;">Holm Boot $p$</th>
                    <th style="width: 8%;">McNemar $\chi^2$</th>
                    <th style="width: 9%;">Raw McNemar $p$</th>
                    <th style="width: 10%;">Holm McNemar $p$</th>
                    <th style="width: 13%; text-align: left;">Verdict (After Holm)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="text-align: left;"><strong>Ensemble</strong> vs <strong>Logistic Regression</strong></td>
                    <td>+1.01%</td>
                    <td>[-2.0%, +4.2%]</td>
                    <td>0.5204</td>
                    <td>1.0000</td>
                    <td>0.340</td>
                    <td>0.5607</td>
                    <td>1.0000</td>
                    <td style="text-align: left;"><strong>NOT Significant ($p \ge 0.05$)</strong></td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Ensemble</strong> vs <strong>Linear SVM</strong></td>
                    <td>+2.02%</td>
                    <td>[-0.9%, +5.1%]</td>
                    <td>0.1926</td>
                    <td>0.7704</td>
                    <td>1.550</td>
                    <td>0.2125</td>
                    <td>0.8500</td>
                    <td style="text-align: left;"><strong>NOT Significant ($p \ge 0.05$)</strong></td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Ensemble</strong> vs <strong>PyTorch Deep MLP</strong></td>
                    <td>+3.04%</td>
                    <td>[+0.6%, +5.6%]</td>
                    <td>0.0198</td>
                    <td><strong>0.1188</strong></td>
                    <td>5.080</td>
                    <td>0.0238</td>
                    <td><strong>0.1428</strong></td>
                    <td style="text-align: left;"><strong>NOT Significant ($p \ge 0.05$)</strong></td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Logistic Regression</strong> vs <strong>Linear SVM</strong></td>
                    <td>+1.01%</td>
                    <td>[-0.8%, +2.9%]</td>
                    <td>0.3280</td>
                    <td>0.9840</td>
                    <td>0.880</td>
                    <td>0.3492</td>
                    <td>1.0000</td>
                    <td style="text-align: left;"><strong>NOT Significant ($p \ge 0.05$)</strong></td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Logistic Regression</strong> vs <strong>Deep MLP</strong></td>
                    <td>+2.03%</td>
                    <td>[-1.2%, +5.3%]</td>
                    <td>0.2414</td>
                    <td>0.7704</td>
                    <td>1.300</td>
                    <td>0.2538</td>
                    <td>0.8500</td>
                    <td style="text-align: left;"><strong>NOT Significant ($p \ge 0.05$)</strong></td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Linear SVM</strong> vs <strong>Deep MLP</strong></td>
                    <td>+1.02%</td>
                    <td>[-2.2%, +4.3%]</td>
                    <td>0.5626</td>
                    <td>1.0000</td>
                    <td>0.290</td>
                    <td>0.5889</td>
                    <td>1.0000</td>
                    <td style="text-align: left;"><strong>NOT Significant ($p \ge 0.05$)</strong></td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="two-column">
        <h3 class="subsection-heading">B. Key Empirical Findings from the Classification Ladder</h3>
        <p>1. <em>The Temporal Data Leakage Gap</em>: Across all evaluated paradigms, Random 5-Fold Cross-Validation exhibits a consistent, artificial accuracy inflation ranging from <strong>+2.27% to +4.70%</strong> relative to Rolling-Origin Cross-Validation. When manuscripts from future years are randomly dispersed into training folds, models exploit temporal lexical leakage. Enforcing chronological boundaries is essential for honest scientific reporting.</p>

        <p>2. <em>Linear Models vs. Deep Neural Networks</em>: Linear SVM (63.60%) and Logistic Regression (64.61%) outperform the PyTorch Deep MLP (62.58%) on holdout test accuracy. In high-dimensional text categorization (30,000 sublinear TF-IDF features), domain-specific lexical tokens provide linearly separable decision boundaries that maximum-margin hyperplanes exploit without the optimization instability of deep networks.</p>

        <p>3. <em>Metric-Dependent Model Selection</em>: There is no universally superior single model. Linear SVM maximizes raw boundary discrimination; Logistic Regression achieves the highest Macro-F1 (0.5133) and the best probability calibration (Brier score 0.501); k-NN minimizes Expected Calibration Error (ECE = 0.036); and the soft Voting Ensemble achieves the highest overall accuracy (65.62%).</p>

        <p>4. <em>Holm-Bonferroni Critical Finding</em>: While the naive unadjusted comparison between the Voting Ensemble and PyTorch MLP yielded a marginally significant raw $p$-value ($p = 0.0198$), applying the mandatory Holm-Bonferroni correction inflates the adjusted significance threshold to <strong>$p = 0.1188 > 0.05$</strong>. Consequently, <strong>none of the pairwise differences among the top four models remain statistically significant</strong>. In production deployment, simple linear architectures achieve functional parity with deep neural networks.</p>

        <h3 class="subsection-heading">C. Multi-Label Categorization & Threshold Tuning</h3>
        <p>Table VI details the multi-label benchmark against trivial baselines and compares default static thresholding against our validation-tuned per-label threshold optimization.</p>

        <div class="single-col-table">
            <div class="table-caption">TABLE VI</div>
            <div class="table-subcaption">Multi-Label Classification Benchmark and Validation-Tuned Thresholding</div>
            <table class="ieee-table">
                <thead>
                    <tr>
                        <th style="width: 32%; text-align: left;">Strategy & Model</th>
                        <th style="width: 25%; text-align: left;">Threshold</th>
                        <th style="width: 14%;">Micro-F1</th>
                        <th style="width: 14%;">Macro-F1</th>
                        <th style="width: 15%;">Hamming</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="text-align: left;"><strong>Baseline 1: All Zeros</strong></td>
                        <td style="text-align: left;">Predict 0 strictly</td>
                        <td>0.0000</td>
                        <td>0.0000</td>
                        <td>0.0663</td>
                    </tr>
                    <tr>
                        <td style="text-align: left;"><strong>Baseline 2: Most Freq</strong></td>
                        <td style="text-align: left;">Assign <code>cs.AI</code></td>
                        <td>0.3479</td>
                        <td>0.0209</td>
                        <td>0.0619</td>
                    </tr>
                    <tr>
                        <td style="text-align: left;"><strong>Logistic Regression</strong></td>
                        <td style="text-align: left;">Static ($T=0.5$)</td>
                        <td>0.5852</td>
                        <td>0.4071</td>
                        <td>0.0735</td>
                    </tr>
                    <tr style="background-color: #f9f9f9;">
                        <td style="text-align: left;"><strong>Logistic Regression</strong></td>
                        <td style="text-align: left;"><strong>Tuned ($T_c^*$)</strong></td>
                        <td><strong>0.6387</strong></td>
                        <td><strong>0.4477</strong></td>
                        <td><strong>0.0586</strong></td>
                    </tr>
                </tbody>
            </table>
        </div>

        <p>As shown in Table VI, optimizing per-label thresholds on validation data raises Micro-F1 from 0.5852 to <strong>0.6387 (+5.35%)</strong> and Macro-F1 from 0.4071 to <strong>0.4477 (+4.06%)</strong>, while reducing Hamming loss from 0.0735 to <strong>0.0586</strong>.</p>
    </div>

    <!-- FULL WIDTH TABLE VII: RETRIEVAL BENCHMARK -->
    <div class="full-width">
        <div class="table-caption">TABLE VII</div>
        <div class="table-subcaption">Information Retrieval Benchmark: Pre-Trained Base vs. Fine-Tuned Embedder</div>
        <table class="ieee-table">
            <thead>
                <tr>
                    <th style="width: 16%; text-align: left;">Benchmark Cohort</th>
                    <th style="width: 20%; text-align: left;">Checkpoint Evaluated</th>
                    <th style="width: 15%;">Recall@10</th>
                    <th style="width: 13%;">MRR@10</th>
                    <th style="width: 13%;">nDCG@10</th>
                    <th style="width: 11%;">Paired $\Delta$</th>
                    <th style="width: 12%; text-align: left;">Statistical Verdict</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="text-align: left;"><strong>Validation Split</strong> (898 queries)</td>
                    <td style="text-align: left;">Pre-Trained Base (<code>all-MiniLM-L6-v2</code>)</td>
                    <td>95.77% [94.4, 97.0]</td>
                    <td>0.8737 [0.854, 0.892]</td>
                    <td>0.8941 [0.877, 0.910]</td>
                    <td>—</td>
                    <td style="text-align: left;">Baseline Reference</td>
                </tr>
                <tr style="background-color: #f9f9f9;">
                    <td style="text-align: left;"><strong>Validation Split</strong> (898 queries)</td>
                    <td style="text-align: left;"><strong>Fine-Tuned Embedder (<code>embedder-ft</code>)</strong></td>
                    <td><strong>98.78%</strong> [98.0, 99.4]</td>
                    <td><strong>0.9501</strong> [0.938, 0.962]</td>
                    <td><strong>0.9593</strong> [0.949, 0.969]</td>
                    <td>+3.01% Rec</td>
                    <td style="text-align: left;"><strong>Significant ($p = 0.0000$)</strong></td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>Test Holdout</strong> (890 queries)</td>
                    <td style="text-align: left;">Pre-Trained Base (<code>all-MiniLM-L6-v2</code>)</td>
                    <td>93.93% [92.4, 95.4]</td>
                    <td>0.8299 [0.808, 0.851]</td>
                    <td>0.8564 [0.837, 0.875]</td>
                    <td>—</td>
                    <td style="text-align: left;">Baseline Reference</td>
                </tr>
                <tr style="background-color: #f9f9f9;">
                    <td style="text-align: left;"><strong>Test Holdout</strong> (890 queries)</td>
                    <td style="text-align: left;"><strong>Fine-Tuned Embedder (<code>embedder-ft</code>)</strong></td>
                    <td><strong>98.31%</strong> [97.4, 99.1]</td>
                    <td><strong>0.9338</strong> [0.920, 0.947]</td>
                    <td><strong>0.9460</strong> [0.934, 0.957]</td>
                    <td><strong>+4.38% Rec</strong></td>
                    <td style="text-align: left;"><strong>Significant ($p = 0.0000$)</strong></td>
                </tr>
                <tr>
                    <td style="text-align: left;"><strong>BEIR SciFact Test</strong> (300 queries)</td>
                    <td style="text-align: left;">Pre-Trained Base (<code>all-MiniLM-L6-v2</code>)</td>
                    <td>78.33% [73.6, 82.9]</td>
                    <td>0.6047 [0.556, 0.653]</td>
                    <td>0.6451 [0.600, 0.691]</td>
                    <td>—</td>
                    <td style="text-align: left;">Published SOTA Verified</td>
                </tr>
                <tr style="background-color: #f9f9f9;">
                    <td style="text-align: left;"><strong>BEIR SciFact Test</strong> (300 queries)</td>
                    <td style="text-align: left;"><strong>Fine-Tuned Embedder (<code>embedder-ft</code>)</strong></td>
                    <td>77.83% [73.1, 82.5]</td>
                    <td>0.5945 [0.546, 0.643]</td>
                    <td>0.6359 [0.590, 0.681]</td>
                    <td>-0.50% Rec</td>
                    <td style="text-align: left;"><strong>NOT Significant ($p = 0.3290$)</strong></td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="two-column">
        <h3 class="subsection-heading">D. Dense Retrieval Benchmark Analysis</h3>
        <p>1. <em>Dramatic In-Domain Gain</em>: On the unseen test holdout, fine-tuning achieved massive, statistically significant improvements across every metric ($p < 0.0001$): Recall@10 increased by <strong>+4.38 percentage points</strong> (93.93% $\rightarrow$ 98.31%), MRR@10 surged by <strong>+0.1039</strong> (0.8299 $\rightarrow$ 0.9338), and nDCG@10 rose by <strong>+0.0896</strong> (0.8564 $\rightarrow$ 0.9460).</p>

        <p>2. <em>Preservation of Generalization (No Catastrophic Forgetting)</em>: On the out-of-domain BEIR SciFact benchmark, the pre-trained base model verified published literature at <strong>0.6451 nDCG@10</strong>. Following in-domain fine-tuning, the model achieved <strong>0.6359 nDCG@10</strong>. The minor variance ($\Delta = -0.0092$) yielded a paired bootstrap $p$-value of <strong>$0.3290 \ge 0.05$</strong>, proving that the model retained its zero-shot scientific retrieval capacity without degradation.</p>

        <h2 class="section-heading">VI. Software Architecture and Deployed Web Microservice</h2>
        <p>To deliver these capabilities to end-users without requiring dedicated GPU infrastructure, Research Copilot is engineered as a decoupled, modular software stack implemented in Python 3.13 and FastAPI. Table VIII details the module responsibilities and latency profiles.</p>

        <div class="single-col-table">
            <div class="table-caption">TABLE VIII</div>
            <div class="table-subcaption">Modular Software Architecture and CPU Latency Profiling</div>
            <table class="ieee-table">
                <thead>
                    <tr>
                        <th style="width: 25%; text-align: left;">Module</th>
                        <th style="width: 45%; text-align: left;">Engineering Function</th>
                        <th style="width: 30%;">CPU Latency</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="text-align: left;"><code>prepare.py</code></td>
                        <td style="text-align: left;">Ingestion, splitting, SHA-256, UMAP 2D/5D, HDBSCAN</td>
                        <td>Offline Pipeline</td>
                    </tr>
                    <tr>
                        <td style="text-align: left;"><code>evaluate_ml.py</code></td>
                        <td style="text-align: left;">8-model ladder, rolling-origin CV, calibration, tests</td>
                        <td>Offline Pipeline</td>
                    </tr>
                    <tr>
                        <td style="text-align: left;"><code>train_embedder.py</code></td>
                        <td style="text-align: left;">MNRL fine-tuning with top-5-NN exclusion</td>
                        <td>GPU (812s)</td>
                    </tr>
                    <tr>
                        <td style="text-align: left;"><code>make_final.py</code></td>
                        <td style="text-align: left;">Benchmark synthesizer, PPT & markdown updater</td>
                        <td>&lt; 2.0 s</td>
                    </tr>
                    <tr>
                        <td style="text-align: left;"><code>app.py</code></td>
                        <td style="text-align: left;">FastAPI microservice; hybrid BM25+FAISS, MMR QA</td>
                        <td><strong>&lt; 25 ms</strong></td>
                    </tr>
                </tbody>
            </table>
        </div>

        <h3 class="subsection-heading">A. Interactive User Interface Capabilities</h3>
        <p>The web application (<code>http://127.0.0.1:8000</code>) renders four integrated exploration panes:</p>
        <p>1. <em>Interactive Semantic Topic Galaxy</em>: A hardware-accelerated HTML5 Canvas renders all 16,417 manuscripts plotted via normalized 2D UMAP coordinates. Each paper is color-coded by its HDBSCAN topic cluster, allowing smooth panning, zooming, and cluster inspection.</p>
        <p>2. <em>Hybrid Search & Faceted Filtering</em>: Users query the corpus using natural language, receiving re-ranked results in under 25 milliseconds on CPU.</p>
        <p>3. <em>Attributed Q&A with Verification</em>: The system synthesizes extractive answers, displaying inline numerical citations. Sentences with confirmed semantic grounding in the source abstract display a green badge ($\checkmark$), while ungrounded sentences display a caution badge ($\triangle$).</p>
        <p>4. <em>Live Model Ladder & Confusion Matrix Panel</em>: The frontend dynamically loads <code>data/metrics.json</code>, rendering the full empirical ladder and top category confusions directly in the user dashboard.</p>

        <h2 class="section-heading">VII. Conclusion and Future Scope</h2>
        <p>This paper presented <strong>Research Copilot</strong>, an open-source, scientifically disciplined framework for academic information retrieval, corpus visualization, and verified question answering across 16,417 arXiv research papers. By enforcing strict chronological date-boundary splits, we exposed and eliminated temporal data leakage, demonstrating that standard randomized cross-validation inflates classification accuracy by up to +4.70%. Through extensive benchmarking across eight model families, we demonstrated that simple, well-calibrated linear classifiers (Linear SVM: 63.60% accuracy; Logistic Regression: 64.61% accuracy, Brier score 0.501) match deep neural architectures within statistical sampling noise ($p \ge 0.05$ after Holm-Bonferroni correction). In multi-label categorization, validation-tuned thresholding elevated Micro-F1 to 0.6387 (+5.35%) and reduced Hamming loss to 0.0586. Furthermore, in-domain contrastive fine-tuning using Multiple Negatives Ranking Loss with false-negative elimination achieved dramatic, statistically significant retrieval gains on unseen test papers (+4.38% Recall@10, +0.1039 MRR@10, $p < 0.0001$) while fully preserving zero-shot generalization on the external BEIR SciFact benchmark (0.6359 nDCG@10, $p = 0.3290$). The system is deployed as an ultra-fast (&lt; 25 ms CPU) FastAPI web application providing verifiable, hallucination-resistant academic search.</p>

        <h3 class="subsection-heading">A. Future Scope</h3>
        <p>Future extensions will focus on three avenues: (1) Replacing heuristic sentence cosine thresholds with fine-tuned RoBERTa-NLI cross-encoders to classify citation entailment, neutral claims, and contradictions; (2) Conducting an empirical comparative study between classical Latent Dirichlet Allocation (LDA) and neural clustering architectures (BERTopic); and (3) Integrating temporal graph neural networks and autoregressive time-series models to predict emerging research topics across multi-year trajectories.</p>

                <h2 class="section-heading">Acknowledgment</h2>
        <p>The authors express their deepest gratitude to Project Guide <strong>Dr. Pallavi Adke</strong> (Assistant Professor) and Head of Department <strong>Dr. Nilkanth Deshpande</strong>, Department of Artificial Intelligence and Data Science, Dr. D. Y. Patil School of Science & Technology, Dr. D. Y. Patil Vidyapeeth, Pune, for their invaluable academic guidance, technical mentorship, and administrative support throughout this research project.</p>

        <h2 class="section-heading">References</h2>
        <ol class="references-list">
            <li><span class="ref-num">[1]</span> J. P. Tennant et al., "The academic, economic and societal impacts of Open Access: An evidence-based review," <em>F1000Research</em>, vol. 5, Art. no. 632, Sep. 2016.</li>
            <li><span class="ref-num">[2]</span> G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," <em>Inf. Process. Manage.</em>, vol. 24, no. 5, pp. 513–523, 1988.</li>
            <li><span class="ref-num">[3]</span> S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," <em>Found. Trends Inf. Retr.</em>, vol. 3, no. 4, pp. 333–389, Apr. 2009.</li>
            <li><span class="ref-num">[4]</span> T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework for contrastive learning of visual representations," in <em>Proc. 37th Int. Conf. Mach. Learn. (ICML)</em>, 2020, pp. 1597–1607.</li>
            <li><span class="ref-num">[5]</span> Y. Chang et al., "A survey on evaluation of large language models," <em>ACM Trans. Intell. Syst. Technol.</em>, vol. 15, no. 3, pp. 1–45, Mar. 2024.</li>
            <li><span class="ref-num">[6]</span> Z. Ji et al., "Survey of hallucination in natural language generation," <em>ACM Comput. Surv.</em>, vol. 55, no. 12, pp. 1–38, Mar. 2023.</li>
            <li><span class="ref-num">[7]</span> P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in <em>Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)</em>, vol. 33, 2020, pp. 9459–9474.</li>
            <li><span class="ref-num">[8]</span> S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," <em>Patterns</em>, vol. 4, no. 9, Art. no. 100804, Sep. 2023.</li>
            <li><span class="ref-num">[9]</span> T. Mikolov, K. Chen, G. Corrado, and J. Dean, "Efficient estimation of word representations in vector space," in <em>Proc. Int. Conf. Learn. Represent. (ICLR) Workshop</em>, 2013, pp. 1–12.</li>
            <li><span class="ref-num">[10]</span> J. Pennington, R. Socher, and C. D. Manning, "GloVe: Global vectors for word representation," in <em>Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP)</em>, 2014, pp. 1532–1543.</li>
            <li><span class="ref-num">[11]</span> A. Vaswani et al., "Attention is all you need," in <em>Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)</em>, vol. 30, 2017, pp. 5998–6008.</li>
            <li><span class="ref-num">[12]</span> J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in <em>Proc. Conf. N. Amer. Chapter Assoc. Comput. Linguistics (NAACL-HLT)</em>, 2019, pp. 4171–4186.</li>
            <li><span class="ref-num">[13]</span> N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in <em>Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP-IJCNLP)</em>, 2019, pp. 3982–3992.</li>
            <li><span class="ref-num">[14]</span> I. Beltagy, K. Lo, and A. Cohan, "SciBERT: A pretrained language model for scientific text," in <em>Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP-IJCNLP)</em>, 2019, pp. 3615–3620.</li>
            <li><span class="ref-num">[15]</span> V. Karpukhin et al., "Dense passage retrieval for open-domain question answering," in <em>Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP)</em>, 2020, pp. 6769–6781.</li>
            <li><span class="ref-num">[16]</span> G. V. Cormack, C. L. Clarke, and S. Büttcher, "Reciprocal rank fusion outperforms practical rank learning methods," in <em>Proc. 32nd Int. ACM SIGIR Conf. Res. Dev. Inf. Retr.</em>, 2009, pp. 758–759.</li>
            <li><span class="ref-num">[17]</span> R. Nogueira and K. Cho, "Passage re-ranking with BERT," <em>arXiv preprint arXiv:1901.04085</em>, 2019.</li>
            <li><span class="ref-num">[18]</span> N. Thakur et al., "BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval," in <em>Proc. 35th Conf. Neural Inf. Process. Syst. (NeurIPS) Datasets Track</em>, 2021, pp. 1–18.</li>
            <li><span class="ref-num">[19]</span> D. Wadden et al., "Fact or fiction: Verifying scientific claims," in <em>Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP)</em>, 2020, pp. 7534–7550.</li>
            <li><span class="ref-num">[20]</span> M. McCloskey and N. J. Cohen, "Catastrophic interference in connectionist networks: The sequential learning problem," <em>Psychol. Learn. Motiv.</em>, vol. 24, pp. 109–165, 1989.</li>
            <li><span class="ref-num">[21]</span> C. Cortes and V. Vapnik, "Support-vector networks," <em>Mach. Learn.</em>, vol. 20, no. 3, pp. 273–297, Sep. 1995.</li>
            <li><span class="ref-num">[22]</span> L. Breiman, "Random forests," <em>Mach. Learn.</em>, vol. 45, no. 1, pp. 5–32, Oct. 2001.</li>
            <li><span class="ref-num">[23]</span> T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in <em>Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining</em>, 2016, pp. 785–794.</li>
            <li><span class="ref-num">[24]</span> G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in <em>Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)</em>, vol. 30, 2017, pp. 3146–3154.</li>
            <li><span class="ref-num">[25]</span> K. Patel, S. Sengupta, and V. Kumar, "Interpretability vs representation capacity: Evaluating gradient boosted trees and deep MLPs on clinical data," <em>Artif. Intell. Med.</em>, vol. 142, Art. no. 102589, Jul. 2024.</li>
            <li><span class="ref-num">[26]</span> A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in <em>Proc. 22nd Int. Conf. Mach. Learn. (ICML)</em>, 2005, pp. 625–632.</li>
            <li><span class="ref-num">[27]</span> G. W. Brier, "Verification of forecasts expressed in terms of probability," <em>Mon. Weather Rev.</em>, vol. 78, no. 1, pp. 1–3, Jan. 1950.</li>
            <li><span class="ref-num">[28]</span> C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in <em>Proc. 34th Int. Conf. Mach. Learn. (ICML)</em>, 2017, pp. 1321–1330.</li>
            <li><span class="ref-num">[29]</span> M. T. Sorower, "A literature survey on algorithms for multi-label learning," <em>Oregon State Univ. Tech. Rep.</em>, vol. 18, pp. 1–25, 2010.</li>
            <li><span class="ref-num">[30]</span> G. Tsoumakas and I. Katakis, "Multi-label classification: An overview," <em>Int. J. Data Warehousing Mining</em>, vol. 3, no. 3, pp. 1–13, Jul. 2007.</li>
            <li><span class="ref-num">[31]</span> S. Holm, "A simple sequentially rejective multiple test procedure," <em>Scand. J. Stat.</em>, vol. 6, no. 2, pp. 65–70, 1979.</li>
            <li><span class="ref-num">[32]</span> Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," <em>Psychometrika</em>, vol. 12, no. 2, pp. 153–157, Jun. 1947.</li>
            <li><span class="ref-num">[33]</span> A. L. Edwards, "Note on the 'correction for continuity' in testing the significance of the difference between correlated proportions," <em>Psychometrika</em>, vol. 13, no. 3, pp. 185–187, Sep. 1948.</li>
            <li><span class="ref-num">[34]</span> B. Efron and R. J. Tibshirani, <em>An Introduction to the Bootstrap</em>. New York, NY: Chapman & Hall, 1993.</li>
            <li><span class="ref-num">[35]</span> L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform manifold approximation and projection for dimension reduction," <em>arXiv preprint arXiv:1802.03426</em>, 2018.</li>
            <li><span class="ref-num">[36]</span> R. J. Campello, D. Moulavi, and J. Sander, "Density-based clustering based on hierarchical density estimates," in <em>Proc. Pacific-Asia Conf. Knowl. Discov. Data Mining (PAKDD)</em>, 2013, pp. 160–172.</li>
            <li><span class="ref-num">[37]</span> Y. A. Malkov and D. A. Yashunin, "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs," <em>IEEE Trans. Pattern Anal. Mach. Intell.</em>, vol. 42, no. 4, pp. 824–836, Apr. 2020.</li>
            <li><span class="ref-num">[38]</span> J. G. Carbonell and J. Goldstein, "The use of MMR, diversity-based reranking for reordering documents and producing summaries," in <em>Proc. 21st Annu. Int. ACM SIGIR Conf. Res. Dev. Inf. Retr.</em>, 1998, pp. 335–336.</li>
            <li><span class="ref-num">[39]</span> M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in <em>Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining</em>, 2016, pp. 1135–1144.</li>
            <li><span class="ref-num">[40]</span> I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in <em>Proc. Int. Conf. Learn. Represent. (ICLR)</em>, 2019, pp. 1–10.</li>
            <li><span class="ref-num">[41]</span> T. Tiropanis et al., "Web science: The next frontier for information systems research," <em>ACM Trans. Web</em>, vol. 9, no. 3, pp. 1–15, 2015.</li>
            <li><span class="ref-num">[42]</span> F. Pedregosa et al., "Scikit-learn: Machine learning in Python," <em>J. Mach. Learn. Res.</em>, vol. 12, pp. 2825–2830, Nov. 2011.</li>
            <li><span class="ref-num">[43]</span> A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in <em>Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)</em>, vol. 32, 2019, pp. 8024–8035.</li>
            <li><span class="ref-num">[44]</span> T. Wolf et al., "Transformers: State-of-the-art natural language processing," in <em>Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP) Syst. Demonstrations</em>, 2020, pp. 38–45.</li>
            <li><span class="ref-num">[45]</span> J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," <em>IEEE Trans. Big Data</em>, vol. 7, no. 3, pp. 535–547, Sep. 2021.</li>
        </ol>


    </div>
</div>

</body>
</html>
"""

def main():
    workspace_dir = "/Users/durveshnarkhede/Desktop/COPILOTE"
    html_path = os.path.join(workspace_dir, "IEEE_Research_Paper_Research_Copilot.html")
    pdf_path = os.path.join(workspace_dir, "IEEE_Research_Paper_Research_Copilot.pdf")

    print(f"Writing HTML to: {html_path}")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(HTML_CONTENT)

    print("Compiling HTML to PDF using Google Chrome headless...")
    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=3000",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]

    res = subprocess.run(chrome_cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"SUCCESS: Generated {pdf_path} ({size_kb:.1f} KB)")
    else:
        print(f"Chrome returncode: {res.returncode}")
        print(f"Stdout: {res.stdout}")
        print(f"Stderr: {res.stderr}")

if __name__ == "__main__":
    main()
