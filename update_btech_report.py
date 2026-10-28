import re

with open("build_btech_report.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update cover page student table & guide/HOD
old_student_table = """        <table class="student-table">
            <tr>
                <td><strong>Durvessh Narkhede</strong></td>
                <td>Roll No.: 17</td>
                <td>Final Year B.Tech (AI & DS)</td>
            </tr>
            <tr>
                <td><strong>Om Shinde</strong></td>
                <td>Roll No.: 32</td>
                <td>Final Year B.Tech (AI & DS)</td>
            </tr>
            <tr>
                <td><strong>Vipul Sawarkar</strong></td>
                <td>Roll No.: 59</td>
                <td>Final Year B.Tech (AI & DS)</td>
            </tr>
            <tr>
                <td><strong>Atharv Sawant</strong></td>
                <td>Roll No.: 37</td>
                <td>Final Year B.Tech (AI & DS)</td>
            </tr>
        </table>

        <div class="report-declaration" style="margin-top: 14pt;">UNDER THE GUIDANCE OF</div>
        <div style="font-size: 11pt; font-weight: bold; margin-top: 3pt;">Project Guide & Research Mentorship Panel</div>
        <div style="font-size: 9.5pt; color: #4b5563;">Department of Artificial Intelligence & Data Science</div>

        <div class="academic-year">ACADEMIC YEAR: 2025–2026</div>"""

new_student_table = """        <table class="student-table">
            <tr>
                <td><strong>Durvesh Narkhede</strong></td>
                <td>Roll No.: 33</td>
                <td>Final Year B.Tech (AI & DS)</td>
            </tr>
            <tr>
                <td><strong>Atharv Sawant</strong></td>
                <td>Roll No.: 46</td>
                <td>Final Year B.Tech (AI & DS)</td>
            </tr>
            <tr>
                <td><strong>Vipul Sawarkar</strong></td>
                <td>Roll No.: 47</td>
                <td>Final Year B.Tech (AI & DS)</td>
            </tr>
            <tr>
                <td><strong>Om Shinde</strong></td>
                <td>Roll No.: 49</td>
                <td>Final Year B.Tech (AI & DS)</td>
            </tr>
        </table>

        <div class="report-declaration" style="margin-top: 8pt;">UNDER THE GUIDANCE OF</div>
        <div style="font-size: 11pt; font-weight: bold; margin-top: 2pt; color: #111827;">Dr. Pallavi Adke</div>
        <div style="font-size: 9pt; color: #4b5563;">Assistant Professor, Department of Artificial Intelligence & Data Science</div>

        <div class="report-declaration" style="margin-top: 6pt;">HEAD OF DEPARTMENT</div>
        <div style="font-size: 11pt; font-weight: bold; margin-top: 2pt; color: #111827;">Dr. Nilkanth Deshpande</div>
        <div style="font-size: 9pt; color: #4b5563;">Head of Department, Department of Artificial Intelligence & Data Science</div>

        <div class="academic-year" style="margin-top: 8pt;">ACADEMIC YEAR: 2025–2026</div>"""

code = code.replace(old_student_table, new_student_table)

# 2. Update Certificate and Declaration
old_cert = """<div class="cert-body">
    This is to certify that the B.Tech Project Report entitled <strong>"Research Copilot: A Chronologically-Disciplined Hybrid Information Retrieval, Contrastive Representation Learning, and Calibrated Classification Framework for Academic Discovery"</strong> is a bonafide record of authentic engineering project work carried out by <strong>Durvessh Narkhede (Roll No. 17)</strong>, <strong>Om Shinde (Roll No. 32)</strong>, <strong>Vipul Sawarkar (Roll No. 59)</strong>, and <strong>Atharv Sawant (Roll No. 37)</strong> in partial fulfillment of the requirements for the award of the degree of <em>Bachelor of Technology in Artificial Intelligence and Data Science</em> at Dr. D. Y. Patil School of Science & Technology, Dr. D. Y. Patil Vidyapeeth (DPU), Pimpri, Pune during the Academic Year 2025–2026.
</div>

<table class="sign-table">
    <tr>
        <td>
            _______________________________<br>
            Project Guide<br>
            Dept. of AI & Data Science
        </td>
        <td>
            _______________________________<br>
            Head of Department<br>
            Department of AI & Data Science<br>
            DPU School of Science & Technology
        </td>
    </tr>
    <tr>
        <td>
            _______________________________<br>
            External Examiner<br>
            University Examination Panel
        </td>
        <td>
            _______________________________<br>
            Director / Principal<br>
            Dr. D. Y. Patil School of Science & Technology
        </td>
    </tr>
</table>

<div class="decl-heading" style="margin-top: 25pt;">CANDIDATES' DECLARATION & ACKNOWLEDGMENT</div>
<p>
The project team hereby declares that the software implementation, experimental evaluations, and technical documentation presented in this project report represent original engineering work conducted at the Department of Artificial Intelligence and Data Science, Dr. D. Y. Patil School of Science & Technology, Pune. All external open-access corpora (including the Kaggle arXiv OAI snapshot, the live Cornell arXiv REST API, and the BEIR SciFact retrieval benchmark) as well as peer-reviewed scholarly literature have been formally cited. Sincere gratitude is extended to the faculty, laboratory technical staff, and leadership of Dr. D. Y. Patil School of Science & Technology for providing the computational infrastructure, academic encouragement, and research environment necessary to bring this engineering project to successful fruition.
</p>"""

new_cert = """<div class="cert-body">
    This is to certify that the B.Tech Project Report entitled <strong>"Research Copilot: A Chronologically-Disciplined Hybrid Information Retrieval, Contrastive Representation Learning, and Calibrated Classification Framework for Academic Discovery"</strong> is a bonafide record of authentic engineering project work carried out by <strong>Durvesh Narkhede (Roll No. 33)</strong>, <strong>Atharv Sawant (Roll No. 46)</strong>, <strong>Vipul Sawarkar (Roll No. 47)</strong>, and <strong>Om Shinde (Roll No. 49)</strong> in partial fulfillment of the requirements for the award of the degree of <em>Bachelor of Technology in Artificial Intelligence and Data Science</em> at Dr. D. Y. Patil School of Science & Technology, Dr. D. Y. Patil Vidyapeeth (DPU), Pimpri, Pune during the Academic Year 2025–2026.
</div>

<table class="sign-table">
    <tr>
        <td>
            _______________________________<br>
            <strong>Dr. Pallavi Adke</strong><br>
            Project Guide<br>
            Dept. of AI & Data Science
        </td>
        <td>
            _______________________________<br>
            <strong>Dr. Nilkanth Deshpande</strong><br>
            Head of Department<br>
            Department of AI & Data Science<br>
            DPU School of Science & Technology
        </td>
    </tr>
    <tr>
        <td>
            _______________________________<br>
            External Examiner<br>
            University Examination Panel
        </td>
        <td>
            _______________________________<br>
            Director / Principal<br>
            Dr. D. Y. Patil School of Science & Technology
        </td>
    </tr>
</table>

<div class="decl-heading" style="margin-top: 20pt;">CANDIDATES' DECLARATION & ACKNOWLEDGMENT</div>
<p>
The project team hereby declares that the software implementation, experimental evaluations, and technical documentation presented in this project report represent original engineering work conducted at the Department of Artificial Intelligence and Data Science, Dr. D. Y. Patil School of Science & Technology, Pune. All external open-access corpora (including the Kaggle arXiv OAI snapshot, the live Cornell arXiv REST API, and the BEIR SciFact retrieval benchmark) as well as peer-reviewed scholarly literature have been formally cited. Sincere gratitude is extended to our Project Guide, <strong>Dr. Pallavi Adke</strong>, and Head of Department, <strong>Dr. Nilkanth Deshpande</strong>, for their invaluable academic guidance, technical mentorship, and continuous encouragement throughout the conception, architecture, and deployment of this project. Sincere appreciation is also expressed to the faculty, laboratory technical staff, and institutional leadership of Dr. D. Y. Patil School of Science & Technology for providing the computational infrastructure and scholarly environment essential to this work.
</p>"""

code = code.replace(old_cert, new_cert)

# 3. Add CSS for Case Study Diagnostic Cards
case_css = """
    /* Case Study Diagnostic Cards */
    .case-card {
        background: #fdfdfd;
        border: 0.75pt solid #cbd5e1;
        border-left: 3.5pt solid #0f2b5c;
        border-radius: 3px;
        padding: 8pt 10pt;
        margin: 8pt 0 10pt 0;
        page-break-inside: avoid;
        font-size: 8.5pt;
        line-height: 1.4;
    }
    .case-title {
        font-weight: bold;
        color: #0f2b5c;
        font-size: 9.5pt;
        margin-bottom: 3pt;
        display: flex;
        justify-content: space-between;
    }
    .badge-grounded {
        background: #ecfdf5;
        color: #065f46;
        border: 0.5pt solid #a7f3d0;
        font-size: 7.5pt;
        font-weight: bold;
        padding: 1pt 4pt;
        border-radius: 2px;
    }
    .badge-unsupported {
        background: #fffbeb;
        color: #92400e;
        border: 0.5pt solid #fde68a;
        font-size: 7.5pt;
        font-weight: bold;
        padding: 1pt 4pt;
        border-radius: 2px;
    }
    .badge-refusal {
        background: #fef2f2;
        color: #991b1b;
        border: 0.5pt solid #fecaca;
        font-size: 7.5pt;
        font-weight: bold;
        padding: 1pt 4pt;
        border-radius: 2px;
    }
"""
code = code.replace("/* Figures */", case_css + "\n    /* Figures */")

# 4. Remove broken screenshot figures and replace with detailed Case Studies & System Tables
old_figures_block = """<div class="figure-box">
    <img src="__IMG_SEARCH_RESULTS__" alt="Search Results View">
    <div class="figure-caption">Figure 6.4: Deployed search interface showcasing ranked hybrid results, category tags, cross-encoder relevance scores, and instant visual canvas highlighting.</div>
</div>

<div class="figure-box">
    <img src="__IMG_ANSWER_CITATIONS__" alt="Cited Answer with Grounding Checkmarks">
    <div class="figure-caption">Figure 6.5: Cited Question Answering output displaying inline numerical citation badges `[1]`, `[2]` and verified green checkmarks confirming sentence-level semantic grounding against source abstracts.</div>
</div>"""

new_figures_block = """<div class="table-caption">Table 6.5: Measured Latency Profile by Subsystem Stage (50 Independent Production Queries)</div>
<table class="report-table">
    <thead>
        <tr>
            <th style="width: 25%;">Search & Q&A Stage</th>
            <th style="width: 15%;">Median (p50)</th>
            <th style="width: 15%;">Mean Latency</th>
            <th style="width: 15%;">95th Pct (p95)</th>
            <th style="width: 15%;">99th Pct (p99)</th>
            <th style="width: 15%;">Max Latency</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Query Dense Encoding</strong></td>
            <td><strong>5.25 ms</strong></td>
            <td>34.63 ms</td>
            <td>182.76 ms</td>
            <td>453.18 ms</td>
            <td>541.09 ms</td>
        </tr>
        <tr>
            <td><strong>FAISS HNSW Vector Search</strong></td>
            <td><strong>0.26 ms</strong></td>
            <td>0.32 ms</td>
            <td>0.42 ms</td>
            <td>1.65 ms</td>
            <td>2.70 ms</td>
        </tr>
        <tr>
            <td><strong>BM25 Okapi Sparse Search</strong></td>
            <td><strong>17.55 ms</strong></td>
            <td>17.67 ms</td>
            <td>22.23 ms</td>
            <td>25.40 ms</td>
            <td>26.59 ms</td>
        </tr>
        <tr>
            <td><strong>Reciprocal Rank Fusion (RRF)</strong></td>
            <td><strong>0.02 ms</strong></td>
            <td>0.02 ms</td>
            <td>0.03 ms</td>
            <td>0.03 ms</td>
            <td>0.03 ms</td>
        </tr>
        <tr>
            <td><strong>Cross-Encoder Re-Ranking</strong></td>
            <td><strong>191.66 ms</strong></td>
            <td>194.08 ms</td>
            <td>231.10 ms</td>
            <td>240.12 ms</td>
            <td>241.02 ms</td>
        </tr>
        <tr style="background-color: #f1f5f9; font-weight: bold;">
            <td><strong>End-to-End Search (/api/search)</strong></td>
            <td><strong>216.81 ms</strong></td>
            <td>246.72 ms</td>
            <td>431.79 ms</td>
            <td>681.73 ms</td>
            <td>744.34 ms</td>
        </tr>
        <tr style="background-color: #e2e8f0; font-weight: bold;">
            <td><strong>End-to-End Q&A (/api/ask)</strong></td>
            <td><strong>354.12 ms</strong></td>
            <td>399.32 ms</td>
            <td>722.72 ms</td>
            <td>927.55 ms</td>
            <td>957.90 ms</td>
        </tr>
    </tbody>
</table>

<div class="table-caption">Table 6.6: System Hardware Footprint, Resident Memory Allocation, and File Sizes</div>
<table class="report-table">
    <thead>
        <tr>
            <th style="width: 25%;">Component / Artifact</th>
            <th style="width: 25%;">Measured Footprint</th>
            <th style="width: 50%;">Technical Description & Verification</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Resident Set Size (RAM)</strong></td>
            <td><strong>659.05 MB</strong></td>
            <td class="left-align">Physical RAM consumed with full 16,417-paper corpus, FAISS HNSW, BM25 index, and dual transformers resident.</td>
        </tr>
        <tr>
            <td><strong>Corpus Database (<code>papers.pkl</code>)</strong></td>
            <td><strong>24,285,306 bytes (23.16 MB)</strong></td>
            <td class="left-align">Serialized DataFrame containing 16,417 verified papers across 15 schema columns.</td>
        </tr>
        <tr>
            <td><strong>FAISS Vector Index (<code>index_ft.faiss</code>)</strong></td>
            <td><strong>29,685,998 bytes (28.31 MB)</strong></td>
            <td class="left-align">Hierarchical Navigable Small World (HNSW, M=32) index storing 16,417 384-D normalized vectors.</td>
        </tr>
        <tr>
            <td><strong>Fine-Tuned Embedder Directory</strong></td>
            <td><strong>91,611,557 bytes (87.37 MB)</strong></td>
            <td class="left-align">SafeTensors checkpoint of contrastively fine-tuned all-MiniLM-L6-v2 model.</td>
        </tr>
        <tr>
            <td><strong>Chronological Date Splits</strong></td>
            <td><strong>169,527 bytes</strong></td>
            <td class="left-align">Strict date partitions: Train (14,629), Validation (898), and Test Holdout (890).</td>
        </tr>
    </tbody>
</table>

<div class="section-header">6.5 Production Case Studies, Grounding Diagnostics, and Refusal Verification</div>
<p class="no-indent">
To evaluate the end-to-end question answering pipeline in practice, four production query case studies, one out-of-domain refusal case, and one domain disambiguation failure case were systematically audited:
</p>

<div class="case-card">
    <div class="case-title">
        <span>Case 1: "Contrastive learning for graph neural networks"</span>
        <span class="badge-grounded">Top Score: +8.1913 | Cat: cs.LG</span>
    </div>
    <div><strong>Top Paper Retrieved:</strong> <em>"Cross-view Self-Supervised Learning on Heterogeneous Graph Neural Network via Bootstrapping"</em> (arXiv: 2201.03340)</div>
    <div style="margin: 4pt 0;"><strong>Generated Cited Answer:</strong> In this paper, we propose the novel Label Contrastive Coding based Graph Neural Network (LCGNN) to utilize label information more effectively and comprehensively. [2] A crucial step in contrastive learning is the creation of 'augmented' graphs from the input graphs. [4] A large number of graph contrastive learning approaches have shown promising performance for representation learning on graphs, which train models by maximizing agreement between original graphs and their augmented views. [6]</div>
    <div><strong>Sentence-Level Grounding Checks:</strong></div>
    <ul>
        <li>Sentence 1: Similarity = 0.00 &rarr; <span class="badge-unsupported">Unsupported (0.00)</span></li>
        <li>Sentence 2: Similarity = 0.30 &rarr; <span class="badge-unsupported">Unsupported (0.30)</span></li>
        <li>Sentence 3: Similarity = 0.67 &rarr; <span class="badge-grounded">✓ Grounded & Entailed (0.67 &ge; 0.45)</span></li>
    </ul>
</div>

<div class="case-card">
    <div class="case-title">
        <span>Case 2: "Diffusion models for high resolution image synthesis"</span>
        <span class="badge-grounded">Top Score: +9.8136 | Cat: cs.CV</span>
    </div>
    <div><strong>Top Paper Retrieved:</strong> <em>"Layered Diffusion Model for One-Shot High Resolution Text-to-Image Synthesis"</em> (arXiv: 2407.06079)</div>
    <div style="margin: 4pt 0;"><strong>Generated Cited Answer:</strong> We demonstrate that higher resolution synthesis can be achieved by layering convolutions at additional resolution scales, in contrast to other methods which require additional models for super-resolution synthesis. [1] Diffusion models (DMs) have recently gained attention with state-of-the-art performance in text-to-image synthesis. [2] Current large-scale diffusion models represent a giant leap forward in conditional image synthesis. [3]</div>
    <div><strong>Sentence-Level Grounding Checks:</strong></div>
    <ul>
        <li>Sentence 1: Similarity = 0.00 &rarr; <span class="badge-unsupported">Unsupported (0.00)</span></li>
        <li>Sentence 2: Similarity = 1.00 &rarr; <span class="badge-grounded">✓ Grounded & Entailed (1.00 &ge; 0.45)</span></li>
        <li>Sentence 3: Similarity = 0.58 &rarr; <span class="badge-grounded">✓ Grounded & Entailed (0.58 &ge; 0.45)</span></li>
    </ul>
</div>

<div class="case-card">
    <div class="case-title">
        <span>Case 3: "Parameter-efficient fine-tuning LoRA"</span>
        <span class="badge-grounded">Top Score: +9.9576 | Cat: cs.LG</span>
    </div>
    <div><strong>Top Paper Retrieved:</strong> <em>"LoRA-R: An Adaptive Rank Allocation Strategy for LoRA Fine-Tuning"</em> (arXiv: 2503.11664)</div>
    <div style="margin: 4pt 0;"><strong>Generated Cited Answer:</strong> Low-Rank Adaptation (LoRA) has become a widely used method for parameter-efficient fine-tuning of large-scale, pre-trained neural networks. [1] However, LoRA and its extensions face several challenges, including the need for rank adaptivity, robustness, and computational efficiency during the fine-tuning process. [1] Parameter-efficient fine-tuning (PEFT) approaches have emerged as a viable alternative. [2]</div>
    <div><strong>Sentence-Level Grounding Checks:</strong></div>
    <ul>
        <li>Sentence 1: Similarity = 0.00 &rarr; <span class="badge-unsupported">Unsupported (0.00)</span></li>
        <li>Sentence 2: Similarity = 1.00 &rarr; <span class="badge-grounded">✓ Grounded & Entailed (1.00 &ge; 0.45)</span></li>
        <li>Sentence 3: Similarity = 0.46 &rarr; <span class="badge-grounded">✓ Grounded & Entailed (0.46 &ge; 0.45)</span></li>
    </ul>
</div>

<div class="case-card">
    <div class="case-title">
        <span>Case 4: "Federated learning differential privacy"</span>
        <span class="badge-grounded">Top Score: +9.3286 | Cat: cs.LG</span>
    </div>
    <div><strong>Top Paper Retrieved:</strong> <em>"Evaluation of Differential Privacy Mechanisms on Federated Learning"</em> (arXiv: 2510.09691)</div>
    <div style="margin: 4pt 0;"><strong>Generated Cited Answer:</strong> Differential privacy is one of the methods to solve the problem of privacy protection in federated learning. [2] Therefore, we proposed an adaptive differential privacy method based on federated learning. [2] This paper presents a gradient leakage resilient approach to privacy-preserving federated learning with per training example-based client differential privacy, coined as Fed-CDP. [6]</div>
    <div><strong>Sentence-Level Grounding Checks:</strong></div>
    <ul>
        <li>Sentence 1: Similarity = 0.00 &rarr; <span class="badge-unsupported">Unsupported (0.00)</span></li>
        <li>Sentence 2: Similarity = 1.00 &rarr; <span class="badge-grounded">✓ Grounded & Entailed (1.00 &ge; 0.45)</span></li>
        <li>Sentence 3: Similarity = 0.70 &rarr; <span class="badge-grounded">✓ Grounded & Entailed (0.70 &ge; 0.45)</span></li>
    </ul>
</div>

<div class="case-card" style="border-left-color: #991b1b;">
    <div class="case-title">
        <span>Case 5: Out-of-Domain Refusal Protocol Verification</span>
        <span class="badge-refusal">Score: -10.7906 &lt; -5.0 Threshold &rarr; REFUSED</span>
    </div>
    <div><strong>Query Submitted:</strong> <em>"quantum teleportation recipe for baking banana bread in hypergravity"</em></div>
    <div style="margin: 4pt 0;"><strong>Top Candidate Retrieved:</strong> <em>"Multimarked Spatial Search by Continuous-Time Quantum Walk"</em> (Cross-Encoder Score: -10.7906)</div>
    <div><strong>System Behavior:</strong> Because the cross-encoder relevance score falls substantially below the calibrated refusal gatekeeper threshold of -5.0, the synthesis engine refuses execution: <code>{"answer": "", "confident": false, "mode": "none"}</code>. The web client renders: <em>"Refused (no confident papers found)"</em>, effectively eliminating generative hallucination on out-of-domain prompts.</div>
</div>

<div class="case-card" style="border-left-color: #d97706;">
    <div class="case-title">
        <span>Case 6: Semantic Disambiguation and Domain Boundary Failure Analysis</span>
        <span class="badge-unsupported">Score: +0.3402 | Ambiguous Keyword</span>
    </div>
    <div><strong>Query Submitted:</strong> <em>"Transformers in electric power transmission networks"</em></div>
    <div style="margin: 4pt 0;"><strong>Top Candidate Retrieved:</strong> <em>"Moving Target Defense for Robust Monitoring of Electric Grid Transformers in Adversarial Environments"</em> (arXiv: cs.GT)</div>
    <div><strong>Scientific Analysis:</strong> The user intends to find power engineering literature concerning physical high-voltage step-up/step-down magnetic transformers. Because the corpus is strictly bounded to Computer Science, the system retrieves game-theoretic cybersecurity and deep learning Transformer language model papers applied to grid telemetry data. All extracted claims receive near-zero grounding scores (0.00, 0.22, 0.27), illustrating domain boundary constraints.</div>
</div>"""

code = code.replace(old_figures_block, new_figures_block)

with open("build_btech_report.py", "w", encoding="utf-8") as f:
    f.write(code)

print("build_btech_report.py updated successfully!")
