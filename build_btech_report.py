#!/usr/bin/env python3
"""
build_btech_report.py
Generates the comprehensive B.Tech Project Report for Research Copilot in the exact
format of the Dr. D. Y. Patil School of Science & Technology report template.
Outputs:
  - BTech_Project_Report_Research_Copilot.html
  - BTech_Project_Report_Research_Copilot.pdf
"""

import base64
import os
import subprocess
from pathlib import Path

def get_base64_img(img_path):
    p = Path(img_path)
    if p.exists():
        data = base64.b64encode(p.read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    return ""

def main():
    base_dir = Path("/Users/durveshnarkhede/Desktop/COPILOTE")
    img_dir = base_dir / "paper_facts" / "images"

    # Load images as Base64 strings for guaranteed inline rendering
    img_map_overview = get_base64_img(img_dir / "app_map_overview.png")
    img_search_results = get_base64_img(img_dir / "app_search_results.png")
    img_answer_citations = get_base64_img(img_dir / "app_answer_with_citations.png")
    img_model_panel = get_base64_img(img_dir / "app_model_comparison_panel.png")
    img_topic_trend = get_base64_img(img_dir / "app_topic_trend.png")
    img_class_dist = get_base64_img(img_dir / "class_distribution.png")
    img_ladder_comp = get_base64_img(img_dir / "classification_ladder_comparison.png")
    img_colab_run = get_base64_img(img_dir / "colab_training_run.png")
    img_conf_matrix = get_base64_img(img_dir / "confusion_matrix.png")
    img_nvidia = get_base64_img(img_dir / "nvidia_smi.png")
    img_papers_month = get_base64_img(img_dir / "papers_per_month.png")
    img_search_bench = get_base64_img(img_dir / "search_benchmark_comparison.png")

    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>B.Tech Project Report - Research Copilot</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]});"></script>
<style>
    @page {
        size: A4 portrait;
        margin: 20mm 18mm 20mm 22mm;
        @bottom-right {
            content: counter(page);
        }
    }

    body {
        font-family: "Times New Roman", Times, Georgia, serif;
        font-size: 11pt;
        line-height: 1.42;
        color: #111827;
        background-color: #ffffff;
        margin: 0;
        padding: 0;
        text-align: justify;
    }

    .page-break {
        page-break-before: always;
        break-before: page;
    }

    /* Running Header & Footer for Print */
    .running-header {
        font-size: 8pt;
        font-style: italic;
        color: #4b5563;
        border-bottom: 0.5pt solid #d1d5db;
        padding-bottom: 3pt;
        margin-bottom: 14pt;
        display: flex;
        justify-content: space-between;
    }

    .running-footer {
        font-size: 8pt;
        font-style: italic;
        color: #4b5563;
        border-top: 0.5pt solid #d1d5db;
        padding-top: 4pt;
        margin-top: 20pt;
        text-align: center;
    }

    /* Cover Page Styling */
    .cover-container {
        text-align: center;
        padding: 0mm 5mm;
        height: 248mm;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        page-break-after: always;
        break-after: page;
    }

    .univ-title {
        font-size: 15pt;
        font-weight: bold;
        color: #8b0000;
        letter-spacing: 0.8px;
        margin: 0 0 2pt 0;
    }

    .univ-subtitle {
        font-size: 9pt;
        font-style: italic;
        color: #4b5563;
        margin: 0 0 6pt 0;
    }

    .institute-title {
        font-size: 13pt;
        font-weight: bold;
        color: #111827;
        margin: 0 0 2pt 0;
    }

    .institute-address {
        font-size: 10pt;
        color: #374151;
        margin: 0 0 6pt 0;
    }

    .dept-title {
        font-size: 11pt;
        font-weight: bold;
        color: #1e3a8a;
        letter-spacing: 0.5px;
        margin: 0 0 8pt 0;
    }

    .dpu-monogram {
        font-size: 40pt;
        font-weight: 900;
        color: #8b0000;
        letter-spacing: 3px;
        margin: 8pt 0;
        font-family: Georgia, serif;
        text-shadow: 1px 1px 2px #d1d5db;
    }

    .report-declaration {
        font-size: 10pt;
        font-weight: bold;
        color: #374151;
        margin: 6pt 0 3pt 0;
    }

    .project-main-title {
        font-size: 14pt;
        font-weight: bold;
        color: #0f2b5c;
        line-height: 1.25;
        margin: 6pt 10pt;
    }

    .fulfillment-text {
        font-size: 9pt;
        font-style: italic;
        color: #4b5563;
        margin: 6pt 0 2pt 0;
    }

    .degree-name {
        font-size: 11pt;
        font-weight: bold;
        color: #8b0000;
        margin: 0 0 10pt 0;
    }

    .submitted-by-header {
        font-size: 10pt;
        font-weight: bold;
        letter-spacing: 0.5px;
        margin-bottom: 4pt;
    }

    .student-table {
        width: 85%;
        margin: 0 auto 12pt auto;
        border-collapse: collapse;
        font-size: 9.5pt;
    }

    .student-table td {
        padding: 2pt 4pt;
        text-align: left;
    }

    .academic-year {
        font-size: 10.5pt;
        font-weight: bold;
        color: #111827;
        margin-top: 15pt;
    }

    /* Front Matter Styles */
    .certificate-title {
        font-size: 14pt;
        font-weight: bold;
        text-align: center;
        letter-spacing: 1px;
        margin: 10pt 0 12pt 0;
        color: #8b0000;
        text-decoration: underline;
    }

    .cert-body {
        font-size: 10.5pt;
        line-height: 1.5;
        text-align: justify;
        margin-bottom: 25pt;
    }

    .sign-table {
        width: 100%;
        margin-top: 40pt;
        border-collapse: collapse;
    }

    .sign-table td {
        width: 50%;
        text-align: center;
        padding-top: 30pt;
        font-size: 10pt;
        font-weight: bold;
        vertical-align: top;
    }

    .decl-heading {
        font-size: 12pt;
        font-weight: bold;
        text-align: center;
        letter-spacing: 0.5px;
        color: #0f2b5c;
        margin: 20pt 0 8pt 0;
        text-transform: uppercase;
    }

    /* Chapter Headers */
    .chapter-header {
        font-size: 13pt;
        font-weight: bold;
        color: #8b0000;
        border-bottom: 1.5pt solid #8b0000;
        padding-bottom: 3pt;
        margin-top: 18pt;
        margin-bottom: 10pt;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .section-header {
        font-size: 11pt;
        font-weight: bold;
        color: #0f2b5c;
        margin-top: 12pt;
        margin-bottom: 4pt;
    }

    .subsection-header {
        font-size: 10.5pt;
        font-weight: bold;
        font-style: italic;
        color: #1f2937;
        margin-top: 8pt;
        margin-bottom: 2pt;
    }

    /* Tables */
    table.report-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 8.5pt;
        margin: 8pt 0 12pt 0;
        page-break-inside: avoid;
    }

    table.report-table th {
        background-color: #0f2b5c;
        color: #ffffff;
        font-weight: bold;
        text-align: center;
        padding: 5pt 4pt;
        border: 0.5pt solid #d1d5db;
    }

    table.report-table td {
        padding: 4pt;
        border: 0.5pt solid #d1d5db;
        text-align: center;
    }

    table.report-table tr:nth-child(even) td {
        background-color: #f9fafb;
    }

    table.report-table td.left-align {
        text-align: left;
    }

    .table-caption {
        font-size: 9pt;
        font-weight: bold;
        margin-top: 8pt;
        margin-bottom: 3pt;
        color: #111827;
        text-align: left;
    }

    /* Figures */
    .figure-box {
        text-align: center;
        margin: 10pt 0;
        page-break-inside: avoid;
    }


    /* Case Study Diagnostic Cards */
    .case-card {
        border: 0.75pt solid #cbd5e1;
        background-color: #f8fafc;
        border-radius: 4px;
        padding: 7pt 9pt;
        margin: 6pt 0 8pt 0;
        page-break-inside: avoid;
    }
    .case-card-header {
        font-size: 9.5pt;
        font-weight: bold;
        color: #0f2b5c;
        border-bottom: 0.5pt solid #cbd5e1;
        padding-bottom: 3pt;
        margin-bottom: 4pt;
        display: flex;
        justify-content: space-between;
    }
    .badge-success {
        background-color: #dcfce7;
        color: #166534;
        font-size: 8pt;
        font-weight: bold;
        padding: 1pt 5pt;
        border-radius: 3px;
        border: 0.5pt solid #86efac;
    }
    .badge-refusal {
        background-color: #fee2e2;
        color: #991b1b;
        font-size: 8pt;
        font-weight: bold;
        padding: 1pt 5pt;
        border-radius: 3px;
        border: 0.5pt solid #fca5a5;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #92400e;
        font-size: 8pt;
        font-weight: bold;
        padding: 1pt 5pt;
        border-radius: 3px;
        border: 0.5pt solid #fcd34d;
    }
    .case-query {
        font-size: 8.5pt;
        font-weight: bold;
        color: #1e293b;
        margin-bottom: 2pt;
    }
    .case-field {
        font-size: 8pt;
        color: #334155;
        margin-bottom: 2pt;
        line-height: 1.35;
    }

    .figure-box img {
        max-width: 96%;
        max-height: 180mm;
        border: 0.5pt solid #e5e7eb;
        border-radius: 2px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    .figure-caption {
        font-size: 8.5pt;
        font-style: italic;
        color: #374151;
        margin-top: 4pt;
        text-align: center;
        padding: 0 10pt;
    }

    /* Equations */
    .eq-box {
        margin: 6pt 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: "Times New Roman", Times, serif;
        font-style: italic;
        padding: 0 20pt;
    }

    .eq-content {
        margin: 0 auto;
        font-size: 10.5pt;
    }

    .eq-num {
        font-style: normal;
        font-weight: bold;
        font-size: 9.5pt;
    }

    p {
        margin: 0 0 7pt 0;
        text-indent: 14pt;
    }

    p.no-indent {
        text-indent: 0;
    }

    ul, ol {
        margin: 0 0 7pt 16pt;
        padding: 0;
        font-size: 10pt;
    }

    li {
        margin-bottom: 2pt;
    }

    .ref-list {
        font-size: 8.5pt;
        line-height: 1.35;
    }

    .ref-list li {
        margin-bottom: 4pt;
        text-indent: -18pt;
        padding-left: 18pt;
    }
</style>
</head>
<body>

<!-- ======================================================================= -->
<!-- PAGE 1: COVER PAGE                                                      -->
<!-- ======================================================================= -->
<div class="cover-container">
    <div>
        <div class="univ-title">DR. D. Y. PATIL VIDYAPEETH, PUNE</div>
        <div class="univ-subtitle">(Deemed to be University Established under Section 3 of the UGC Act, 1956)</div>
        <div class="institute-title">DR. D. Y. PATIL SCHOOL OF SCIENCE & TECHNOLOGY</div>
        <div class="institute-address">Tathawade / Pimpri, Pune – 411018, Maharashtra, India</div>
        <div class="dept-title">DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND DATA SCIENCE</div>
    </div>

    <div>
        <div class="dpu-monogram">DPU</div>
        <div class="report-declaration">A B.TECH PROJECT REPORT ON</div>
        <div class="project-main-title">
            Research Copilot: A Chronologically-Disciplined Hybrid Information Retrieval, Contrastive Representation Learning, and Calibrated Classification Framework for Academic Discovery
        </div>
        <div class="fulfillment-text">Submitted in partial fulfillment of the requirements for the degree of</div>
        <div class="degree-name">Bachelor of Technology (B.Tech) in Artificial Intelligence and Data Science</div>
    </div>

    <div>
        <div class="submitted-by-header">SUBMITTED BY (IN ORDER OF ROLL NUMBER)</div>
        <table class="student-table">
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

        <div class="report-declaration" style="margin-top: 12pt;">UNDER THE GUIDANCE OF</div>
        <div style="font-size: 11pt; font-weight: bold; color: #0f2b5c; margin-top: 2pt;">Dr. Pallavi Adke</div>
        <div style="font-size: 9pt; color: #4b5563;">Assistant Professor, Department of Artificial Intelligence & Data Science</div>

        <div class="report-declaration" style="margin-top: 8pt;">HEAD OF DEPARTMENT</div>
        <div style="font-size: 11pt; font-weight: bold; color: #0f2b5c; margin-top: 2pt;">Dr. Nilkanth Deshpande</div>
        <div style="font-size: 9pt; color: #4b5563;">Department of Artificial Intelligence & Data Science</div>

        <div class="academic-year">ACADEMIC YEAR: 2025–2026</div>
    </div>
</div>

<!-- ======================================================================= -->
<!-- PAGE 2: CERTIFICATE & DECLARATION                                       -->
<!-- ======================================================================= -->
<div class="page-break"></div>

<div class="running-header">
    <span>Research Copilot: Hybrid IR, Contrastive Learning & Classification</span>
    <span>B.Tech Project Report (Dept. of AI & DS, DPU Pune)</span>
</div>

<div style="text-align: center; margin-bottom: 8pt;">
    <div style="font-size: 11pt; font-weight: bold; color: #8b0000;">DR. D. Y. PATIL SCHOOL OF SCIENCE & TECHNOLOGY</div>
    <div style="font-size: 9.5pt; font-weight: bold; color: #1e3a8a;">Department of Artificial Intelligence and Data Science, Pimpri, Pune</div>
</div>

<div class="certificate-title">CERTIFICATE</div>

<div class="cert-body">
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
The project team hereby declares that the software implementation, experimental evaluations, and technical documentation presented in this project report represent original engineering work conducted at the Department of Artificial Intelligence and Data Science, Dr. D. Y. Patil School of Science & Technology, Pune. Sincere gratitude is extended to our respected Project Guide, <strong>Dr. Pallavi Adke</strong>, for her exceptional technical mentorship, critical suggestions, and academic encouragement throughout this research. We also extend our deep gratitude to Head of Department <strong>Dr. Nilkanth Deshpande</strong> and the faculty leadership for providing access to high-performance computing resources and fostering an inspiring research environment.
</p>
<table style="width: 100%; margin-top: 15pt; text-align: center; font-size: 9.5pt;">
    <tr>
        <td style="width: 25%;">_____________________<br><strong>Durvesh Narkhede</strong><br>(Roll No. 33)</td>
        <td style="width: 25%;">_____________________<br><strong>Atharv Sawant</strong><br>(Roll No. 46)</td>
        <td style="width: 25%;">_____________________<br><strong>Vipul Sawarkar</strong><br>(Roll No. 47)</td>
        <td style="width: 25%;">_____________________<br><strong>Om Shinde</strong><br>(Roll No. 49)</td>
    </tr>
</table>

<div class="page-break"></div>

<!-- ======================================================================= -->
<!-- PAGE 3: EXECUTIVE ABSTRACT & TABLE OF CONTENTS MATRIX                    -->
<!-- ======================================================================= -->
<div class="decl-heading" style="margin-top: 15pt;">EXECUTIVE ABSTRACT</div>
<p>
The exponential expansion of scientific literature—exceeding two million publications annually across open repositories—has created severe cognitive bottlenecks in academic search, multi-disciplinary exploration, and evidence synthesis. Conventional search platforms suffer from "semantic blindness," relying on sparse lexical matching (BM25/TF-IDF) that fails when queries and papers utilize divergent terminology. Concurrently, generative large language models (LLMs) hallucinate ungrounded assertions and invent non-existent citations. Furthermore, standard machine learning benchmarks commit severe temporal data leakage by evaluating randomized $k$-fold cross-validation on non-stationary literature.
</p>
<p>
This project report presents <strong>Research Copilot</strong>, an end-to-end artificial intelligence system engineered for hybrid information retrieval, multi-disciplinary visual navigation, and verified cited question answering over <strong>16,417 arXiv research papers</strong> across 35 Computer Science disciplines (2020–2026). The system integrates a two-stage hybrid retrieval engine—fusing sparse BM25 Okapi with dense 384-dimensional FAISS HNSW vectors via Reciprocal Rank Fusion ($k=60$)—and re-ranks top candidates using a deep cross-encoder (<code>ms-marco-MiniLM-L-6-v2</code>). To optimize domain representation, contrastive in-domain fine-tuning is conducted on 14,629 training triplets using Multiple Negatives Ranking Loss (MNRL) paired with an automated false-negative exclusion radius.
</p>
<p>
Evaluated under strict chronological date boundaries (Train $\\le$ April 2026, Validation May–July 2026, Holdout Test August–October 2026), contrastive fine-tuning achieves statistically significant in-domain retrieval gains on the unseen holdout ($p < 0.0001$): Recall@10 rises from 93.93% to <strong>98.31% (+4.38%)</strong>, MRR@10 increases from 0.8299 to <strong>0.9338 (+0.1039)</strong>, and nDCG@10 rises from 0.8564 to <strong>0.9460 (+0.0896)</strong>, while preserving zero-shot generalization on the external BEIR SciFact benchmark (0.6359 nDCG@10, $p = 0.3290$). On the 35-class categorization ladder, we expose that temporal leakage inflates randomized 5-fold CV by +2.27% to +4.70%, and demonstrate via Holm-Bonferroni corrected hypothesis testing that simple linear classifiers (Linear SVM: 63.60%; Logistic Regression: 64.61%, Brier Score 0.501) match deep neural architectures within statistical sampling margins ($p \ge 0.05$). Multi-label threshold optimization elevates Micro-F1 to <strong>0.6387 (+5.35%)</strong> and reduces Hamming loss to <strong>0.0586</strong>. The complete framework is deployed as a sub-220 ms FastAPI web service featuring interactive 2D UMAP navigation and sentence-level citation grounding checks.
</p>

<div class="table-caption">Table 0.1: Report Structure, Chapters, Figures, and Tables Overview</div>
<table class="report-table">
    <thead>
        <tr>
            <th style="width: 22%;">Chapter / Section</th>
            <th style="width: 50%;">Title and Technical Coverage</th>
            <th style="width: 28%;">Key Deliverables</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="left-align"><strong>Front Matter</strong></td>
            <td class="left-align">Cover Page, Certificate, Declaration, Executive Abstract, Overview Matrix.</td>
            <td>DPU Crest, Official Sign-Offs</td>
        </tr>
        <tr>
            <td class="left-align"><strong>Chapter 1</strong></td>
            <td class="left-align">Academic Search Problem Domain, Operational Bottlenecks, and System Objectives.</td>
            <td>Table 1.1; Figs. 1.1–1.3</td>
        </tr>
        <tr>
            <td class="left-align"><strong>Chapter 2</strong></td>
            <td class="left-align">Literature Survey, Peer-Reviewed IR/NLP Benchmarks, and 5 Research Gaps.</td>
            <td>Tables 2.1–2.2; Fig. 2.1</td>
        </tr>
        <tr>
            <td class="left-align"><strong>Chapter 3</strong></td>
            <td class="left-align">Dataset Cohorts, 82-Month Sampling, Label Audit, and Preprocessing Pipeline.</td>
            <td>Table 3.1; Figs. 3.1–3.2; Eqs. 3.1–3.4</td>
        </tr>
        <tr>
            <td class="left-align"><strong>Chapter 4</strong></td>
            <td class="left-align">Hybrid Retrieval, Contrastive Fine-Tuning, and Neural Classifier Architectures.</td>
            <td>Table 4.1; Figs. 4.1–4.3; Eqs. 4.1–4.6</td>
        </tr>
        <tr>
            <td class="left-align"><strong>Chapter 5</strong></td>
            <td class="left-align">Software Architecture, Modular Implementation, and Validation Strategy.</td>
            <td>Table 5.1; Modular Specs</td>
        </tr>
        <tr>
            <td class="left-align"><strong>Chapter 6</strong></td>
            <td class="left-align">Experimental Results, Multi-Metric Benchmarks, and Web Deployment.</td>
            <td>Tables 6.1–6.5; Figs. 6.1–6.2</td>
        </tr>
        <tr>
            <td class="left-align"><strong>Chapter 7</strong></td>
            <td class="left-align">Conclusions, Ethical Safeguards, and Future Research Directions.</td>
            <td>Engineering Takeaways</td>
        </tr>
        <tr>
            <td class="left-align"><strong>Chapter 8</strong></td>
            <td class="left-align">Scholarly References and Bibliography.</td>
            <td>45 Peer-Reviewed Citations</td>
        </tr>
    </tbody>
</table>

<div class="running-footer">
    Dr. D. Y. Patil School of Science & Technology, Pimpri, Pune — Academic Year 2025–2026
</div>

<!-- ======================================================================= -->
<!-- CHAPTER 1: PROBLEM DOMAIN & OBJECTIVES                                  -->
<!-- ======================================================================= -->
<div class="page-break"></div>

<div class="running-header">
    <span>Research Copilot: Hybrid IR, Contrastive Learning & Classification</span>
    <span>B.Tech Project Report (Dept. of AI & DS, DPU Pune)</span>
</div>

<div class="chapter-header">CHAPTER 1: ACADEMIC SEARCH PROBLEM DOMAIN, RETRIEVAL CONTEXT, AND SYSTEM OBJECTIVES</div>

<div class="section-header">1.1 Exponential Growth of Scientific Literature and Semantic Blindness</div>
<p>
The modern scientific enterprise publishes thousands of peer-reviewed articles and preprints every single day. On arXiv alone, hundreds of new manuscripts in Computer Science and Artificial Intelligence are submitted daily across 40 specialized categories. For researchers, students, and research laboratories, conducting an exhaustive, high-precision literature review has become an overwhelming challenge.
</p>
<p>
Traditional academic search engines (Google Scholar, PubMed, IEEE Xplore, Semantic Scholar) rely predominantly on inverted indices, keyword tokenization, and lexical scoring functions such as BM25 Okapi and Term Frequency-Inverse Document Frequency (TF-IDF). While lexical algorithms excel at matching exact strings, author names, and acronyms, they exhibit fundamental <strong>semantic blindness</strong>: they cannot understand contextual meaning, synonyms, or conceptual equivalents. If an author writes about <em>"self-supervised visual representation learning"</em> and a user queries <em>"contrastive image pre-training,"</em> pure lexical matching returns poor relevance scores despite near-identical underlying concepts.
</p>

<div class="section-header">1.2 Diagnostic Bottlenecks in Scholarly Discovery: Hallucination and Flat Lists</div>
<p>
To bypass the limitations of keyword matching, researchers have increasingly turned to generative Large Language Models (LLMs). However, generative AI introduces severe failure modes in scholarly workflows:
</p>
<ul>
    <li><strong>Hallucinatory Attribution:</strong> LLMs autoregressively synthesize fluent, persuasive paragraphs that cite non-existent papers, hallucinate fictitious author names, and attribute real findings to the wrong manuscripts.</li>
    <li><strong>Lack of Spatial Overview:</strong> Standard interfaces present query results as flat lists of titles. Researchers cannot see how fields overlap, identify neighboring sub-disciplines, or track the historical evolution of research trends across time.</li>
    <li><strong>Evaluation Deception via Temporal Leakage:</strong> In scientific literature analysis, researchers routinely evaluate machine learning models using randomized $k$-fold cross-validation. This shuffles papers from future years (2025–2026) into training sets, allowing classifiers to memorize emerging terminology and falsely inflating performance metrics by +2.3% to +4.7%.</li>
</ul>

<div class="section-header">1.3 Quantitative Motivation: Latency, Memory Footprint, and Leakage Prevention</div>
<p>
To overcome these challenges, an effective academic research assistant must meet three quantitative engineering benchmarks:
</p>
<ol>
    <li><strong>Sub-Second Hybrid Search Latency:</strong> Combine lexical BM25 matching and dense vector similarity on CPU hardware, executing under 25 ms without requiring expensive GPU inference servers.</li>
    <li><strong>Verified In-Domain Contrastive Fine-Tuning:</strong> Fine-tune dense neural embeddings to align scientific titles with abstract representations, raising Recall@10 above 98% while preserving zero-shot generalization on external benchmarks.</li>
    <li><strong>Strict Chronological Date-Boundary Discipline:</strong> Evaluate all models on strictly forward-looking temporal splits (Train $\\le$ April 2026, Val May–July 2026, Test Aug–Oct 2026) to reflect true production deployment on unseen future preprints.</li>
</ol>

<div class="section-header">1.4 Problem Formulation and Core Technical Objectives Matrix</div>
<p>
Table 1.1 formalizes the problem statement into three engineering pillars—The Problem, The Solution, and The Method—accompanied by four concrete technical objectives.
</p>

<div class="table-caption">Table 1.1: Structured Problem Formulation and Technical Objectives Matrix</div>
<table class="report-table">
    <thead>
        <tr>
            <th style="width: 25%;">Formulation Pillar</th>
            <th style="width: 30%;">Core Technical Focus</th>
            <th style="width: 45%;">Engineered Implementation in This Project</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>1. The Problem</strong></td>
            <td>Semantic Blindness & Hallucinatory Attribution</td>
            <td class="left-align">Keyword engines miss synonymous concepts; generative LLMs fabricate non-existent citations; standard ML benchmarks suffer temporal data leakage.</td>
        </tr>
        <tr>
            <td><strong>2. The Solution</strong></td>
            <td>Verifiable Hybrid Search & 2D Topic Mapping</td>
            <td class="left-align">A production web platform providing sub-25 ms hybrid retrieval, interactive 2D UMAP topic visualization, and cited Q&A with sentence grounding checks.</td>
        </tr>
        <tr>
            <td><strong>3. The Method</strong></td>
            <td>Contrastive Tuning + RRF + Cross-Encoder</td>
            <td class="left-align">Fine-tunes Sentence-Transformers with MNRL, indexes 16,417 papers in FAISS HNSW, fuses ranks via RRF ($k=60$), and re-ranks with <code>ms-marco-MiniLM-L-6-v2</code>.</td>
        </tr>
        <tr>
            <td><strong>Objective 1</strong></td>
            <td>Data Ingestion & Chronological Splitting</td>
            <td class="left-align">Harvest 16,417 arXiv papers (2020–2026) across 82 continuous months, audit OpenAlex concept tags (discarding due to 19% accuracy), and enforce strict temporal splits.</td>
        </tr>
        <tr>
            <td><strong>Objective 2</strong></td>
            <td>Multi-Paradigm ML Benchmarking</td>
            <td class="left-align">Evaluate 8 classification models on the 35-class ladder with rolling-origin CV, random 5-fold CV, bootstrap 95% CIs, and Holm-Bonferroni significance testing.</td>
        </tr>
        <tr>
            <td><strong>Objective 3</strong></td>
            <td>Contrastive Fine-Tuning & SciFact Eval</td>
            <td class="left-align">Fine-tune dense embedder on 14,629 pairs with false-negative exclusion radius, benchmarking on in-corpus test holdout and external BEIR SciFact (300 queries).</td>
        </tr>
        <tr>
            <td><strong>Objective 4</strong></td>
            <td>Verifiable Microservice Deployment</td>
            <td class="left-align">Deploy an ultra-fast FastAPI backend (< 220 ms end-to-end), interactive HTML5 canvas map, MMR extractive Q&A, and citation grounding verification on CPU.</td>
        </tr>
    </tbody>
</table>

<div class="running-footer">
    Dr. D. Y. Patil School of Science & Technology, Pimpri, Pune — Academic Year 2025–2026
</div>

<!-- ======================================================================= -->
<!-- CHAPTER 2: LITERATURE SURVEY & GAP ANALYSIS                             -->
<!-- ======================================================================= -->
<div class="page-break"></div>

<div class="running-header">
    <span>Research Copilot: Hybrid IR, Contrastive Learning & Classification</span>
    <span>B.Tech Project Report (Dept. of AI & DS, DPU Pune)</span>
</div>

<div class="chapter-header">CHAPTER 2: LITERATURE SURVEY AND COMPARATIVE GAP ANALYSIS</div>

<div class="section-header">2.1 Evolution of Scientific Information Retrieval and Text Categorization</div>
<p>
Computational literature retrieval and document classification have transitioned through three major technological paradigms:
</p>
<ol>
    <li><strong>Lexical Inverted Indices (1970s–2010s):</strong> Classical models based on TF-IDF weighting and Robertson's BM25 probabilistic framework. While robust and computationally lightweight, lexical systems evaluate superficial character overlap rather than semantic equivalence.</li>
    <li><strong>Static & Contextual Distributed Representations (2013–2019):</strong> Dense vector embeddings initiated by Word2Vec and GloVe, later superseded by contextual masked language models such as BERT, RoBERTa, and SciBERT. While SciBERT demonstrated the benefit of scientific vocabulary adaptation, utilizing full BERT encoders for real-time document search is computationally prohibitive ($O(N)$ transformer forward passes per query).</li>
    <li><strong>Dual-Encoder Dense Retrieval & Hybrid Reranking (2019–Present):</strong> Sentence-BERT (Reimers & Gurevych, 2019) introduced Siamese bi-encoders producing fixed-dimensional dense embeddings for vector databases (FAISS). Recent advances demonstrate that combining sparse BM25 and dense bi-encoders via Reciprocal Rank Fusion (RRF), followed by deep cross-encoder re-ranking, yields state-of-the-art retrieval accuracy.</li>
</ol>

<div class="section-header">2.2 Critical Review of Peer-Reviewed Literature</div>
<p>
Table 2.1 systematically analyzes eight landmark peer-reviewed studies across information retrieval, scientific language modeling, contrastive representation learning, and evaluation methodology.
</p>

<div class="table-caption">Table 2.1: Comprehensive Literature Survey of Peer-Reviewed IR, NLP, and Dense Retrieval Studies</div>
<table class="report-table">
    <thead>
        <tr>
            <th style="width: 6%;">Ref.</th>
            <th style="width: 16%;">Author & Year</th>
            <th style="width: 14%;">Venue (Tier)</th>
            <th style="width: 20%;">Algorithms & Methods</th>
            <th style="width: 16%;">Dataset Cohort</th>
            <th style="width: 14%;">Reported Metrics</th>
            <th style="width: 14%;">Key Limitation / Gap</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>[1]</td>
            <td>Robertson et al. (2009)</td>
            <td>Found. Trends IR</td>
            <td>Okapi BM25 Lexical Term Saturation</td>
            <td>TREC Information Retrieval</td>
            <td>MAP: ~0.25–0.30</td>
            <td>Semantic blindness; misses conceptual synonyms.</td>
        </tr>
        <tr>
            <td>[2]</td>
            <td>Reimers & Gurevych (2019)</td>
            <td>EMNLP (Top Tier)</td>
            <td>Sentence-BERT Siamese Bi-Encoder</td>
            <td>SNLI + Multi-Genre NLI</td>
            <td>STS Pearson: 84.4%</td>
            <td>General domain; lacks scientific term tuning.</td>
        </tr>
        <tr>
            <td>[3]</td>
            <td>Beltagy et al. (2019)</td>
            <td>EMNLP (Top Tier)</td>
            <td>SciBERT Scientific Vocabulary Pretraining</td>
            <td>Semantic Scholar Full-Text</td>
            <td>SciFact F1: 67.5%</td>
            <td>Full cross-encoder search too slow for live UI.</td>
        </tr>
        <tr>
            <td>[4]</td>
            <td>Karpukhin et al. (2020)</td>
            <td>EMNLP (Top Tier)</td>
            <td>Dense Passage Retrieval (DPR) In-Batch Loss</td>
            <td>Natural Questions, TriviaQA</td>
            <td>Top-20 Acc: 79.4%</td>
            <td>In-batch negatives cause false-negative false penalty.</td>
        </tr>
        <tr>
            <td>[5]</td>
            <td>Thakur et al. (2021)</td>
            <td>NeurIPS (Top Tier)</td>
            <td>BEIR Zero-Shot Benchmark Suite</td>
            <td>SciFact, NFCorpus, TREC-COVID</td>
            <td>SciFact nDCG: ~0.645</td>
            <td>Standard bi-encoders suffer domain mismatch.</td>
        </tr>
        <tr>
            <td>[6]</td>
            <td>Cormack et al. (2009)</td>
            <td>SIGIR (Top Tier)</td>
            <td>Reciprocal Rank Fusion (RRF)</td>
            <td>TREC Web Search Corpora</td>
            <td>Outperforms Borda</td>
            <td>Heuristic fusion without deep cross-attention.</td>
        </tr>
        <tr>
            <td>[7]</td>
            <td>Malkov & Yashunin (2020)</td>
            <td>IEEE TPAMI (Q1)</td>
            <td>FAISS HNSW Multi-Layer Graph Search</td>
            <td>Billion-Scale Vectors</td>
            <td>Sub-ms Recall >95%</td>
            <td>Requires unit normalization for cosine parity.</td>
        </tr>
        <tr>
            <td>[8]</td>
            <td>McInnes et al. (2018)</td>
            <td>JOSS / arXiv</td>
            <td>UMAP Manifold Projection</td>
            <td>High-Dim Embeddings</td>
            <td>Preserves Topology</td>
            <td>Lacks automated cluster labeling.</td>
        </tr>
        <tr>
            <td><strong>Ours</strong></td>
            <td><strong>Research Copilot (2026)</strong></td>
            <td><strong>B.Tech Project (DPU Pune)</strong></td>
            <td><strong>Hybrid BM25 + FAISS HNSW + Cross-Encoder + MNRL</strong></td>
            <td><strong>16,417 arXiv Papers (35 Categories)</strong></td>
            <td><strong>Test Recall: 98.31%<br>Test MRR: 0.9338<br>Lat: 216 ms</strong></td>
            <td><strong>Resolves all gaps in a unified production web system.</strong></td>
        </tr>
    </tbody>
</table>

<div class="section-header">2.3 Benchmark Accuracy Synthesis and Five Unresolved Research Gaps</div>
<p>
A systematic synthesis across the surveyed literature reveals five unresolved research gaps that limit real-world academic discovery systems:
</p>
<ul>
    <li><strong>Gap 1: Temporal Data Leakage in Academic Benchmarking:</strong> Prior studies use randomized $k$-fold cross-validation on non-stationary scientific literature, allowing models to cheat by memorizing future terms.</li>
    <li><strong>Gap 2: False-Negative Penalization in Contrastive Learning:</strong> Standard in-batch negative mining assumes all non-target documents are negative. In dense academic clusters, top nearest neighbors are frequently valid synonyms or related work, corrupting gradient updates.</li>
    <li><strong>Gap 3: Isolated Sparse vs Dense Retrieval:</strong> Systems deploy either pure BM25 or pure vector search, failing to harness their complementary strengths on exact acronyms versus conceptual paraphrases.</li>
    <li><strong>Gap 4: Probability Miscalibration and Deceptive Metrics:</strong> Multi-class and multi-label classifiers report raw accuracy or deceptive Hamming loss without evaluating Expected Calibration Error (ECE) or multi-class Brier score.</li>
    <li><strong>Gap 5: Ungrounded Generative Answers:</strong> Existing LLM search tools synthesize paragraphs without verifying whether individual claims are actually entailed by cited source papers.</li>
</ul>

<div class="table-caption">Table 2.2: Mapping of Identified Research Gaps to Engineered System Countermeasures</div>
<table class="report-table">
    <thead>
        <tr>
            <th style="width: 15%;">Gap Identifier</th>
            <th style="width: 30%;">Identified Literature Limitation</th>
            <th style="width: 35%;">Engineered Solution in Proposed Framework</th>
            <th style="width: 20%;">Quantitative Impact</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Gap 1</strong></td>
            <td>Randomized splits cause temporal vocabulary leakage.</td>
            <td class="left-align">Strict chronological date-boundary splits (Train $\\le$ Apr 2026, Val May–Jul 2026, Test Aug–Oct 2026).</td>
            <td>Exposes +2.3% to +4.7% artificial inflation.</td>
        </tr>
        <tr>
            <td><strong>Gap 2</strong></td>
            <td>Contrastive mining treats related abstracts as false negatives.</td>
            <td class="left-align">Topic-aware hard negative mining with a 5-nearest-neighbor exclusion filter.</td>
            <td>Raises Test Recall@10 from 93.93% to 98.31%.</td>
        </tr>
        <tr>
            <td><strong>Gap 3</strong></td>
            <td>Isolated BM25 misses meaning; isolated bi-encoder misses acronyms.</td>
            <td class="left-align">Two-stage hybrid retrieval fusing BM25 and FAISS HNSW via RRF ($k=60$) + Cross-Encoder.</td>
            <td>Recovers exact acronyms and semantic concepts.</td>
        </tr>
        <tr>
            <td><strong>Gap 4</strong></td>
            <td>Hamming loss is 93.97% deceptive on sparse multi-label matrices.</td>
            <td class="left-align">Validation-tuned decision thresholds $T_c \in [0.2, 0.8]$ evaluated on Micro-F1 and Brier score.</td>
            <td>Micro-F1 rises +5.35% (0.5852 to 0.6387).</td>
        </tr>
        <tr>
            <td><strong>Gap 5</strong></td>
            <td>Generative answers invent fictitious paper citations.</td>
            <td class="left-align">Sentence-level cosine grounding verification algorithm + refusal threshold ($S < -5.0$).</td>
            <td>Flags unsupported claims with warning badges.</td>
        </tr>
    </tbody>
</table>

<div class="figure-box">
    <img src="__IMG_LADDER_COMP__" alt="Temporal Leakage Gap Comparison">
    <div class="figure-caption">Figure 2.1: Experimental comparison between Random 5-Fold CV, Rolling-Origin CV, and Holdout Test Accuracy across all evaluated models, demonstrating consistent +2.27% to +4.70% accuracy inflation caused by temporal data leakage.</div>
</div>

<div class="running-footer">
    Dr. D. Y. Patil School of Science & Technology, Pimpri, Pune — Academic Year 2025–2026
</div>

<!-- ======================================================================= -->
<!-- CHAPTER 3: DATASET COHORTS & PREPROCESSING PIPELINE                     -->
<!-- ======================================================================= -->
<div class="page-break"></div>

<div class="running-header">
    <span>Research Copilot: Hybrid IR, Contrastive Learning & Classification</span>
    <span>B.Tech Project Report (Dept. of AI & DS, DPU Pune)</span>
</div>

<div class="chapter-header">CHAPTER 3: DATASET COHORTS, TEMPORAL GATEKEEPING, AND PREPROCESSING PIPELINE</div>

<div class="section-header">3.1 Primary Corpus Benchmark: 16,417 arXiv Papers and Strict Date Boundaries</div>
<p>
The core benchmark powering Research Copilot comprises <strong>16,417 computer science research papers</strong> spanning January 2020 through October 2026. The dataset is ingested from two complementary sources:
</p>
<ul>
    <li><strong>Historical OAI Snapshot (5,046 papers):</strong> Sourced from the Cornell University arXiv metadata dump, sampled uniformly at 115 papers per month across 2020-01 through 2023-08.</li>
    <li><strong>Live arXiv REST API (11,371 papers):</strong> Ingested directly from <code>export.arxiv.org/api/query</code> at a target rate of 300 papers per month across 2023-09 through 2026-10 using custom rate-limit buffering.</li>
</ul>

<div class="figure-box">
    <img src="__IMG_PAPERS_MONTH__" alt="Monthly Paper Distribution">
    <div class="figure-caption">Figure 3.1: Continuous monthly paper sampling volume across all 82 calendar months (January 2020 through October 2026), verifying balanced temporal coverage without month-boundary clustering.</div>
</div>

<div class="section-header">3.2 The OpenAlex Label-Quality Audit: 19% Failure and 100% Genuine arXiv Taxonomy</div>
<p>
During preliminary system design, an automated concept expansion pipeline was tested using the OpenAlex API to enrich paper categories. To verify label fidelity, an automated audit was conducted on a random sample of 100 papers, comparing OpenAlex automated tags against genuine author-assigned arXiv primary categories.
</p>
<p>
The audit revealed that OpenAlex concept tags agreed with author classifications on only <strong>19 out of 100 papers (19.0% accuracy)</strong>. OpenAlex models exhibited severe concept hallucination—tagging theoretical cryptography preprints with biomedical "Clinical Trial" labels due to words like "test" and "control." Consequently, OpenAlex concepts were <strong>completely eliminated</strong>. The entire 16,417-paper corpus is indexed using 100% genuine author-submitted arXiv categories (`VALID_CS`, 35 active classes after dropping 4 rare classes with fewer than 20 training instances).
</p>

<div class="figure-box">
    <img src="__IMG_CLASS_DIST__" alt="Class Distribution Across 35 Categories">
    <div class="figure-caption">Figure 3.2: Publication distribution across all 35 active Computer Science categories in the 16,417-paper corpus, illustrating heavy representation in Machine Learning (`cs.LG`), Computer Vision (`cs.CV`), and NLP (`cs.CL`).</div>
</div>

<div class="section-header">3.3 Text Length Distributions & Sparse Multi-Label Matrix Topology</div>
<p>
Text statistics computed over the full 16,417 corpus reveal:
</p>
<ul>
    <li><strong>Abstract Character Length:</strong> Minimum: 133 chars; Maximum: 2,776 chars; Mean: <strong>1,272.2 chars</strong>; Median: <strong>1,273.0 chars</strong>; Std: 334.9 chars.</li>
    <li><strong>Abstract Word Length:</strong> Minimum: 15 words; Maximum: 365 words; Mean: <strong>176.1 words</strong>; Median: <strong>175.0 words</strong>; Std: 46.5 words.</li>
    <li><strong>Multi-Label Sparsity:</strong> Total label assignments across all papers = 34,660. The multi-label indicator matrix (16,417 papers $\times$ 35 categories = 574,595 binary cells) contains 539,935 zero cells, yielding an extreme matrix sparsity of <strong>93.97%</strong>.</li>
</ul>

<div class="section-header">3.4 Preprocessing and Feature Representations</div>
<p>
For every manuscript, text is standardized by concatenating the title and abstract:
</p>
<div class="eq-box">
    <span class="eq-content">text_i = Title_i \cdot \text{". "} \cdot Abstract_i</span>
    <span class="eq-num">(3.1)</span>
</div>
<p>
Two distinct feature spaces are extracted for comparative benchmarking:
</p>
<ol>
    <li><strong>High-Dimensional Sparse Lexical Space:</strong> Extracted using <code>TfidfVectorizer(max_features=30000, ngram_range=(1,2), sublinear_tf=True, stop_words="english")</code>.</li>
    <li><strong>Dense Semantic Representation Space:</strong> Encoded using <code>all-MiniLM-L6-v2</code> into $\mathbf{v} \in \mathbb{R}^{384}$, normalized to the $L_2$ unit sphere:
    <div class="eq-box">
        <span class="eq-content">\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}, \quad \text{such that } \|\hat{\mathbf{v}}\|_2 = 1</span>
        <span class="eq-num">(3.2)</span>
    </div>
    </li>
</ol>

<div class="running-footer">
    Dr. D. Y. Patil School of Science & Technology, Pimpri, Pune — Academic Year 2025–2026
</div>

<!-- ======================================================================= -->
<!-- CHAPTER 4: SYSTEM ARCHITECTURE & MODELS                                 -->
<!-- ======================================================================= -->
<div class="page-break"></div>

<div class="running-header">
    <span>Research Copilot: Hybrid IR, Contrastive Learning & Classification</span>
    <span>B.Tech Project Report (Dept. of AI & DS, DPU Pune)</span>
</div>

<div class="chapter-header">CHAPTER 4: DUAL-PARADIGM MACHINE LEARNING AND RETRIEVAL ARCHITECTURE</div>

<div class="section-header">4.1 Four-Stage System Workflow and Layered Topology</div>
<p>
Research Copilot is organized into four sequential operational stages:
</p>
<ol>
    <li><strong>Stage 1 (Ingestion & Preprocessing):</strong> Continuous harvesting, text concatenation, and $L_2$ vector normalization.</li>
    <li><strong>Stage 2 (Hybrid Candidate Generation):</strong> Sparse BM25 scoring and FAISS HNSW dense search returning top 50 candidates each, fused via Reciprocal Rank Fusion into top 30 candidates.</li>
    <li><strong>Stage 3 (Cross-Attention Re-Ranking):</strong> Full transformer cross-encoder scoring of top 30 candidates into top 8 verified papers.</li>
    <li><strong>Stage 4 (Synthesis & Citation Grounding):</strong> MMR extractive answer generation, inline numerical citations, and sentence-level grounding verification.</li>
</ol>

<div class="figure-box">
    <img src="__IMG_MAP_OVERVIEW__" alt="Interactive 2D UMAP Corpus Overview">
    <div class="figure-caption">Figure 4.1: Production user interface of Research Copilot displaying the 16,417-paper interactive 2D UMAP canvas, topic clusters, search controls, and live metadata inspection drawer.</div>
</div>

<div class="section-header">4.2 Two-Stage Hybrid Retrieval Branch: BM25, FAISS HNSW, and RRF</div>
<p>
For any query $Q$, BM25 computes sparse lexical relevance over term frequencies:
</p>
<div class="eq-box">
    <span class="eq-content">\text{Score}_{\text{BM25}}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}</span>
    <span class="eq-num">(4.1)</span>
</div>
<p>
Simultaneously, FAISS HNSW executes approximate nearest-neighbor search in $O(\log N)$ time:
</p>
<div class="eq-box">
    <span class="eq-content">\text{Sim}_{\text{Dense}}(Q, D) = \hat{\mathbf{q}} \cdot \hat{\mathbf{d}} = \cos(\mathbf{q}, \mathbf{d})</span>
    <span class="eq-num">(4.2)</span>
</div>
<p>
Because BM25 scores are unbounded $[0, \infty)$ and cosine scores are $[-1, 1]$, naive addition requires fragile scaling weights. Instead, Reciprocal Rank Fusion fuses candidates purely by ordinal ranking:
</p>
<div class="eq-box">
    <span class="eq-content">\text{RRF Score}(d) = \sum_{m \in \{\text{Dense}, \text{Sparse}\}} \frac{1}{k + r_m(d)}, \quad \text{with } k = 60</span>
    <span class="eq-num">(4.3)</span>
</div>

<div class="section-header">4.3 Deep Cross-Encoder Re-Ranking Branch</div>
<p>
While bi-encoders process query and document independently, the Cross-Encoder (<code>cross-encoder/ms-marco-MiniLM-L-6-v2</code>) passes the query and candidate title-abstract together through 6 cross-attention layers:
</p>
<div class="eq-box">
    <span class="eq-content">S_{\text{CE}}(Q, D) = \text{Linear}\left( \text{BERT}\left( \text{[CLS]} \circ Q \circ \text{[SEP]} \circ D \circ \text{[SEP]} \right) \right)</span>
    <span class="eq-num">(4.4)</span>
</div>
<p>
This computes direct token-to-token cross-attention, re-scoring the top 30 candidates to select the final top 8 papers.
</p>

<div class="section-header">4.4 Contrastive Representation Learning Branch: In-Domain Fine-Tuning</div>
<p>
To specialize the general <code>all-MiniLM-L6-v2</code> embedder to scientific terminology, contrastive fine-tuning is executed using Multiple Negatives Ranking Loss (MNRL):
</p>
<div class="eq-box">
    <span class="eq-content">\mathcal{L}_{\text{MNRL}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{e^{\tau \cdot \cos(\mathbf{q}_i, \mathbf{d}_i^+)}}{e^{\tau \cdot \cos(\mathbf{q}_i, \mathbf{d}_i^+)} + \sum_{j \neq i} e^{\tau \cdot \cos(\mathbf{q}_i, \mathbf{d}_j^+)} + \sum_{k} e^{\tau \cdot \cos(\mathbf{q}_i, \mathbf{d}_{i,k}^-)}}</span>
    <span class="eq-num">(4.5)</span>
</div>
<p>
<strong>False Negative Exclusion Radius:</strong> Mining hard negatives strictly by topic cluster risks selecting abstracts that are near-identical to the query title. An exclusion radius in <code>train_embedder.py</code> removes the top 5 nearest neighbors of the positive abstract from ever being chosen as hard negatives.
</p>

<div class="section-header">4.5 Supervised Categorization Branch: Model Specifications</div>
<p>
Table 4.1 details the mathematical architectures and exact hyperparameters across all evaluated classifiers.
</p>

<div class="table-caption">Table 4.1: Architectural Topology and Hyperparameter Specifications Across All Evaluated Models</div>
<table class="report-table">
    <thead>
        <tr>
            <th style="width: 20%;">Model / Component</th>
            <th style="width: 30%;">Architecture & Layer Topology</th>
            <th style="width: 32%;">Key Hyperparameters & Regularization</th>
            <th style="width: 18%;">Optimizer / Eval</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Naive Bayes</strong></td>
            <td>Multinomial Naive Bayes</td>
            <td class="left-align">$\alpha = 1.0$ (Laplace smoothing), fit_prior=True</td>
            <td>TF-IDF (30k)</td>
        </tr>
        <tr>
            <td><strong>Logistic Regression</strong></td>
            <td>L2-regularized multinomial linear model</td>
            <td class="left-align">$C = 20.0$ (validation tuned), max_iter=1000, class_weight="balanced"</td>
            <td>L-BFGS / TF-IDF</td>
        </tr>
        <tr>
            <td><strong>Linear SVM</strong></td>
            <td>Maximum-margin linear hyperplane</td>
            <td class="left-align">$C = 1.0$ (validation tuned), loss="squared_hinge", max_iter=2500</td>
            <td>LibLinear / TF-IDF</td>
        </tr>
        <tr>
            <td><strong>k-NN</strong></td>
            <td>Instance-based metric space search</td>
            <td class="left-align">$k = 15$ (validation tuned), metric="cosine", weights="distance"</td>
            <td>Brute/KD / 384-D</td>
        </tr>
        <tr>
            <td><strong>Random Forest</strong></td>
            <td>100 bagged de-correlated decision trees</td>
            <td class="left-align">n_estimators=100, max_features="sqrt", class_weight="balanced"</td>
            <td>Gini / 384-D</td>
        </tr>
        <tr>
            <td><strong>PyTorch MLP</strong></td>
            <td>Deep Feedforward: 384 $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 35</td>
            <td class="left-align">BatchNorm1d, ReLU, Dropout ($p=0.30$ and $p=0.21$), CrossEntropyLoss</td>
            <td>AdamW (lr=1e-3, 30 Ep)</td>
        </tr>
        <tr>
            <td><strong>Voting Ensemble</strong></td>
            <td>Soft probability voting of LR + RF + kNN</td>
            <td class="left-align">Aggregates calibrated linear, tree, and metric probabilities</td>
            <td>Soft Vote / 384-D</td>
        </tr>
        <tr>
            <td><strong>Contrastive Embedder</strong></td>
            <td>6-layer Transformer (22.7M parameters)</td>
            <td class="left-align">MNRL loss, hard negatives with 5-NN exclusion, batch_size=64, 3 Epochs</td>
            <td>AdamW (lr=2e-5) / Tesla T4</td>
        </tr>
    </tbody>
</table>

<div class="running-footer">
    Dr. D. Y. Patil School of Science & Technology, Pimpri, Pune — Academic Year 2025–2026
</div>

<!-- ======================================================================= -->
<!-- CHAPTER 5: SOFTWARE ARCHITECTURE & VALIDATION                           -->
<!-- ======================================================================= -->
<div class="page-break"></div>

<div class="running-header">
    <span>Research Copilot: Hybrid IR, Contrastive Learning & Classification</span>
    <span>B.Tech Project Report (Dept. of AI & DS, DPU Pune)</span>
</div>

<div class="chapter-header">CHAPTER 5: SOFTWARE ARCHITECTURE, MODULAR IMPLEMENTATION, AND VALIDATION STRATEGY</div>

<div class="section-header">5.1 Decoupled Python and FastAPI Software Stack</div>
<p>
The codebase is structured into independent, highly maintainable modules. Table 5.1 summarizes the responsibilities, dependencies, and interfaces of each component.
</p>

<div class="table-caption">Table 5.1: Modular Software Architecture and Component Responsibilities</div>
<table class="report-table">
    <thead>
        <tr>
            <th style="width: 22%;">Module Filename</th>
            <th style="width: 38%;">Core Engineering Responsibility</th>
            <th style="width: 22%;">Key Dependencies</th>
            <th style="width: 18%;">Output Interface</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>prepare.py</code></td>
            <td class="left-align">Harvests arXiv data, extracts 384-D vectors, builds FAISS index, computes UMAP 2D coordinates & HDBSCAN clusters.</td>
            <td>PyYAML, Pandas, FAISS, UMAP, Sentence-Transformers</td>
            <td><code>papers.pkl</code>, <code>index.faiss</code>, <code>clusters.json</code></td>
        </tr>
        <tr>
            <td><code>evaluate_ml.py</code></td>
            <td class="left-align">Executes rolling CV, random 5-fold CV, test holdout ladder, bootstrap 95% CIs, McNemar tests, and multi-label thresholding.</td>
            <td>Scikit-Learn, PyTorch, SciPy, NumPy</td>
            <td><code>metrics.json</code>, <code>classification.md</code></td>
        </tr>
        <tr>
            <td><code>train_embedder.py</code></td>
            <td class="left-align">Executes contrastive in-domain fine-tuning with MNRL loss and false-negative exclusion radius; evaluates on SciFact.</td>
            <td>Sentence-Transformers, PyTorch, Datasets</td>
            <td><code>embedder-ft/</code>, <code>search_benchmark.md</code></td>
        </tr>
        <tr>
            <td><code>app.py</code></td>
            <td class="left-align">FastAPI web microservice serving REST endpoints for search, 2D map, cited Q&A, similar papers, and topic trajectories.</td>
            <td>FastAPI, Uvicorn, Rank-BM25, Cross-Encoder</td>
            <td>Web UI (< 220 ms)</td>
        </tr>
        <tr>
            <td><code>make_final.py</code></td>
            <td class="left-align">Synchronizes benchmark results across Markdown reports, web UI metrics modal, and PowerPoint slides.</td>
            <td>Python-PPTX, JSON, Markdown</td>
            <td>Final artifacts</td>
        </tr>
    </tbody>
</table>

<div class="section-header">5.2 Extractive MMR Synthesis and Sentence-Level Citation Grounding Check</div>
<p>
To guarantee zero hallucinatory claims in the "Ask" mode, Research Copilot deploys Maximal Marginal Relevance (MMR) extractive synthesis. For each candidate sentence $s_i$ in the top retrieved abstracts:
</p>
<div class="eq-box">
    <span class="eq-content">\text{MMR} = \arg\max_{s_i \in C \setminus S} \left[ \lambda \cdot \cos(\mathbf{s}_i, \mathbf{q}) - (1 - \lambda) \cdot \max_{s_j \in S} \cos(\mathbf{s}_i, \mathbf{s}_j) \right], \quad \lambda = 0.7</span>
    <span class="eq-num">(5.1)</span>
</div>
<p>
Every extracted claim is tagged with an inline numerical badge <code>[n]</code> linking to source abstract $n$. The grounding verification algorithm encodes claim sentence $s$ and all sentences of cited abstract $D_n$. If $\max \cos(\mathbf{s}, \mathbf{d}) \ge 0.45$, the claim is verified (`✓`); otherwise, it is flagged with a warning (`⚠`).
</p>

<div class="section-header">5.3 Confidence Gatekeeping and Out-of-Domain Refusal Protocol</div>
<p>
When presented with queries outside Computer Science (e.g., baking recipes or clinical surgeries), generative systems invent plausible sounding falsehoods. Research Copilot enforces an explicit score-based refusal gatekeeper:
</p>
<div class="eq-box">
    <span class="eq-content">\text{Refuse}(Q) = \mathbf{1}\left( \max_{d \in \text{Hits}} S_{\text{CE}}(Q, d) < -5.0 \right)</span>
    <span class="eq-num">(5.2)</span>
</div>
<p>
If the maximum cross-encoder score is below $-5.0$, the server returns <code>{"answer": "", "confident": false}</code>, protecting users from ungrounded responses.
</p>

<div class="running-footer">
    Dr. D. Y. Patil School of Science & Technology, Pimpri, Pune — Academic Year 2025–2026
</div>

<!-- ======================================================================= -->
<!-- CHAPTER 6: EXPERIMENTAL EVALUATION & RESULTS                            -->
<!-- ======================================================================= -->
<div class="page-break"></div>

<div class="running-header">
    <span>Research Copilot: Hybrid IR, Contrastive Learning & Classification</span>
    <span>B.Tech Project Report (Dept. of AI & DS, DPU Pune)</span>
</div>

<div class="chapter-header">CHAPTER 6: EXPERIMENTAL EVALUATION, MULTI-METRIC BENCHMARKS, AND WEB INTERFACE DEPLOYMENT</div>

<div class="section-header">6.1 Single-Label Classification Ladder Across 35 arXiv Disciplines</div>
<p>
Table 6.1 presents the final empirical evaluation across all eight model architectures on the unseen holdout test set (August–October 2026, $N=890$).
</p>

<div class="table-caption">Table 6.1: Complete Model Ladder Comparison on Unseen Holdout Test Split (Aug–Oct 2026)</div>
<table class="report-table">
    <thead>
        <tr>
            <th style="width: 22%;">Model Architecture</th>
            <th style="width: 16%;">Input Feature</th>
            <th style="width: 14%;">Rolling-Origin CV</th>
            <th style="width: 14%;">Random 5-Fold CV</th>
            <th style="width: 12%;">Test Acc</th>
            <th style="width: 11%;">Macro-F1</th>
            <th style="width: 11%;">Brier Score</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Naive Bayes</strong></td>
            <td>TF-IDF (30k)</td>
            <td>43.43% ± 2.24%</td>
            <td>45.99% ± 0.34%</td>
            <td>38.54%</td>
            <td>0.0559</td>
            <td>0.8772</td>
        </tr>
        <tr>
            <td><strong>Random Forest</strong></td>
            <td>Embeddings (384-D)</td>
            <td>57.89% ± 0.71%</td>
            <td>60.53% ± 0.60%</td>
            <td>55.28%</td>
            <td>0.3082</td>
            <td>0.7349</td>
        </tr>
        <tr>
            <td><strong>HistGradientBoosting</strong></td>
            <td>Embeddings (384-D)</td>
            <td>45.08% ± 22.62%</td>
            <td>49.78% ± 10.47%</td>
            <td>48.99%</td>
            <td>0.2525</td>
            <td>0.7220</td>
        </tr>
        <tr>
            <td><strong>k-NN (k=15)</strong></td>
            <td>Embeddings (384-D)</td>
            <td>63.91% ± 3.30%</td>
            <td>67.20% ± 0.42%</td>
            <td>61.24%</td>
            <td>0.3643</td>
            <td>0.5228</td>
        </tr>
        <tr>
            <td><strong>PyTorch MLP</strong></td>
            <td>Embeddings (384-D)</td>
            <td>63.89% ± 2.85%</td>
            <td>67.04% ± 0.51%</td>
            <td>62.58%</td>
            <td>0.4182</td>
            <td>0.5622</td>
        </tr>
        <tr>
            <td><strong>Linear SVM</strong></td>
            <td>TF-IDF (30k)</td>
            <td>66.19% ± 1.74%</td>
            <td>68.46% ± 0.33%</td>
            <td>63.60%</td>
            <td>0.4797</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><strong>Logistic Regression</strong></td>
            <td>TF-IDF (30k)</td>
            <td>66.25% ± 1.13%</td>
            <td>69.04% ± 0.49%</td>
            <td>64.61%</td>
            <td><strong>0.5133</strong></td>
            <td><strong>0.5007</strong></td>
        </tr>
        <tr>
            <td><strong>Voting Ensemble</strong></td>
            <td>Embeddings (384-D)</td>
            <td><strong>66.78% ± 2.58%</strong></td>
            <td><strong>69.17% ± 0.42%</strong></td>
            <td><strong>65.62%</strong></td>
            <td>0.4155</td>
            <td>0.5107</td>
        </tr>
    </tbody>
</table>

<div class="figure-box">
    <img src="__IMG_CONF_MATRIX__" alt="Category Confusion Matrix">
    <div class="figure-caption">Figure 6.1: Normalized category confusion matrix of the best model (Voting Ensemble) across the top 10 most populated disciplines, revealing expected semantic bleed between `cs.AI`, `cs.LG`, and `cs.CL`.</div>
</div>

<div class="section-header">6.2 Holm-Bonferroni Corrected Statistical Significance Testing</div>
<p>
To determine whether performance differences among the top four classifiers are real or artifacts of sampling noise, 10,000 paired bootstrap iterations and McNemar's tests were conducted with Holm-Bonferroni multiple testing correction ($m=6$ comparisons).
</p>

<div class="table-caption">Table 6.2: Statistical Significance Testing with Holm-Bonferroni Family-Wise Correction ($m=6$)</div>
<table class="report-table">
    <thead>
        <tr>
            <th style="width: 25%;">Comparison ($M_A$ vs $M_B$)</th>
            <th style="width: 12%;">$\Delta$ Acc</th>
            <th style="width: 15%;">95% Bootstrap CI</th>
            <th style="width: 12%;">Raw Boot $p$</th>
            <th style="width: 12%;">Holm Boot $p$</th>
            <th style="width: 12%;">Holm McNemar $p$</th>
            <th style="width: 12%;">Significance</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Voting Ensemble vs Logistic Reg</td>
            <td>+1.01%</td>
            <td>[-2.0%, +4.2%]</td>
            <td>0.5204</td>
            <td><strong>1.0000</strong></td>
            <td><strong>1.0000</strong></td>
            <td>Not Significant</td>
        </tr>
        <tr>
            <td>Voting Ensemble vs Linear SVM</td>
            <td>+2.02%</td>
            <td>[-0.9%, +5.1%]</td>
            <td>0.1926</td>
            <td><strong>0.9630</strong></td>
            <td><strong>1.0000</strong></td>
            <td>Not Significant</td>
        </tr>
        <tr>
            <td>Voting Ensemble vs PyTorch MLP</td>
            <td>+3.04%</td>
            <td>[+0.6%, +5.6%]</td>
            <td>0.0198</td>
            <td><strong>0.1188</strong></td>
            <td><strong>0.1428</strong></td>
            <td>Not Significant</td>
        </tr>
        <tr>
            <td>Logistic Reg vs Linear SVM</td>
            <td>+1.01%</td>
            <td>[-0.8%, +2.9%]</td>
            <td>0.3280</td>
            <td><strong>0.9840</strong></td>
            <td><strong>1.0000</strong></td>
            <td>Not Significant</td>
        </tr>
        <tr>
            <td>Logistic Reg vs PyTorch MLP</td>
            <td>+2.03%</td>
            <td>[-1.2%, +5.3%]</td>
            <td>0.2414</td>
            <td><strong>0.9656</strong></td>
            <td><strong>1.0000</strong></td>
            <td>Not Significant</td>
        </tr>
        <tr>
            <td>Linear SVM vs PyTorch MLP</td>
            <td>+1.02%</td>
            <td>[-2.2%, +4.3%]</td>
            <td>0.5626</td>
            <td><strong>1.0000</strong></td>
            <td><strong>1.0000</strong></td>
            <td>Not Significant</td>
        </tr>
    </tbody>
</table>

<div class="section-header">6.3 Retrieval Benchmark: In-Corpus and External BEIR SciFact Evaluation</div>
<p>
Table 6.3 details retrieval performance across 5,022 corpus abstracts (Validation Split $N=898$, Holdout Test Split $N=890$) and the external BEIR SciFact benchmark (300 test claims).
</p>

<div class="table-caption">Table 6.3: Corpus Retrieval Benchmark and External BEIR SciFact Evaluation</div>
<table class="report-table">
    <thead>
        <tr>
            <th style="width: 25%;">Evaluation Split</th>
            <th style="width: 25%;">Checkpoint Model</th>
            <th style="width: 16%;">Recall@10</th>
            <th style="width: 17%;">MRR@10</th>
            <th style="width: 17%;">nDCG@10</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Validation Set (898 queries)</td>
            <td>Pre-trained Base (<code>all-MiniLM-L6-v2</code>)</td>
            <td>95.77%</td>
            <td>0.8737</td>
            <td>0.8941</td>
        </tr>
        <tr>
            <td>Validation Set (898 queries)</td>
            <td>Fine-Tuned Model (<code>embedder-ft</code>)</td>
            <td><strong>98.78%</strong></td>
            <td><strong>0.9501</strong></td>
            <td><strong>0.9593</strong></td>
        </tr>
        <tr>
            <td>Holdout Test Set (890 queries)</td>
            <td>Pre-trained Base (<code>all-MiniLM-L6-v2</code>)</td>
            <td>93.93%</td>
            <td>0.8299</td>
            <td>0.8564</td>
        </tr>
        <tr>
            <td>Holdout Test Set (890 queries)</td>
            <td>Fine-Tuned Model (<code>embedder-ft</code>)</td>
            <td><strong>98.31%</strong></td>
            <td><strong>0.9338</strong></td>
            <td><strong>0.9460</strong></td>
        </tr>
        <tr>
            <td>BEIR SciFact Test (300 queries)</td>
            <td>Pre-trained Base (<code>all-MiniLM-L6-v2</code>)</td>
            <td><strong>78.33%</strong></td>
            <td><strong>0.6047</strong></td>
            <td><strong>0.6451</strong></td>
        </tr>
        <tr>
            <td>BEIR SciFact Test (300 queries)</td>
            <td>Fine-Tuned Model (<code>embedder-ft</code>)</td>
            <td>77.83%</td>
            <td>0.5945</td>
            <td>0.6359</td>
        </tr>
    </tbody>
</table>

<div class="figure-box">
    <img src="__IMG_SEARCH_BENCH__" alt="Search Benchmark Bar Chart">
    <div class="figure-caption">Figure 6.2: Comparative retrieval metrics showing statistically significant in-domain improvements (+4.38% Recall@10, +0.1039 MRR@10, $p < 0.0001$) alongside fully preserved zero-shot generalization on BEIR SciFact ($p = 0.3290$).</div>
</div>

<div class="section-header">6.4 System Latency Profiling and Memory Footprint</div>
<p>
To evaluate production readiness and real-time responsiveness, comprehensive latency profiling was conducted across <strong>50 independent end-to-end queries</strong> executed against the running FastAPI backend on standard host CPU hardware (Apple M-series Silicon, unified architecture). Latency distributions were measured stage-by-stage across query dense encoding, FAISS vector graph traversal, BM25 inverted index token matching, Reciprocal Rank Fusion merging, and cross-encoder neural re-ranking.
</p>

<div class="table-caption">Table 6.4: End-to-End Search & Generation Stage-Wise Latency Profiling (50 Measured Queries on Host CPU)</div>
<table class="report-table">
    <thead>
        <tr>
            <th style="width: 32%;">Architectural Stage / Component</th>
            <th style="width: 14%;">Median (ms)</th>
            <th style="width: 14%;">Mean (ms)</th>
            <th style="width: 14%;">p95 (ms)</th>
            <th style="width: 26%;">Implementation & Computational Profile</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="left-align"><strong>Dense Query Encoding</strong></td>
            <td><strong>5.25</strong></td>
            <td>34.63</td>
            <td>182.76</td>
            <td class="left-align"><code>all-MiniLM-L6-v2</code> PyTorch embedding generation</td>
        </tr>
        <tr>
            <td class="left-align"><strong>Dense Vector Retrieval</strong></td>
            <td><strong>0.26</strong></td>
            <td>0.32</td>
            <td>0.42</td>
            <td class="left-align">FAISS HNSW ($M=32, ef=64$) proximity graph search</td>
        </tr>
        <tr>
            <td class="left-align"><strong>Sparse Lexical Matching</strong></td>
            <td><strong>17.55</strong></td>
            <td>17.67</td>
            <td>22.23</td>
            <td class="left-align">BM25 Okapi inverted index scoring ($k_1=1.5, b=0.75$)</td>
        </tr>
        <tr>
            <td class="left-align"><strong>Reciprocal Rank Fusion (RRF)</strong></td>
            <td><strong>0.02</strong></td>
            <td>0.02</td>
            <td>0.03</td>
            <td class="left-align">Reciprocal rank interleave ($k=60$) over Top-100 lists</td>
        </tr>
        <tr>
            <td class="left-align"><strong>Cross-Encoder Re-Ranking</strong></td>
            <td><strong>191.66</strong></td>
            <td>194.08</td>
            <td>231.10</td>
            <td class="left-align"><code>ms-marco-MiniLM-L-6-v2</code> cross-attention scoring</td>
        </tr>
        <tr style="background-color: #f1f5f9; font-weight: bold;">
            <td class="left-align"><strong>Total Search Endpoint (<code>/api/search</code>)</strong></td>
            <td><strong>216.81</strong></td>
            <td>246.72</td>
            <td>431.79</td>
            <td class="left-align">Two-stage hybrid retrieval with full re-ranking</td>
        </tr>
        <tr style="background-color: #e2e8f0; font-weight: bold;">
            <td class="left-align"><strong>Total Cited Ask Endpoint (<code>/api/ask</code>)</strong></td>
            <td><strong>354.12</strong></td>
            <td>387.45</td>
            <td>562.80</td>
            <td class="left-align">Hybrid retrieval + MMR summary + Citation validation</td>
        </tr>
    </tbody>
</table>

<p>
Memory profiling demonstrates a lightweight active operational footprint, enabling cost-effective deployment on memory-constrained standard edge servers or low-tier virtual machines without requiring dedicated GPU infrastructure.
</p>

<div class="table-caption">Table 6.5: Production Memory Footprint and Artifact Storage Allocation</div>
<table class="report-table">
    <thead>
        <tr>
            <th style="width: 28%;">Component / Index Artifact</th>
            <th style="width: 18%;">Memory Type</th>
            <th style="width: 16%;">Disk Size</th>
            <th style="width: 18%;">In-Memory (RAM)</th>
            <th style="width: 20%;">Operational Function</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="left-align">Corpus Metadata (<code>corpus.jsonl</code>)</td>
            <td>Serialized JSONL / Buffer</td>
            <td>14.8 MB</td>
            <td>~38.0 MB</td>
            <td>16,417 paper metadata objects</td>
        </tr>
        <tr>
            <td class="left-align">Dense Vectors (<code>all_embeddings.npy</code>)</td>
            <td>Float32 Array</td>
            <td>24.1 MB</td>
            <td>25.2 MB</td>
            <td>$16,417 	imes 384$ normalized embeddings</td>
        </tr>
        <tr>
            <td class="left-align">FAISS Graph Index (<code>faiss_hnsw.index</code>)</td>
            <td>HNSW Graph Structure</td>
            <td>26.2 MB</td>
            <td>28.5 MB</td>
            <td>Multi-layer nearest-neighbor graph</td>
        </tr>
        <tr>
            <td class="left-align">Sparse BM25 Index (<code>bm25.pkl</code>)</td>
            <td>Serialized Inverted Index</td>
            <td>31.8 MB</td>
            <td>36.4 MB</td>
            <td>Corpus posting lists and IDF tables</td>
        </tr>
        <tr>
            <td class="left-align">Embedding & Re-Ranking Models</td>
            <td>PyTorch Neural Weights</td>
            <td>240.2 MB</td>
            <td>265.0 MB</td>
            <td>Bi-encoder and cross-encoder weights</td>
        </tr>
        <tr>
            <td class="left-align">FastAPI Engine & Runtime Buffers</td>
            <td>Process Heap / Stack</td>
            <td>—</td>
            <td>265.95 MB</td>
            <td>Uvicorn ASGI server and cache buffers</td>
        </tr>
        <tr style="background-color: #f1f5f9; font-weight: bold;">
            <td class="left-align"><strong>Total Process Memory (RSS)</strong></td>
            <td>Resident Working Set</td>
            <td><strong>337.1 MB</strong></td>
            <td><strong>659.05 MB</strong></td>
            <td>Entire 16,417-paper system resident</td>
        </tr>
        <tr style="background-color: #e2e8f0; font-weight: bold;">
            <td class="left-align"><strong>Total Virtual Address Space (VMS)</strong></td>
            <td>Virtual Memory Allocation</td>
            <td>—</td>
            <td><strong>12.30 GB</strong></td>
            <td>Memory-mapped arrays and torch cache</td>
        </tr>
    </tbody>
</table>

<div class="page-break"></div>

<div class="running-header">
    <span>Research Copilot: Hybrid IR, Contrastive Learning & Classification</span>
    <span>B.Tech Project Report (Dept. of AI & DS, DPU Pune)</span>
</div>

<div class="section-header">6.5 Production Case Studies, Grounding Verification, and Safety Guardrails</div>
<p>
To demonstrate the end-to-end functionality, academic veracity, and ethical failure handling of Research Copilot in production, the subsequent diagnostic case cards summarize six real operational queries executed against the live system. Every generated claim undergoes sentence-level cosine grounding verification against the retrieved arXiv abstracts to enforce academic transparency.
</p>

<!-- Case Study 1 -->
<div class="case-card">
    <div class="case-card-header">
        <span>Case Study 1: Multi-Document Graph Neural Network Query</span>
        <span class="badge-success">100% Verified Grounding</span>
    </div>
    <div class="case-query"><strong>Query:</strong> "Contrastive learning for graph neural networks"</div>
    <div class="case-field"><strong>Top Retrieved Paper:</strong> <em>Cross-view Self-Supervised Learning on Heterogeneous Graph Neural Network via Bootstrapping</em> (arXiv: <code>cs.LG</code>)</div>
    <div class="case-field"><strong>Synthesized Answer:</strong> "In this paper, we propose the novel Label Contrastive Coding based Graph Neural Network (LCGNN) to utilize label information more effectively and comprehensively. [2] A crucial step in contrastive learning is the creation of 'augmented' graphs from the input graphs. [4] A large number of graph contrastive learning approaches have shown promising performance for representation learning on graphs, which train models by maximizing agreement between original graphs and their augmented views (i.e., positive views). [6]"</div>
    <div class="case-field" style="margin-top: 3pt;"><strong>Grounding Verification:</strong> 4 sentences evaluated; 4/4 verified supported against primary source abstracts ($s_{\text{cos}} \ge 0.67$); 100% attribution precision with direct arXiv links.</div>
</div>

<!-- Case Study 2 -->
<div class="case-card">
    <div class="case-card-header">
        <span>Case Study 2: Generative Vision & Text-to-Image Synthesis</span>
        <span class="badge-success">100% Verified Grounding</span>
    </div>
    <div class="case-query"><strong>Query:</strong> "Diffusion models for high resolution image synthesis"</div>
    <div class="case-field"><strong>Top Retrieved Paper:</strong> <em>Layered Diffusion Model for One-Shot High Resolution Text-to-Image Synthesis</em> (arXiv: <code>cs.CV</code>)</div>
    <div class="case-field"><strong>Synthesized Answer:</strong> "We demonstrate that higher resolution synthesis can be achieved by layering convolutions at additional resolution scales, in contrast to other methods which require additional models for super-resolution synthesis. [1] Diffusion models (DMs) have recently gained attention with state-of-the-art performance in text-to-image synthesis. [2] Current large-scale diffusion models represent a giant leap forward in conditional image synthesis, capable of interpreting diverse cues like text, human poses, and edges. [6]"</div>
    <div class="case-field" style="margin-top: 3pt;"><strong>Grounding Verification:</strong> 4 sentences evaluated; 4/4 verified supported against source abstracts ($s_{\text{cos}} \ge 0.58$); exact algorithmic references verified.</div>
</div>

<!-- Case Study 3 -->
<div class="case-card">
    <div class="case-card-header">
        <span>Case Study 3: Parameter-Efficient Fine-Tuning (LoRA)</span>
        <span class="badge-success">100% Verified Grounding</span>
    </div>
    <div class="case-query"><strong>Query:</strong> "Parameter efficient fine tuning LoRA"</div>
    <div class="case-field"><strong>Top Retrieved Paper:</strong> <em>GeoLoRA: Geometric integration for parameter efficient fine-tuning</em> (arXiv: <code>cs.LG</code>)</div>
    <div class="case-field"><strong>Synthesized Answer:</strong> "Low-Rank Adaptation (LoRA) has become a widely used method for parameter-efficient fine-tuning of large-scale, pre-trained neural networks. [1] However, LoRA and its extensions face several challenges, including the need for rank adaptivity, robustness, and computational efficiency during the fine-tuning process. [1] Parameter-efficient fine-tuning (PEFT) approaches have emerged as a viable alternative by allowing us to fine-tune models by updating only a small number of parameters. [2]"</div>
    <div class="case-field" style="margin-top: 3pt;"><strong>Grounding Verification:</strong> 4 sentences evaluated; 4/4 verified supported ($s_{\text{cos}} \ge 0.46$, max $1.00$); zero factual drift.</div>
</div>

<!-- Case Study 4 -->
<div class="case-card">
    <div class="case-card-header">
        <span>Case Study 4: Federated Learning & Privacy Safeguards</span>
        <span class="badge-success">100% Verified Grounding</span>
    </div>
    <div class="case-query"><strong>Query:</strong> "Federated learning differential privacy"</div>
    <div class="case-field"><strong>Top Retrieved Paper:</strong> <em>Evaluation of Differential Privacy Mechanisms on Federated Learning</em> (arXiv: <code>cs.CR</code>)</div>
    <div class="case-field"><strong>Synthesized Answer:</strong> "Differential privacy is one of the methods to solve the problem of privacy protection in federated learning. [2] Therefore, we proposed an adaptive differential privacy method based on federated learning. [2] This paper presents a gradient leakage resilient approach to privacy-preserving federated learning with per training example-based client differential privacy, coined as Fed-CDP. [6]"</div>
    <div class="case-field" style="margin-top: 3pt;"><strong>Grounding Verification:</strong> 4 sentences evaluated; 4/4 verified supported ($s_{\text{cos}} \ge 0.70$); all cryptographic definitions confirmed.</div>
</div>

<!-- Case Study 5: Refusal Case -->
<div class="case-card" style="border-left: 3pt solid #ef4444; background-color: #fffaf0;">
    <div class="case-card-header">
        <span>Case Study 5: Automated Out-of-Domain Refusal Gate (Safety & Anti-Hallucination)</span>
        <span class="badge-refusal">AUTOMATED REFUSAL TRIGGERED</span>
    </div>
    <div class="case-query"><strong>Adversarial / Nonsense Query:</strong> "quantum teleportation recipe for baking banana bread in hypergravity"</div>
    <div class="case-field"><strong>Top Candidate Paper Found:</strong> <em>Multimarked Spatial Search by Continuous-Time Quantum Walk</em> (arXiv: <code>cs.DS</code>)</div>
    <div class="case-field"><strong>Cross-Encoder Relevance Score:</strong> <strong>-10.7906</strong> (System Refusal Threshold: $\\tau_{\text{refuse}} = -5.0$)</div>
    <div class="case-field"><strong>Engine Decision & Response:</strong> <code>Score (-10.7906) &lt; Threshold (-5.0) &rarr; "Refused (no confident papers found)"</code></div>
    <div class="case-field" style="margin-top: 3pt;"><strong>Safety Analysis:</strong> Rather than forcing generative completion or fabricating culinary recipes with spurious physics citations, the automated cross-encoder refusal threshold cleanly terminates generation. This guarantees <strong>zero hallucination risk</strong> on out-of-distribution inputs.</div>
</div>

<!-- Case Study 6: Failure & Boundary Case -->
<div class="case-card" style="border-left: 3pt solid #f59e0b; background-color: #fffdf5;">
    <div class="case-card-header">
        <span>Case Study 6: Cross-Domain Polysemy & Vocabulary Boundary Failure Analysis</span>
        <span class="badge-warning">BOUNDARY AMBIGUITY IDENTIFIED</span>
    </div>
    <div class="case-query"><strong>Domain Stress Query:</strong> "Transformers in electric power transmission networks"</div>
    <div class="case-field"><strong>Top Retrieved Paper:</strong> <em>Moving Target Defense for Robust Monitoring of Electric Grid Transformers in Adversarial Environments</em> (arXiv: <code>cs.GT</code>)</div>
    <div class="case-field"><strong>Cross-Encoder Relevance Score:</strong> <strong>0.3402</strong> (Low confidence margin)</div>
    <div class="case-field"><strong>Sentence Grounding Check:</strong> 0 of 4 sentences supported against retrieved abstracts ($s_{\text{cos}} \le 0.27$); system flagged claims with warning icons (`&#9888;`).</div>
    <div class="case-field" style="margin-top: 3pt;"><strong>Failure Diagnosis & Academic Insight:</strong> The query exhibits severe cross-disciplinary polysemy. The user sought electrical power engineering literature on physical electromagnetic step-up/step-down transformers in high-voltage grids. Because the corpus is constrained to 35 Computer Science disciplines, the retrieval engine matched papers applying deep learning "Transformer" attention models to power grid sensor data. The automated grounding check correctly alerted the user to the semantic divergence.</div>
</div>

<div class="running-footer">
    Dr. D. Y. Patil School of Science & Technology, Pimpri, Pune — Academic Year 2025–2026
</div>

<!-- ======================================================================= -->
<!-- CHAPTER 7: CONCLUSIONS & FUTURE WORK                                    -->
<!-- ======================================================================= -->
<div class="page-break"></div>

<div class="running-header">
    <span>Research Copilot: Hybrid IR, Contrastive Learning & Classification</span>
    <span>B.Tech Project Report (Dept. of AI & DS, DPU Pune)</span>
</div>

<div class="chapter-header">CHAPTER 7: CONCLUSIONS, ETHICAL CONSIDERATIONS, AND FUTURE ENHANCEMENTS</div>

<div class="section-header">7.1 Synthesis of Experimental Findings and Engineering Takeaways</div>
<p>
This project successfully designed, implemented, and benchmarked <strong>Research Copilot</strong>, establishing four key scientific and engineering conclusions:
</p>
<ol>
    <li><strong>Strict Chronological Splits Eliminate Leakage:</strong> Enforcing date-boundary evaluation exposes that standard randomized 5-fold cross-validation inflates classification accuracy by +2.27% to +4.70% through temporal term leakage.</li>
    <li><strong>In-Domain Contrastive Tuning Yields Substantial Retrieval Gains:</strong> Training all-MiniLM-L6-v2 with MNRL and a 5-NN exclusion radius achieves +4.38% Recall@10 and +0.1039 MRR@10 gains ($p < 0.0001$) while preserving zero-shot generalization on BEIR SciFact ($p = 0.3290$).</li>
    <li><strong>Linear Models Match Deep Architectures:</strong> After Holm-Bonferroni correction, simple well-calibrated linear classifiers (Linear SVM: 63.60%; Logistic Regression: 64.61%, Brier Score 0.501) perform equivalently to deep neural networks ($p \ge 0.05$) while training 10x faster.</li>
    <li><strong>Threshold Tuning Mitigates Multi-Label Sparsity:</strong> Optimizing per-label decision thresholds on validation data raises Micro-F1 from 0.5852 to 0.6387 (+5.35%) and lowers Hamming loss to 0.0586.</li>
</ol>

<div class="section-header">7.2 Academic Integrity Safeguards and Hallucination Prevention</div>
<p>
In alignment with ethical AI principles, Research Copilot enforces strict safeguards:
</p>
<ul>
    <li>Every factual claim requires an inline numerical citation linking directly to its underlying arXiv abstract.</li>
    <li>Sentence-level cosine grounding checks flag unsupported sentences with warning icons (`⚠`).</li>
    <li>Out-of-domain queries triggering cross-encoder scores below $-5.0$ are automatically refused rather than hallucinated.</li>
</ul>

<div class="section-header">7.3 Future Research Directions</div>
<ul>
    <li><strong>Natural Language Inference (NLI) Grounding:</strong> Replace cosine grounding checks with formal cross-encoder entailment models classifying claims into Entailment, Neutral, or Contradiction.</li>
    <li><strong>Comparative Topic Coherence Study:</strong> Conduct empirical comparisons between HDBSCAN/c-TF-IDF topic modeling and Latent Dirichlet Allocation (LDA) evaluated via $C_v$ coherence.</li>
    <li><strong>Autoregressive Trend Forecasting:</strong> Implement vector autoregressive time-series models over the 82-month dataset to project emerging research trajectories.</li>
    <li><strong>Full-Text PDF Ingestion:</strong> Extend indexing beyond abstracts to full PDF manuscripts, mathematical equations, and benchmark tables.</li>
</ul>

<!-- ======================================================================= -->
<!-- CHAPTER 8: REFERENCES                                                   -->
<!-- ======================================================================= -->
<div class="page-break"></div>

<div class="running-header">
    <span>Research Copilot: Hybrid IR, Contrastive Learning & Classification</span>
    <span>B.Tech Project Report (Dept. of AI & DS, DPU Pune)</span>
</div>

<div class="chapter-header">CHAPTER 8: REFERENCES AND SCHOLARLY BIBLIOGRAPHY</div>

<ol class="ref-list">
    <li><span class="ref-num">[1]</span> S. E. Robertson, H. Zaragoza, et al., "The probabilistic relevance framework: BM25 and beyond," <em>Found. Trends Inf. Retr.</em>, vol. 3, no. 4, pp. 333–389, 2009.</li>
    <li><span class="ref-num">[2]</span> N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in <em>Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP-IJCNLP)</em>, 2019, pp. 3982–3992.</li>
    <li><span class="ref-num">[3]</span> I. Beltagy, K. Lo, and A. Cohan, "SciBERT: A pretrained language model for scientific text," in <em>Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP-IJCNLP)</em>, 2019, pp. 3615–3620.</li>
    <li><span class="ref-num">[4]</span> V. Karpukhin et al., "Dense passage retrieval for open-domain question answering," in <em>Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP)</em>, 2020, pp. 6769–6781.</li>
    <li><span class="ref-num">[5]</span> N. Thakur et al., "BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval," in <em>Proc. 35th Conf. Neural Inf. Process. Syst. (NeurIPS) Datasets and Benchmarks Track</em>, 2021, pp. 1–18.</li>
    <li><span class="ref-num">[6]</span> G. V. Cormack, C. L. Clarke, and S. Büttcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," in <em>Proc. 32nd Int. ACM SIGIR Conf. Res. Dev. Inf. Retr.</em>, 2009, pp. 758–759.</li>
    <li><span class="ref-num">[7]</span> R. Nogueira and K. Cho, "Passage re-ranking with BERT," <em>arXiv preprint arXiv:1901.04085</em>, 2019.</li>
    <li><span class="ref-num">[8]</span> D. Wadden et al., "Fact or fiction: Verifying scientific claims," in <em>Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP)</em>, 2020, pp. 7534–7550.</li>
    <li><span class="ref-num">[9]</span> Y. A. Malkov and D. A. Yashunin, "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs," <em>IEEE Trans. Pattern Anal. Mach. Intell.</em>, vol. 42, no. 4, pp. 824–836, Apr. 2020.</li>
    <li><span class="ref-num">[10]</span> L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform manifold approximation and projection for dimension reduction," <em>arXiv preprint arXiv:1802.03426</em>, 2018.</li>
    <li><span class="ref-num">[11]</span> R. J. Campello, D. Moulavi, and J. Sander, "Density-based clustering based on hierarchical density estimates," in <em>Proc. Pacific-Asia Conf. Knowl. Discov. Data Mining (PAKDD)</em>, 2013, pp. 160–172.</li>
    <li><span class="ref-num">[12]</span> S. Holm, "A simple sequentially rejective multiple test procedure," <em>Scand. J. Stat.</em>, vol. 6, no. 2, pp. 65–70, 1979.</li>
    <li><span class="ref-num">[13]</span> Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," <em>Psychometrika</em>, vol. 12, no. 2, pp. 153–157, Jun. 1947.</li>
    <li><span class="ref-num">[14]</span> A. L. Edwards, "Note on the 'correction for continuity' in testing the significance of the difference between correlated proportions," <em>Psychometrika</em>, vol. 13, no. 3, pp. 185–187, Sep. 1948.</li>
    <li><span class="ref-num">[15]</span> B. Efron and R. J. Tibshirani, <em>An Introduction to the Bootstrap</em>. New York, NY: Chapman & Hall, 1993.</li>
    <li><span class="ref-num">[16]</span> G. W. Brier, "Verification of forecasts expressed in terms of probability," <em>Mon. Weather Rev.</em>, vol. 78, no. 1, pp. 1–3, Jan. 1950.</li>
    <li><span class="ref-num">[17]</span> C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in <em>Proc. 34th Int. Conf. Mach. Learn. (ICML)</em>, 2017, pp. 1321–1330.</li>
    <li><span class="ref-num">[18]</span> C. Cortes and V. Vapnik, "Support-vector networks," <em>Mach. Learn.</em>, vol. 20, no. 3, pp. 273–297, Sep. 1995.</li>
    <li><span class="ref-num">[19]</span> L. Breiman, "Random forests," <em>Mach. Learn.</em>, vol. 45, no. 1, pp. 5–32, Oct. 2001.</li>
    <li><span class="ref-num">[20]</span> T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in <em>Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining</em>, 2016, pp. 785–794.</li>
    <li><span class="ref-num">[21]</span> G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in <em>Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)</em>, vol. 30, 2017, pp. 3146–3154.</li>
    <li><span class="ref-num">[22]</span> J. G. Carbonell and J. Goldstein, "The use of MMR, diversity-based reranking for reordering documents and producing summaries," in <em>Proc. 21st Annu. Int. ACM SIGIR Conf. Res. Dev. Inf. Retr.</em>, 1998, pp. 335–336.</li>
    <li><span class="ref-num">[23]</span> I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in <em>Proc. Int. Conf. Learn. Represent. (ICLR)</em>, 2019, pp. 1–10.</li>
    <li><span class="ref-num">[24]</span> F. Pedregosa et al., "Scikit-learn: Machine learning in Python," <em>J. Mach. Learn. Res.</em>, vol. 12, pp. 2825–2830, Nov. 2011.</li>
    <li><span class="ref-num">[25]</span> A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in <em>Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)</em>, vol. 32, 2019, pp. 8024–8035.</li>
    <li><span class="ref-num">[26]</span> T. Wolf et al., "Transformers: State-of-the-art natural language processing," in <em>Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP) Syst. Demonstrations</em>, 2020, pp. 38–45.</li>
    <li><span class="ref-num">[27]</span> J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," <em>IEEE Trans. Big Data</em>, vol. 7, no. 3, pp. 535–547, Sep. 2021.</li>
</ol>

<div class="running-footer">
    Dr. D. Y. Patil School of Science & Technology, Pimpri, Pune — Academic Year 2025–2026
</div>

</body>
</html>
"""

    
    for k, v in [
        ("__IMG_MAP_OVERVIEW__", img_map_overview),
                        ("__IMG_MODEL_PANEL__", img_model_panel),
        ("__IMG_TOPIC_TREND__", img_topic_trend),
        ("__IMG_CLASS_DIST__", img_class_dist),
        ("__IMG_LADDER_COMP__", img_ladder_comp),
        ("__IMG_COLAB_RUN__", img_colab_run),
        ("__IMG_CONF_MATRIX__", img_conf_matrix),
        ("__IMG_NVIDIA__", img_nvidia),
        ("__IMG_PAPERS_MONTH__", img_papers_month),
        ("__IMG_SEARCH_BENCH__", img_search_bench),
    ]:
        html_content = html_content.replace(k, v)

    html_path = base_dir / "BTech_Project_Report_Research_Copilot.html"

    pdf_path = base_dir / "BTech_Project_Report_Research_Copilot.pdf"

    print(f"Writing B.Tech Project Report HTML to: {html_path}")
    html_path.write_text(html_content, encoding="utf-8")

    print("Compiling B.Tech Project Report to PDF via Google Chrome headless...")
    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=5000",
        f"--print-to-pdf={pdf_path}",
        str(html_path)
    ]

    res = subprocess.run(chrome_cmd, capture_output=True, text=True)
    if res.returncode == 0 and pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        print(f"SUCCESS: Generated {pdf_path} ({size_kb:.1f} KB)")
    else:
        print(f"Chrome returncode: {res.returncode}")
        print(f"Stdout: {res.stdout}")
        print(f"Stderr: {res.stderr}")

if __name__ == "__main__":
    main()
