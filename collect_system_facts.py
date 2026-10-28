#!/usr/bin/env python3
"""
collect_system_facts.py
Runs 50 measured queries to gather precise p50/p95 latency per stage,
memory usage, index size, example answers, refusal cases, and failure cases.
"""

import json
import os
import platform
import re
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import psutil
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

def main():
    print("Collecting system facts and measuring latency across 50 queries...")
    t_start = time.time()
    
    # 1. Hardware and Environment Info
    mem = psutil.virtual_memory()
    proc = psutil.Process()
    hw_info = {
        "os": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "python_version": platform.python_version()
    }

    # 2. File and Index Sizes
    base_dir = Path("/Users/durveshnarkhede/Desktop/COPILOTE")
    data_dir = base_dir / "data"
    file_sizes = {
        "papers_pkl_bytes": (data_dir / "papers.pkl").stat().st_size,
        "index_faiss_bytes": (data_dir / "index.faiss").stat().st_size if (data_dir / "index.faiss").exists() else None,
        "index_ft_faiss_bytes": (data_dir / "index_ft.faiss").stat().st_size if (data_dir / "index_ft.faiss").exists() else None,
        "splits_json_bytes": (data_dir / "splits.json").stat().st_size,
        "data_report_json_bytes": (data_dir / "data_report.json").stat().st_size,
        "metrics_json_bytes": (data_dir / "metrics.json").stat().st_size,
        "embedder_ft_dir_bytes": sum(f.stat().st_size for f in (data_dir / "models" / "embedder-ft").glob("**/*") if f.is_file()) if (data_dir / "models" / "embedder-ft").exists() else None
    }

    # 3. Load App Data and Models
    df = pd.read_pickle(data_dir / "papers.pkl")
    papers_served_count = len(df)
    
    model_name = "data/models/embedder-ft" if (data_dir / "models" / "embedder-ft").exists() else "all-MiniLM-L6-v2"
    index_file = data_dir / "index_ft.faiss" if (data_dir / "index_ft.faiss").exists() else data_dir / "index.faiss"
    
    index = faiss.read_index(str(index_file))
    try:
        faiss.downcast_index(index).hnsw.efSearch = 64
    except Exception:
        pass

    labels = {int(k): v for k, v in json.loads((data_dir / "clusters.json").read_text()).items()}
    enc = SentenceTransformer(model_name)
    rerank = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    bm = BM25Okapi([(t + " " + a).lower().split() for t, a in zip(df.title, df.abstract)])

    # Memory after loading
    rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
    vms_mb = round(proc.memory_info().vms / (1024 * 1024), 2)

    MIN_SCORE = -5.0

    def search_profiled(q, k=8):
        # Stage 1: Dense Query Encoding
        t0 = time.perf_counter()
        v = enc.encode([q], normalize_embeddings=True).astype("float32")
        t_enc = (time.perf_counter() - t0) * 1000.0

        # Stage 2: FAISS Dense Search
        t0 = time.perf_counter()
        _, I = index.search(v, 50)
        dense = [int(i) for i in I[0] if i >= 0]
        t_faiss = (time.perf_counter() - t0) * 1000.0

        # Stage 3: BM25 Lexical Search
        t0 = time.perf_counter()
        sparse = [int(i) for i in np.argsort(-bm.get_scores(q.lower().split()))[:50]]
        t_bm25 = (time.perf_counter() - t0) * 1000.0

        # Stage 4: Reciprocal Rank Fusion
        t0 = time.perf_counter()
        fused = {}
        for lst in (dense, sparse):
            for r, i in enumerate(lst):
                fused[i] = fused.get(i, 0) + 1 / (60 + r)
        cand_ids = sorted(fused, key=fused.get, reverse=True)[:30]
        t_rrf = (time.perf_counter() - t0) * 1000.0

        # Stage 5: Cross-Encoder Re-Ranking
        t0 = time.perf_counter()
        pairs = [(q, df.title.iloc[i] + ". " + df.abstract.iloc[i]) for i in cand_ids]
        scores = rerank.predict(pairs)
        ranked_order = np.argsort(-scores)[:k]
        top_hits = []
        for j in ranked_order:
            idx = cand_ids[j]
            r = df.iloc[idx]
            top_hits.append({
                "i": int(idx),
                "title": r.title,
                "abstract": r.abstract,
                "category": r.primary_category if "primary_category" in r else r.category,
                "year": int(r.year),
                "cluster": int(r.cluster),
                "arxiv": r.id,
                "score": float(scores[j])
            })
        t_rerank = (time.perf_counter() - t0) * 1000.0
        t_total = t_enc + t_faiss + t_bm25 + t_rrf + t_rerank

        return {
            "hits": top_hits,
            "timings_ms": {
                "encoding": t_enc,
                "faiss": t_faiss,
                "bm25": t_bm25,
                "rrf": t_rrf,
                "reranking": t_rerank,
                "total_search": t_total
            }
        }

    def extractive_profiled(q, hits, n=4, lam=0.7):
        t0 = time.perf_counter()
        sents = [(i, s) for i, h in enumerate(hits, 1) for s in re.split(r"(?<=[.!?])\s+", h["abstract"]) if len(s) > 40]
        if not sents:
            return "", 0.0
        qv = enc.encode([q], normalize_embeddings=True)[0]
        sv = enc.encode([s for _, s in sents], normalize_embeddings=True)
        rel = sv @ qv
        chosen = []
        while len(chosen) < min(n, len(sents)):
            sc = [(-9 if j in chosen else lam * rel[j] - (1 - lam) * max([sv[j] @ sv[c] for c in chosen] or [0]), j) for j in range(len(sents))]
            chosen.append(max(sc)[1])
        ans = " ".join(f"{sents[j][1]} [{sents[j][0]}]" for j in sorted(chosen, key=lambda j: sents[j][0]))
        t_mmr = (time.perf_counter() - t0) * 1000.0
        return ans, t_mmr

    def check_profiled(answer, hits):
        t0 = time.perf_counter()
        parts = []
        for s in re.split(r"(?<=[.!?])\s+", answer.strip()):
            if not s:
                continue
            cites = [int(n) for n in re.findall(r"\[(\d+)\]", s) if 0 < int(n) <= len(hits)]
            body = re.sub(r"\[\d+\]", "", s).strip()
            score = 0.0
            if cites and len(body) > 20:
                sv = enc.encode([body], normalize_embeddings=True)[0]
                for n in cites:
                    ss = [x for x in re.split(r"(?<=[.!?])\s+", hits[n - 1]["abstract"]) if len(x) > 20]
                    if ss:
                        score = max(score, float((enc.encode(ss, normalize_embeddings=True) @ sv).max()))
            parts.append({"text": s, "score": round(score, 2), "supported": bool(cites) and score >= 0.45})
        t_check = (time.perf_counter() - t0) * 1000.0
        return parts, t_check

    # 4. Run 50 Diverse Queries
    sample_queries = [
        "Contrastive learning for graph neural networks",
        "Diffusion models for high resolution image synthesis",
        "Parameter efficient fine tuning LoRA for large language models",
        "Reinforcement learning from human feedback alignment",
        "Vision transformers for zero shot object detection",
        "Autonomous driving motion planning and trajectory optimization",
        "Federated learning with differential privacy guarantees",
        "Quantum error correction surface codes",
        "Self-supervised speech representation learning",
        "Knowledge graph embedding link prediction",
        "Neural radiance fields 3D view synthesis NeRF",
        "Retrieval augmented generation hallucination reduction",
        "Adversarial robustness deep learning defense certified bounds",
        "Graph neural networks molecular property prediction",
        "Sparse attention mechanisms for long context transformers",
        "Unsupervised domain adaptation covariate shift",
        "Continual learning catastrophic forgetting replay buffer",
        "Explainable AI SHAP values feature attribution",
        "Time series anomaly detection transformers",
        "Multi-modal video captioning temporal grounding",
        "Zero-shot semantic segmentation open vocabulary",
        "Fairness in machine learning algorithmic bias auditing",
        "Causal inference observational data treatment effect",
        "Neural architecture search hardware aware pruning",
        "Bayesian neural networks uncertainty estimation",
        "Point cloud segmentation 3D geometric deep learning",
        "Semantic parsing text to SQL database",
        "Speech recognition conformer streaming latency",
        "Robot manipulation reinforcement learning sim to real",
        "Multi-agent reinforcement learning cooperative game theory",
        "Energy based models contrastive divergence",
        "Optical flow estimation deep convolutional networks",
        "Monocular depth estimation self-supervised geometry",
        "Code generation large language models programming",
        "Question answering passage retrieval dense bi-encoder",
        "Deep clustering latent representation space",
        "Meta learning few-shot learning MAML",
        "Out of distribution detection Mahalanobis distance",
        "Privacy preserving synthetic data generation GANs",
        "Graph contrastive learning topology perturbation",
        "Self-attention linear complexity state space models",
        "Text classification BERT vs logistic regression",
        "Hierarchical multi-label classification taxonomic loss",
        "Image inpainting partial convolutions deep neural network",
        "Active learning query by committee pool based",
        "Physics informed neural networks partial differential equations",
        "Document layout analysis table structure recognition",
        "Event extraction natural language processing triggers",
        "Spiking neural networks neuromorphic computing energy",
        "Direct preference optimization reinforcement learning alignment"
    ]

    timings = {
        "encoding": [],
        "faiss": [],
        "bm25": [],
        "rrf": [],
        "reranking": [],
        "total_search": [],
        "total_ask": []
    }

    print(f"Executing 50 queries across all retrieval stages...")
    for q in sample_queries:
        res = search_profiled(q, k=8)
        hits = res["hits"]
        t = res["timings_ms"]
        timings["encoding"].append(t["encoding"])
        timings["faiss"].append(t["faiss"])
        timings["bm25"].append(t["bm25"])
        timings["rrf"].append(t["rrf"])
        timings["reranking"].append(t["reranking"])
        timings["total_search"].append(t["total_search"])

        # Measure Ask
        t_ask_start = time.perf_counter()
        if hits and hits[0]["score"] >= MIN_SCORE:
            ans, t_mmr = extractive_profiled(q, hits, n=4)
            parts, t_chk = check_profiled(ans, hits)
        t_ask_total = (time.perf_counter() - t_ask_start) * 1000.0 + t["total_search"]
        timings["total_ask"].append(t_ask_total)

    # Latency Stats
    def get_stats(arr):
        return {
            "median_p50_ms": round(float(np.median(arr)), 2),
            "mean_ms": round(float(np.mean(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2)
        }

    latency_summary = {k: get_stats(v) for k, v in timings.items()}

    # 5. Collect 4 Detailed Real Example Queries
    test_queries_for_qa = [
        "Contrastive learning for graph neural networks",
        "Diffusion models for high resolution image synthesis",
        "Parameter efficient fine tuning LoRA",
        "Federated learning differential privacy"
    ]
    examples = []
    for q in test_queries_for_qa:
        res = search_profiled(q, k=6)
        hits = res["hits"]
        ans, _ = extractive_profiled(q, hits, n=3)
        parts, _ = check_profiled(ans, hits)
        examples.append({
            "query": q,
            "top_paper_title": hits[0]["title"],
            "top_paper_arxiv_id": hits[0]["arxiv"],
            "top_paper_score": hits[0]["score"],
            "top_paper_category": hits[0]["category"],
            "answer": ans,
            "citations_count": len(re.findall(r"\[\d+\]", ans)),
            "checked_parts": parts
        })

    # 6. Refusal Case: Query with no scientific grounding, producing low cross-encoder score (< -5.0)
    refusal_query = "quantum teleportation recipe for baking banana bread in hypergravity"
    ref_res = search_profiled(refusal_query, k=6)
    ref_hits = ref_res["hits"]
    top_ref_score = ref_hits[0]["score"] if ref_hits else -99.0
    refusal_triggered = top_ref_score < MIN_SCORE
    refusal_case = {
        "query": refusal_query,
        "top_paper_title": ref_hits[0]["title"] if ref_hits else "None",
        "top_score": top_ref_score,
        "refusal_threshold": MIN_SCORE,
        "refused": refusal_triggered,
        "app_response": "Refused (no confident papers found)" if refusal_triggered else "Not refused"
    }

    # 7. Failure Case: Polysemous/ambiguous acronym query
    failure_query = "Transformers in electric power transmission networks"
    fail_res = search_profiled(failure_query, k=6)
    fail_hits = fail_res["hits"]
    fail_ans, _ = extractive_profiled(failure_query, fail_hits, n=3)
    fail_parts, _ = check_profiled(fail_ans, fail_hits)
    failure_case = {
        "query": failure_query,
        "top_paper_title": fail_hits[0]["title"] if fail_hits else "None",
        "top_paper_category": fail_hits[0]["category"] if fail_hits else "None",
        "top_score": fail_hits[0]["score"] if fail_hits else -99.0,
        "observed_failure": "The search engine retrieves AI Transformer language models applied to grid data instead of mechanical/electrical power step-up/step-down transformers due to Computer Science corpus domain boundaries.",
        "answer": fail_ans,
        "checked_parts": fail_parts
    }

    output = {
        "hardware": hw_info,
        "memory": {
            "rss_mb": rss_mb,
            "vms_mb": vms_mb
        },
        "papers_served_count": papers_served_count,
        "file_sizes": file_sizes,
        "models_used": {
            "embedder": model_name,
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "index_file": str(index_file)
        },
        "n_queries_tested": len(sample_queries),
        "latency_summary_ms": latency_summary,
        "real_examples": examples,
        "refusal_case": refusal_case,
        "failure_case": failure_case
    }

    out_file = data_dir / "measured_system_facts.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nSystem facts saved successfully to {out_file} in {round(time.time() - t_start, 2)}s.")
    print(f"Search p50: {latency_summary['total_search']['median_p50_ms']} ms | p95: {latency_summary['total_search']['p95_ms']} ms")
    print(f"Ask p50: {latency_summary['total_ask']['median_p50_ms']} ms | p95: {latency_summary['total_ask']['p95_ms']} ms")

if __name__ == "__main__":
    main()
