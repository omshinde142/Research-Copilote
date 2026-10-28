import json, os, re, yaml, numpy as np, pandas as pd, faiss
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

D = Path("data")
df = pd.read_pickle(D / "papers.pkl")

# Read config to select base vs fine-tuned model with safe fallback
model_name = "all-MiniLM-L6-v2"
index_file = D / "index.faiss"
cfg_file = Path("config.yaml")
if cfg_file.exists():
    try:
        conf = yaml.safe_load(cfg_file.read_text())
        emb_cfg = conf.get("embedding", {})
        ft_path = Path(emb_cfg.get("fine_tuned_path", "data/models/embedder-ft"))
        if emb_cfg.get("active_model") == "fine_tuned" and ft_path.exists():
            model_name = str(ft_path)
            if (D / "index_ft.faiss").exists():
                index_file = D / "index_ft.faiss"
    except Exception:
        pass

index = faiss.read_index(str(index_file))
try: faiss.downcast_index(index).hnsw.efSearch = 64
except Exception: pass
labels = {int(k): v for k, v in json.loads((D / "clusters.json").read_text()).items()}
metrics = json.loads((D / "metrics.json").read_text())
sb = D / "scibert_metrics.json"
if sb.exists(): metrics["accuracy"].update(json.loads(sb.read_text()))
enc = SentenceTransformer(model_name)
rerank = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
app = FastAPI(title="Research Copilot")

def paper(i, score=None):
    r = df.iloc[i]
    return {"i": int(i), "title": r.title, "abstract": r.abstract, "category": r.category,
            "year": int(r.year), "cluster": int(r.cluster), "arxiv": r.id, "published": bool(r.get("published", False)), "score": score}

bm = BM25Okapi([(t + " " + a).lower().split() for t, a in zip(df.title, df.abstract)])
MIN_SCORE = float(os.getenv("MIN_SCORE", "-5"))     # below this the best match is too weak to answer from

def search(q, k=8):
    v = enc.encode([q], normalize_embeddings=True).astype("float32")
    _, I = index.search(v, 50); dense = [int(i) for i in I[0] if i >= 0]
    sparse = [int(i) for i in np.argsort(-bm.get_scores(q.lower().split()))[:50]]
    fused = {}                                       # stage 1: keyword + meaning search, merged with RRF
    for lst in (dense, sparse):
        for r, i in enumerate(lst): fused[i] = fused.get(i, 0) + 1 / (60 + r)
    I = sorted(fused, key=fused.get, reverse=True)[:30]
    s = rerank.predict([(q, df.title.iloc[i] + ". " + df.abstract.iloc[i]) for i in I])   # stage 2: re-rank
    return [paper(I[j], float(s[j])) for j in np.argsort(-s)[:k]]

def extractive(q, hits, n=4, lam=0.7):
    """Pick n quotes that are relevant but not repetitive (MMR). Every sentence is a real quote."""
    sents = [(i, s) for i, h in enumerate(hits, 1) for s in re.split(r"(?<=[.!?])\s+", h["abstract"]) if len(s) > 40]
    qv = enc.encode([q], normalize_embeddings=True)[0]
    sv = enc.encode([s for _, s in sents], normalize_embeddings=True)
    rel, chosen = sv @ qv, []
    while len(chosen) < min(n, len(sents)):
        sc = [(-9 if j in chosen else lam * rel[j] - (1 - lam) * max([sv[j] @ sv[c] for c in chosen] or [0]), j) for j in range(len(sents))]
        chosen.append(max(sc)[1])
    return " ".join(f"{sents[j][1]} [{sents[j][0]}]" for j in sorted(chosen, key=lambda j: sents[j][0]))

_llm = None
def generate(q, hits):
    """Claude API if ANTHROPIC_API_KEY is set, else a local Qwen model if USE_LLM=1, else extractive."""
    global _llm
    ctx = "\n".join(f"[{n}] {h['title']}. {h['abstract'][:700]}" for n, h in enumerate(hits, 1))
    prompt = ("Answer the question using ONLY these paper abstracts. Cite sources like [1] after every claim. "
              "If the abstracts do not contain the answer, say so.\n\nQuestion: " + q + "\n\n" + ctx)
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        r = anthropic.Anthropic().messages.create(model=os.getenv("CLAUDE_MODEL", "claude-haiku-5-5"), max_tokens=500,
                                                  messages=[{"role": "user", "content": prompt}])
        return r.content[0].text, "Claude API"
    if os.getenv("USE_LLM") == "1":
        if _llm is None:
            from transformers import pipeline
            _llm = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
        out = _llm([{"role": "user", "content": prompt}], max_new_tokens=300)[0]["generated_text"][-1]["content"]
        return out, "Qwen2.5-1.5B (local)"
    return extractive(q, hits), "extractive"

def check(answer, hits):
    """Flags each answer sentence whose cited abstract does not contain a similar sentence (embedding proxy)."""
    parts = []
    for s in re.split(r"(?<=[.!?])\s+", answer.strip()):
        if not s: continue
        cites = [int(n) for n in re.findall(r"\[(\d+)\]", s) if 0 < int(n) <= len(hits)]
        body = re.sub(r"\[\d+\]", "", s).strip(); score = 0.0
        if cites and len(body) > 20:
            sv = enc.encode([body], normalize_embeddings=True)[0]
            for n in cites:
                ss = [x for x in re.split(r"(?<=[.!?])\s+", hits[n - 1]["abstract"]) if len(x) > 20]
                if ss: score = max(score, float((enc.encode(ss, normalize_embeddings=True) @ sv).max()))
        parts.append({"text": s, "score": round(score, 2), "supported": bool(cites) and score >= 0.45})
    return parts

xs, ys = df.x.values, df.y.values
nx, ny = (xs - xs.min()) / np.ptp(xs), (ys - ys.min()) / np.ptp(ys)
MAP = {"x": np.round(nx, 4).tolist(), "y": np.round(ny, 4).tolist(), "c": df.cluster.tolist(), "metrics": metrics,
       "clusters": [{"id": c, "label": labels[c], "x": float(nx[df.cluster == c].mean()),
                     "y": float(ny[df.cluster == c].mean()), "n": int((df.cluster == c).sum())} for c in labels]}

@app.get("/api/map")
def get_map(): return MAP

@app.get("/api/paper/{i}")
def get_paper(i: int): return paper(i)

@app.get("/api/search")
def do_search(q: str): return search(q, 10)

@app.get("/api/ask")
def ask(q: str):
    hits = search(q, 6)
    if hits[0]["score"] < MIN_SCORE:                 # refuse instead of guessing
        return {"answer": "", "parts": [], "mode": "none", "confident": False, "sources": hits}
    ans, mode = generate(q, hits)
    parts = check(ans, hits) if mode != "extractive" else \
        [{"text": s, "supported": True} for s in re.split(r"(?<=\])\s+", ans) if s]
    return {"answer": ans, "parts": parts, "mode": mode, "confident": True, "sources": hits}

@app.get("/api/similar/{i}")
def similar(i: int):
    _, I = index.search(index.reconstruct(i).reshape(1, -1), 9)
    return [paper(int(j)) for j in I[0] if j != i][:8]

@app.get("/api/trend/{c}")
def trend(c: int):
    s = df[df.cluster == c].year.value_counts().sort_index()
    return {"years": s.index.tolist(), "counts": s.values.tolist(), "label": labels[c]}

@app.get("/")
def home(): return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")
