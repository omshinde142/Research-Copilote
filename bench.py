"""Measures search quality. For 500 papers, can each method find the paper's abstract from its title alone?
Usage: python bench.py   (run after prepare.py)"""
import json, numpy as np, pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

D = Path("data")
df = pd.read_pickle(D / "papers.pkl").sample(n=5000, random_state=1).reset_index(drop=True)
Q, docs = df.title.head(500).tolist(), df.abstract.tolist()

def score(ranks):                       # the right abstract for query i is document i
    rr = [1 / (r[:10].index(i) + 1) if i in r[:10] else 0 for i, r in enumerate(ranks)]
    return {"recall@10": round(float(np.mean([x > 0 for x in rr])), 3), "mrr": round(float(np.mean(rr)), 3)}

res = {}
tv = TfidfVectorizer(stop_words="english", sublinear_tf=True); Xd = tv.fit_transform(docs)
S = (tv.transform(Q) @ Xd.T).toarray()
res["TF-IDF"] = score([list(np.argsort(-s)[:30]) for s in S])
bm = BM25Okapi([d.lower().split() for d in docs])
sp = [list(np.argsort(-bm.get_scores(q.lower().split()))[:30]) for q in Q]
res["BM25"] = score(sp)
m = SentenceTransformer("all-MiniLM-L6-v2")
E = m.encode(docs, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
dense = [list(np.argsort(-s)[:30]) for s in m.encode(Q, normalize_embeddings=True) @ E.T]
res["SBERT embeddings"] = score(dense)
def rrf(a, b):                      # reciprocal rank fusion: merge keyword and meaning rankings
    f = {}
    for l in (a, b):
        for r, i in enumerate(l): f[i] = f.get(i, 0) + 1 / (60 + r)
    return sorted(f, key=f.get, reverse=True)[:30]
hyb = [rrf(a, b) for a, b in zip(dense, sp)]
res["Hybrid BM25 + SBERT"] = score(hyb)
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"); rer = []
for q, cand in zip(Q, hyb):
    s = ce.predict([(q, docs[j]) for j in cand]); rer.append([cand[k] for k in np.argsort(-s)])
res["Hybrid + cross-encoder re-rank"] = score(rer)
for n, v in res.items(): print(n, v)
mp = D / "metrics.json"; M = json.loads(mp.read_text())
M["retrieval"] = {"task": "Search quality: find a paper's abstract from its title (500 queries, 5,000 papers)", "results": res}
mp.write_text(json.dumps(M))
