import urllib.request
import xml.etree.ElementTree as ET
import time
import json
import calendar
from pathlib import Path
from collections import defaultdict

cache_file = Path("data/arxiv_true_cache.jsonl")
cached_months = defaultdict(list)
seen_ids = set()

if cache_file.exists():
    with open(cache_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            aid = str(d.get("id"))
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            ym = str(d.get("created_date"))[:7]
            cached_months[ym].append(d)

print(f"Existing cache: {len(seen_ids)} unique papers across {len(cached_months)} months.")

target_months = []
# 2023-09 to 2023-12
for m in range(9, 13):
    target_months.append(f"2023-{m:02d}")
# 2024
for m in range(1, 13):
    target_months.append(f"2024-{m:02d}")
# 2025
for m in range(1, 13):
    target_months.append(f"2025-{m:02d}")
# 2026
for m in range(1, 11):
    target_months.append(f"2026-{m:02d}")

TARGET_PER_MONTH = 305  # buffer slightly above 300
headers = {"User-Agent": "ResearchCopilot/2.0 (student@dypatil.edu)"}

VALID_CS = {
    'cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL', 'cs.CR', 'cs.CV', 'cs.CY',
    'cs.DB', 'cs.DC', 'cs.DL', 'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL', 'cs.GL', 'cs.GR',
    'cs.GT', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO', 'cs.MA', 'cs.MM', 'cs.MS',
    'cs.NA', 'cs.NE', 'cs.NI', 'cs.OH', 'cs.OS', 'cs.PF', 'cs.PL', 'cs.RO', 'cs.SC',
    'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY'
}

print(f"Targeting at least {TARGET_PER_MONTH} papers/month for {len(target_months)} months ({target_months[0]} to {target_months[-1]})...")

for ym in target_months:
    current_count = len(cached_months[ym])
    if current_count >= TARGET_PER_MONTH:
        print(f"[{ym}] already has {current_count} papers. Skipping.")
        continue

    needed = TARGET_PER_MONTH - current_count
    y, m = int(ym[:4]), int(ym[5:7])
    last_day = calendar.monthrange(y, m)[1]
    from_date = f"{y}{m:02d}010000"
    to_date = f"{y}{m:02d}{last_day:02d}2359"

    # Start index is current_count to avoid duplicates
    start_idx = current_count
    # Request slightly more than needed in case of non-CS primary categories
    fetch_count = min(needed + 25, 300)

    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=cat:cs.*+AND+submittedDate:[{from_date}+TO+{to_date}]"
        f"&start={start_idx}&max_results={fetch_count}"
    )

    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            t0 = time.time()
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=25) as r:
                xml_data = r.read()
                root = ET.fromstring(xml_data)
                entries = root.findall("{http://www.w3.org/2005/Atom}entry")
                
                batch = []
                for e in entries:
                    aid_elem = e.find("{http://www.w3.org/2005/Atom}id")
                    if aid_elem is None or not aid_elem.text:
                        continue
                    aid_raw = aid_elem.text.split("/abs/")[-1]
                    aid = aid_raw.split("v")[0] if ("v" in aid_raw and aid_raw[0].isdigit()) else aid_raw
                    if aid in seen_ids:
                        continue

                    title_elem = e.find("{http://www.w3.org/2005/Atom}title")
                    title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else ""
                    
                    snode = e.find("{http://www.w3.org/2005/Atom}summary")
                    abstract = " ".join(snode.text.split()) if (snode is not None and snode.text) else ""
                    if len(abstract) < 40 or len(title) < 5:
                        continue

                    pub_elem = e.find("{http://www.w3.org/2005/Atom}published")
                    pub = pub_elem.text[:10] if pub_elem is not None and pub_elem.text else f"{ym}-15"

                    prim_elem = e.find("{http://arxiv.org/schemas/atom}primary_category")
                    prim = prim_elem.attrib.get("term") if prim_elem is not None else None
                    all_c = [c.attrib.get("term") for c in e.findall("{http://www.w3.org/2005/Atom}category") if c.attrib.get("term")]
                    cs_cats = [c for c in all_c if c in VALID_CS]
                    if not cs_cats:
                        continue
                    if prim not in VALID_CS:
                        prim = cs_cats[0]

                    doi_node = e.find("{http://arxiv.org/schemas/atom}doi")
                    jr_node = e.find("{http://arxiv.org/schemas/atom}journal_ref")
                    published = (doi_node is not None) or (jr_node is not None)

                    rec = {
                        "id": aid,
                        "title": title,
                        "abstract": abstract,
                        "categories": cs_cats,
                        "primary_category": prim,
                        "category": prim,
                        "year": int(pub[:4]),
                        "created_date": pub,
                        "published": published,
                        "source": "arxiv_api"
                    }
                    seen_ids.add(aid)
                    batch.append(rec)
                    cached_months[ym].append(rec)

                with open(cache_file, "a") as f:
                    for r in batch:
                        f.write(json.dumps(r) + "\n")

                elapsed = time.time() - t0
                print(f"[{ym}] Fetched {len(batch)} papers in {elapsed:.2f}s (Month Total: {len(cached_months[ym])})")
                time.sleep(3.2)
                break
        except Exception as exc:
            retries += 1
            print(f"[{ym}] Error fetching (attempt {retries}/{max_retries}): {exc}. Sleeping {4 * retries}s...")
            time.sleep(4 * retries)

print(f"\nFinished! Total unique papers in cache: {len(seen_ids)}")
