#!/usr/bin/env python3
# query_related_work.py — B10/F19：串行查 arxiv export API 三类检索词，输出已核验候选文献。
# arxiv API 返回的 entry 即 ID 可达性核验（非编造）。串行 + 3s sleep 尊重 arxiv 限流。
import urllib.request, urllib.parse, time, xml.etree.ElementTree as ET, json, sys

ATOM = "{http://www.w3.org/2005/Atom}"
QUERIES = [
    ("rank_allocation", 'all:"rank allocation" AND all:"LoRA"'),
    ("quant_aware_peft", 'all:"LoRA" AND all:"quantization" AND cat:cs.LG'),
    ("adaptive_lora_rank", 'all:"adaptive" AND all:"LoRA" AND all:"rank"'),
    ("qlora_4bit", 'all:"QLoRA" OR all:"4-bit" AND all:"fine-tuning"'),
    ("rank_pruning_lora", 'all:"rank" AND all:"pruning" AND all:"LoRA"'),
]

def fetch(query, max_results=10):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": query, "start": 0, "max_results": max_results,
        "sortBy": "relevance", "sortOrder": "descending"})
    req = urllib.request.Request(url, headers={"User-Agent": "adaq-lora-relatedwork/1.0 (mailto:research@example.com)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return r.read()
        except Exception as e:
            sys.stderr.write(f"  retry {attempt+1} after err: {e}\n"); time.sleep(5)
    return None

def parse(xml_bytes):
    if not xml_bytes: return []
    out = []
    try:
        root = ET.fromstring(xml_bytes)
    except Exception as e:
        sys.stderr.write(f"  parse err: {e}\n"); return []
    for e in root.findall(f"{ATOM}entry"):
        aid = e.findtext(f"{ATOM}id", "")
        aid = aid.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "")
        title = (e.findtext(f"{ATOM}title", "") or "").strip().replace("\n", " ")
        pub = e.findtext(f"{ATOM}published", "")[:10]
        auths = [a.findtext(f"{ATOM}name", "") for a in e.findall(f"{ATOM}author")]
        summ = (e.findtext(f"{ATOM}summary", "") or "").strip().replace("\n", " ")
        out.append({"id": aid, "title": title, "published": pub,
                    "authors": auths[:4], "n_authors": len(auths), "summary": summ[:400]})
    return out

result = {}
for name, q in QUERIES:
    sys.stderr.write(f"[query] {name}: {q}\n")
    entries = parse(fetch(q, 10))
    result[name] = entries
    sys.stderr.write(f"  -> {len(entries)} entries\n")
    time.sleep(3.5)  # arxiv 限流

# 去重汇总
seen = {}
for name, entries in result.items():
    for e in entries:
        base = e["id"].split("v")[0]
        if base not in seen:
            e2 = dict(e); e2["found_via"] = [name]; seen[base] = e2
        else:
            seen[base]["found_via"].append(name)
print(json.dumps({"by_query": result, "unique": sorted(seen.values(), key=lambda x: x["published"], reverse=True)}, ensure_ascii=False, indent=1))
