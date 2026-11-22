#!/usr/bin/env python
"""Generate two alternative rank allocations in ranks.json format.

1. ranks_normaware.json -- r_l proportional to ||W_l||_F only (no epsilon term),
   same budget/r_min/r_max as the AdaQ allocation, so the two differ only in the
   signal used.
2. ranks_extreme.json   -- a maximum-contrast allocation (q_proj 14, v_proj 2,
   same total 512) used as a dose-response manipulation check: if a far more
   aggressive allocation still cannot move accuracy, the null is not a
   consequence of a weak intervention.
"""
import json
import os

import torch
from safetensors.torch import load_file

MODEL = os.environ.get("ADAQ_MODEL", "models/Meta-Llama-3-8B")
BUDGET, R_MIN, R_MAX = 512, 4, 128
OUT = {"norm": "configs/ranks_normaware.json", "extreme": "configs/ranks_extreme.json"}


def load_norms():
    index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
    wm = index["weight_map"]
    cache, norms = {}, {}
    for key, fname in sorted(wm.items()):
        if not key.endswith(".weight"):
            continue
        if "q_proj" not in key and "v_proj" not in key:
            continue
        if fname not in cache:
            cache[fname] = load_file(os.path.join(MODEL, fname))
        norms[key] = float(cache[fname][key].float().norm())
        cache.pop(fname, None) if False else None
    return norms


def allocate(norms):
    total = sum(norms.values())
    raw = {k: BUDGET * v / total for k, v in norms.items()}
    ranks = {k: max(R_MIN, min(R_MAX, int(round(v)))) for k, v in raw.items()}
    cur = sum(ranks.values())
    if cur != BUDGET:
        order = sorted(((k, raw[k] - ranks[k]) for k in ranks), key=lambda x: x[1],
                       reverse=(cur < BUDGET))
        i = 0
        while cur != BUDGET and i < 10000:
            k = order[i % len(order)][0]
            step = 1 if cur < BUDGET else -1
            if R_MIN <= ranks[k] + step <= R_MAX:
                ranks[k] += step
                cur += step
            i += 1
    return ranks


def main():
    norms = load_norms()
    print("modules:", len(norms))
    ranks = allocate(norms)
    json.dump({"model": MODEL, "budget": BUDGET, "r_min": R_MIN, "r_max": R_MAX,
               "per_layer_ranks": ranks}, open(OUT["norm"], "w"), indent=2)
    q = [v for k, v in ranks.items() if "q_proj" in k]
    v = [v for k, v in ranks.items() if "v_proj" in k]
    print("norm-aware: sum=%d q_mean=%.2f v_mean=%.2f q_range=%d..%d v_range=%d..%d"
          % (sum(ranks.values()), sum(q) / len(q), sum(v) / len(v), min(q), max(q), min(v), max(v)))

    ext = {}
    for k in norms:
        ext[k] = 14 if "q_proj" in k else 2
    total = sum(ext.values())
    assert total == BUDGET, total
    json.dump({"model": MODEL, "budget": BUDGET, "r_min": R_MIN, "r_max": R_MAX,
               "per_layer_ranks": ext}, open(OUT["extreme"], "w"), indent=2)
    print("extreme: sum=%d q=14 v=2 (uniform reference 8)" % total)


if __name__ == "__main__":
    main()
