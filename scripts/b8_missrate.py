#!/usr/bin/env python3
"""量化 flex 漏抽率：flex!=gold 但 model_output 含 gold（数字边界）。"""
import csv, re


def norm_num(s):
    s = str(s).strip().replace(",", "").replace("$", "")
    return s.rstrip(".")


for arm in ("adaq", "qlora"):
    path = f"docs/b8-worksheet/{arm}_annotated.tsv"
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    miss = tot = 0
    for r in rows:
        g = norm_num(r["gold"])
        f = norm_num(r.get("flex_extracted") or "")
        out = r.get("model_output") or ""
        if not g:
            continue
        tot += 1
        if f != g and re.search(r"(?<![\d.])" + re.escape(g) + r"(?![\d.])", out):
            miss += 1
    print(f"{arm}: flex 漏抽(输出含gold但抓错) {miss}/{tot} = {miss/tot:.4f}")
