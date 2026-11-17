#!/usr/bin/env python3
"""B8 自动双标注：规则A（保守）+ 规则B（宽松），生成 annotated TSV。

用法: python b8_auto_annotate.py <worksheet.tsv> <annotated.tsv>
"""
import csv, re, sys


def norm_num(s):
    s = str(s).strip().replace(",", "").replace("$", "")
    return s.rstrip(".")


def ruleA(row):
    """保守：flex==gold 且 gold 以数字边界出现在 model_output 中。"""
    g = norm_num(row["gold"])
    f = norm_num(row.get("flex_extracted") or "")
    if not g:
        return "UNCLEAR"
    out = row.get("model_output") or ""
    if f == g and re.search(r"(?<![\d.])" + re.escape(g) + r"(?![\d.])", out):
        return "Y"
    return "N"


def ruleB(row):
    """宽松：flex==gold，或 model_output 末段含答案表述上下文中的 gold。"""
    g = norm_num(row["gold"])
    f = norm_num(row.get("flex_extracted") or "")
    if not g:
        return "UNCLEAR"
    if f == g:
        return "Y"
    out = (row.get("model_output") or "").strip()
    tail = out[-300:]
    if re.search(r"(answer|最终|因此|=|\$)\D{0,6}" + re.escape(g) + r"(?![\d.])", tail, re.I):
        return "Y"
    return "N"


def main():
    src, dst = sys.argv[1], sys.argv[2]
    rows = list(csv.DictReader(open(src, encoding="utf-8"), delimiter="\t"))
    fields = list(rows[0].keys())
    for col in ("annotA_flex_captures_intent", "annotB_flex_captures_intent"):
        if col not in fields:
            fields.append(col)
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            r["annotA_flex_captures_intent"] = ruleA(r)
            r["annotB_flex_captures_intent"] = ruleB(r)
            w.writerow(r)
    a = sum(1 for r in rows if r["annotA_flex_captures_intent"] == "Y")
    b = sum(1 for r in rows if r["annotB_flex_captures_intent"] == "Y")
    print(f"{dst}: {len(rows)} rows  A(Y)={a} B(Y)={b}")


if __name__ == "__main__":
    main()
