#!/usr/bin/env python3
"""
audit_flex_extract.py — B8: GSM8K flexible-extract 抽取器人工审计工具。

背景：14B 全部结论建立在 lm_eval gsm8k 的 flexible-extract（flex）指标上。
flex 正则 (-?[$0-9.,]{2,})|(-?[0-9]+) group_select=-1 取「生成中最后一个数字串」。
对 instruct 模型（不输出 #### N），flex 常能抓到答案，但也可能抓到推理中的数字/步骤号
→ 必须人工审计抽取是否真的捕获了模型意图答案（JCR 阻断项 B8，阈值 Cohen's κ≥0.95）。

三模式：
  inspect   <samples.jsonl>       打印首条记录键+样例（适配 lm_eval 版本字段；先跑这个）
  worksheet <samples_dir|jsonl> --arm NAME --out TSV
                                  生成双标注工作表（每样本：问题/gold/原始输出/flex抽取/strict抽取/自动比对 + 两列空白人工标注）
  kappa     <annotated.tsv>       读人工双标注 TSV → Cohen's κ(A vs B) + flex 准确率 + verdict

★依赖说明：当前全仓库零 samples_*.jsonl（评测从未用 --log_samples）→
B8 不是零 GPU 任务：须先用 run_b8_flex_audit.sh 带 --log_samples 重跑 14B GSM8K
（排队 phase3 之后），生成 samples 后再跑本工具 worksheet/kappa。
"""
import argparse, json, re, sys, os, glob, csv

# lm_eval gsm8k filter 正则（逐字复制自 site-packages/.../tasks/gsm8k/gsm8k.yaml）
STRICT_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
FLEX_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def flex_extract(text):
    """复刻 lm_eval flexible-extract：取最后一个匹配（group_select=-1）。"""
    if not text:
        return None
    matches = list(FLEX_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return m.group(1) or m.group(2)


def strict_extract(text):
    """复刻 strict-match：第一个 #### N（take_first）。"""
    if not text:
        return None
    m = STRICT_RE.search(text)
    return m.group(1) if m else None


def norm_num(s):
    """归一到 float 做等价比较（去 $ , 尾随 .）。"""
    if s is None:
        return None
    s = str(s).strip().replace("$", "").replace(",", "").rstrip(".")
    if s in ("", "-", "."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def gsm8k_gold(answer_field):
    """GSM8K doc['answer'] 形如 '...\\n#### 42'；取 #### 后数字。"""
    if answer_field is None:
        return None
    m = STRICT_RE.search(str(answer_field))
    return m.group(1) if m else answer_field


# ---- 防御式字段提取（lm_eval 0.4.x samples schema 随版本变化，先 inspect 确认）----
def get_raw_gen(rec):
    """从样本记录里取模型原始生成文本（尝试多个可能键名）。"""
    for k in ("resps", "resp", "prediction", "pred", "filtered_resps", "arguments"):
        v = rec.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            if v and isinstance(v[0], list):
                if v[0] and isinstance(v[0][0], str):
                    return v[0][0]
            if v and isinstance(v[0], str):
                return v[0]
    return ""


def get_doc(rec):
    d = rec.get("doc", rec.get("document", {}))
    return d if isinstance(d, dict) else {}


def find_samples(path):
    if os.path.isfile(path):
        return [path]
    out = glob.glob(os.path.join(path, "**", "samples_*.jsonl"), recursive=True)
    if not out:
        out = glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)
    return sorted(out)


def cmd_inspect(args):
    with open(args.samples) as f:
        line = f.readline()
    if not line.strip():
        print("空文件", file=sys.stderr)
        sys.exit(2)
    rec = json.loads(line)
    print("=== 首条记录 keys ===")
    print(list(rec.keys()))
    print("=== 记录全文（截断 2000 字符）===")
    print(json.dumps(rec, ensure_ascii=False)[:2000])
    doc = get_doc(rec)
    print("=== 推断字段 ===")
    print("doc keys:", list(doc.keys()) if isinstance(doc, dict) else type(doc))
    raw = get_raw_gen(rec)
    print("raw_gen[:200]:", repr(raw[:200]))
    print("flex_extract:", repr(flex_extract(raw)))
    print("strict_extract:", repr(strict_extract(raw)))
    print("gold(answer_field):", repr(gsm8k_gold(doc.get("answer"))))


def cmd_worksheet(args):
    files = find_samples(args.samples)
    if not files:
        print(f"未找到 samples_*.jsonl（{args.samples} 下）。先带 --log_samples 重跑。", file=sys.stderr)
        sys.exit(2)
    rows = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                doc = get_doc(rec)
                raw = get_raw_gen(rec)
                gold = gsm8k_gold(doc.get("answer"))
                fx = flex_extract(raw)
                sx = strict_extract(raw)
                ng, nfx, nsx = norm_num(gold), norm_num(fx), norm_num(sx)
                rows.append({
                    "doc_id": rec.get("doc_id", ""),
                    "arm": args.arm,
                    "source_file": os.path.basename(fp),
                    "question": (doc.get("question", "") or "")[:500].replace("\n", " "),
                    "gold": gold,
                    "model_output": raw[:600].replace("\n", " ").replace("\t", " "),
                    "flex_extracted": fx,
                    "strict_extracted": sx,
                    "auto_flex_eq_gold": int(nfx is not None and ng is not None and nfx == ng),
                    "auto_strict_eq_gold": int(nsx is not None and ng is not None and nsx == ng),
                    "annotA_flex_captures_intent": "",
                    "annotB_flex_captures_intent": "",
                    "notes": "",
                })
    if not rows:
        print("samples 文件为空。", file=sys.stderr)
        sys.exit(2)
    cols = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n = len(rows)
    flex_acc = sum(r["auto_flex_eq_gold"] for r in rows) / n
    strict_acc = sum(r["auto_strict_eq_gold"] for r in rows) / n
    print(f"工作表写入 {args.out}：{n} 行（来自 {len(files)} 个 samples 文件）")
    print(f"[自动预览] flex==gold: {flex_acc:.4f}  strict==gold: {strict_acc:.4f}  (n={n})")
    print("人工双标注：填 annotA_flex_captures_intent 与 annotB_... 两列（Y/N/UNCLEAR：flex 是否捕获模型意图答案），再跑 kappa 模式。")


def cohen_kappa(a, b):
    assert len(a) == len(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = {c: a.count(c) / n for c in cats}
    cb = {c: b.count(c) / n for c in cats}
    pe = sum(ca[c] * cb[c] for c in cats)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def cmd_kappa(args):
    rows = []
    with open(args.annotated) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append(row)
    A = [row.get("annotA_flex_captures_intent", "").strip().upper() for row in rows]
    B = [row.get("annotB_flex_captures_intent", "").strip().upper() for row in rows]
    paired = [(a, b) for a, b in zip(A, B) if a and b]
    if len(paired) < 2:
        print(f"双标注行不足（{len(paired)}）；先填 annotA/annotB 两列。", file=sys.stderr)
        sys.exit(2)
    a2, b2 = zip(*paired)
    k = cohen_kappa(list(a2), list(b2))
    n = len(paired)
    agree = sum(1 for x, y in zip(a2, b2) if x == y) / n
    yes_rate = sum(1 for x in a2 if x == "Y") / n
    auto_flex = sum(int(row.get("auto_flex_eq_gold", 0) or 0) for row in rows) / max(1, len(rows))
    verdict = "PASS" if k >= 0.95 else "FAIL"
    print("=== B8 flex 抽取器审计结果 ===")
    print(f"双标注行数: {n}")
    print(f"Cohen's κ (A vs B): {k:.4f}  [{verdict}，阈值 0.95]")
    print(f"原始一致率 agreement: {agree:.4f}")
    print(f"标注员A 判 'flex 捕获模型意图' 比例: {yes_rate:.4f}")
    print(f"自动 flex==gold 准确率: {auto_flex:.4f}")
    print(f"判定: {'κ≥0.95，flex 抽取可审计可靠' if k >= 0.95 else 'κ<0.95，标注任务定义模糊或 flex 抽取不稳，需排查'}")
    print(f"提醒: flex==gold({auto_flex:.3f}) 是「答对率」；κ 是「可标注性」；两者都报，勿混。")


def main():
    ap = argparse.ArgumentParser(description="B8 GSM8K flex 抽取器审计工具")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("inspect", help="打印首条样本记录键+样例")
    p1.add_argument("samples")
    p1.set_defaults(fn=cmd_inspect)
    p2 = sub.add_parser("worksheet", help="生成双标注工作表")
    p2.add_argument("samples")
    p2.add_argument("--arm", default="adaq")
    p2.add_argument("--out", required=True)
    p2.set_defaults(fn=cmd_worksheet)
    p3 = sub.add_parser("kappa", help="从已标注 TSV 算 κ")
    p3.add_argument("annotated")
    p3.set_defaults(fn=cmd_kappa)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
