#!/usr/bin/env python3
# [2026-11-16 评审裁定 F1] 命名更正：本脚本的 baseline_nf4 = bnb NF4+double-quant 无校准量化（无 adapter），
# 不是 PTQ 文献的 RTN(INT4,g=128)。产物键名保留 baseline_nf4 以兼容既有 summary.json，
# 一切文档/论文表述必须写 'NF4+DQ base, no adapter'；真 INT4-RTN 参照臂见 review/fair-baseline.md F18。
"""P0-B1: NF4 加载口径下的 PTQ 基线对比评测。

背景：主实验评测用 BF16 全精度（dtype=bfloat16 无 load_in_4bit），
但 AdaQ 的「量化感知」发生在训练时（train.py load_in_4bit + nf4 + double_quant）。
本脚本把评测也切到 NF4 加载口径，回答「量化下 AdaQ 秩分配是否补偿量化误差」。

对比矩阵（同 load_in_4bit + nf4 + double_quant + bf16 compute 口径，单种子 seed42）：
  baseline_nf4  : 纯 NF4 基座，零微调（无 LoRA）
  qlora_nf4     : NF4 + qlora_seed42 adapter
  dora_nf4      : NF4 + dora_seed42 adapter
  adalora_nf4   : NF4 + adalora_seed42 adapter
  adaq_nf4      : NF4 + adaq_seed42 adapter

任务：mmlu (5-shot) + gsm8k (4-shot)（主任务；safety 留待 P1）

环境变量:
  OUT_DIR   默认 <root>/results_v2_ptq_nf4
  BASE_MODEL 默认 <root>/models/Meta-Llama-3-8B
  GPUS      逗号分隔，默认 "0,1"
  RUNS      逗号分隔，只处理指定 run（冒烟用），默认全部 5 项
  LIMIT     每任务样本上限，留空=全量（冒烟可设 8）
  FORCE     1=忽略已有结果与 marker 强制重跑
  AGGREGATE 1=全部成功后自动落 summary.json（默认 1）
"""
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.environ.get("PY", "/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python")
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "results_v2_ptq_nf4"))
BASE = os.environ.get("BASE_MODEL", os.path.join(ROOT, "models", "Meta-Llama-3-8B"))
GPUS = [g.strip() for g in os.environ.get("GPUS", "0,1").split(",") if g.strip()]
RUNS = [r.strip() for r in os.environ.get("RUNS", "").split(",") if r.strip()]
LIMIT = os.environ.get("LIMIT", "")
FORCE = os.environ.get("FORCE", "0") == "1"
AGGREGATE = os.environ.get("AGGREGATE", "1") == "1"

# 5 项对比：name -> (adapter 子路径 或 None=无 LoRA 基线, seed)
# 注意 seed42 用于 seed 字段（评测随机种子），adapter 也是 seed42 训练产物
CONFIGS = {
    "baseline_nf4": None,          # 无 LoRA，纯 NF4 基座
    "qlora_nf4":    "qlora_seed42",
    "dora_nf4":     "dora_seed42",
    "adalora_nf4":  "adalora_seed42",
    "adaq_nf4":     "adaq_seed42",
}

SPECS = {
    "mmlu":  ("mmlu", 5, ("mmlu",)),
    "gsm8k": ("gsm8k", 4, ("gsm8k",)),
}
ORDER = {"mmlu": 0, "gsm8k": 1}


def now():
    return datetime.now().strftime("%F %T")


def tasks_in(run_dir):
    have = set()
    for f in glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True):
        try:
            with open(f) as fh:
                have |= set((json.load(fh).get("results") or {}).keys())
        except (OSError, json.JSONDecodeError):
            pass
    return have


def build_jobs():
    names = RUNS if RUNS else list(CONFIGS.keys())
    jobs = []
    for name in names:
        if name not in CONFIGS:
            print(f"[{now()}] [SKIP] unknown config: {name}", flush=True)
            continue
        adapter_rel = CONFIGS[name]
        run_dir = os.path.join(OUT, name)
        os.makedirs(run_dir, exist_ok=True)
        if adapter_rel:
            adapter = os.path.join(ROOT, "outputs", "adapters", adapter_rel)
            if not os.path.isfile(os.path.join(adapter, "adapter_config.json")):
                print(f"[{now()}] [SKIP] {name}: adapter missing {adapter}", flush=True)
                continue
        else:
            adapter = None
        have = tasks_in(run_dir)
        for marker in SPECS:
            need = SPECS[marker][2]
            done_file = os.path.join(run_dir, f".done_{marker}")
            if not FORCE and os.path.isfile(done_file):
                continue
            if not FORCE and all(t in have for t in need):
                open(done_file, "a").close()
                continue
            tasks, shots, _ = SPECS[marker]
            jobs.append(dict(name=name, adapter=adapter, marker=marker,
                             tasks=tasks, shots=shots, run_dir=run_dir))
    return jobs


def model_args_str(adapter):
    # 统一 NF4 加载口径（与 train.py 的量化配置一致）
    base = (
        f"pretrained={BASE},"
        f"load_in_4bit=True,"
        f"bnb_4bit_quant_type=nf4,"
        f"bnb_4bit_compute_dtype=bfloat16,"
        f"bnb_4bit_use_double_quant=True,"
        f"dtype=bfloat16"
    )
    if adapter:
        base += f",peft={adapter}"
    return base


def run_one(job, gpu):
    tasks, shots, marker = job["tasks"], job["shots"], job["marker"]
    run_dir, name = job["run_dir"], job["name"]
    log = os.path.join(run_dir, f"eval_{marker}.log")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "TOKENIZERS_PARALLELISM": "false",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "NLTK_DATA": os.environ.get("NLTK_DATA", "/home/ubuntu/nltk_data"),
    })
    cmd = [
        PY, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args_str(job["adapter"]),
        "--tasks", tasks,
        "--num_fewshot", str(shots),
        "--batch_size", "auto",
        "--device", "cuda:0",
        "--seed", "42",
        "--output_path", run_dir,
    ]
    if LIMIT:
        cmd += ["--limit", LIMIT]
    t0 = time.time()
    print(f"[{now()}] START gpu{gpu} {name}/{marker} limit={LIMIT or 'full'}", flush=True)
    with open(log, "a") as fh:
        fh.write(f"\n===== PTQ-NF4 gpu{gpu} {now()} tasks={tasks} limit={LIMIT or 'full'} =====\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    dt = round(time.time() - t0, 1)
    if rc != 0:
        print(f"[{now()}] FAIL  gpu{gpu} {name}/{marker} rc={rc} {dt}s -> {log}", flush=True)
        return (job, False, f"rc={rc}")
    v = subprocess.call(
        [PY, os.path.join(ROOT, "scripts", "verify_eval_artifact.py"), run_dir, marker]
    )
    if v != 0:
        print(f"[{now()}] FAIL  gpu{gpu} {name}/{marker} artifact-missing {dt}s -> {log}", flush=True)
        return (job, False, "artifact-missing")
    open(os.path.join(run_dir, f".done_{marker}"), "a").close()
    print(f"[{now()}] OK    gpu{gpu} {name}/{marker} {dt}s", flush=True)
    return (job, True, "")


def worker(gpu, queue):
    return [run_one(j, gpu) for j in queue]


def aggregate():
    # 自定义聚合：读 5 项 × 2 任务的结果，产 PTQ-NF4 对比表
    rows = {}
    for name in CONFIGS:
        run_dir = os.path.join(OUT, name)
        mmlu = gsm8k = None
        for f in sorted(glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True)):
            try:
                res = json.load(open(f)).get("results") or {}
            except (OSError, json.JSONDecodeError):
                continue
            if "mmlu" in res and len(res) > 10:
                mmlu = res["mmlu"].get("acc,none")
            if "gsm8k" in res:
                gsm8k = res["gsm8k"].get("exact_match,strict-match")
        rows[name] = {"mmlu": mmlu, "gsm8k": gsm8k}
    out = os.path.join(OUT, "summary.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"[{now()}] summary -> {out}", flush=True)
    print(f"{'name':<14}{'MMLU':>10}{'GSM8K':>10}", flush=True)
    for name in CONFIGS:
        m = rows[name]["mmlu"]
        g = rows[name]["gsm8k"]
        ms = f"{m:.4f}" if m is not None else "  N/A"
        gs = f"{g:.4f}" if g is not None else "  N/A"
        print(f"{name:<14}{ms:>10}{gs:>10}", flush=True)


def main():
    if not os.path.isdir(BASE):
        print(f"[FATAL] base model not found: {BASE}", file=sys.stderr)
        return 1
    jobs = build_jobs()
    print(f"[{now()}] OUT={OUT} RUNS={RUNS or 'all'} GPUS={GPUS} LIMIT={LIMIT or 'full'} jobs={len(jobs)}", flush=True)
    for j in jobs:
        print(f"   {j['name']:14s} {j['marker']}", flush=True)
    if not jobs:
        print("nothing to do", flush=True)
    else:
        byrun = defaultdict(list)
        for j in jobs:
            byrun[j["name"]].append(j)
        for r in byrun:
            byrun[r].sort(key=lambda j: ORDER.get(j["marker"], 9))
        groups = sorted(
            byrun.items(),
            key=lambda kv: (min(ORDER.get(j["marker"], 9) for j in kv[1]), kv[0]),
        )
        queues = {g: [] for g in GPUS}
        for i, (_r, js) in enumerate(groups):
            queues[GPUS[i % len(GPUS)]].extend(js)
        results = []
        with ThreadPoolExecutor(max_workers=len(GPUS)) as ex:
            futs = [ex.submit(worker, g, q) for g, q in queues.items()]
            for f in as_completed(futs):
                results.extend(f.result())
        bad = [(j["name"], j["marker"], msg) for j, ok, msg in results if not ok]
        print(f"[{now()}] DONE ok={len(results)-len(bad)} fail={len(bad)}", flush=True)
        for n, m, msg in bad:
            print(f"   FAILED {n}/{m}: {msg}", flush=True)
        if bad:
            return 2
    if AGGREGATE:
        aggregate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
