#!/usr/bin/env python3
"""补齐 OUT_DIR 中每个 run 缺失的评测任务。

背景：v2.0 的 main 任务集把 mmlu 与 gsm8k 写成同一个 marker=main，
第一个任务 touch .done_main 后第二个被 SKIP；且 mmlu 遇 hf-mirror 429
失败后由后续 gsm8k 的成功 touch 同一个 marker，把失败掩盖掉。结果是
12 个 run 没有任何一个同时拥有 MMLU + GSM8K。

本脚本：
  * 扫描每个 run 实际已有的 tasks，按 WANT 生成缺失清单（不认旧 .done_main）
  * 同一 run 的任务串行、不同 run 双卡并行（每张 GPU 一个串行队列）
  * 默认 HF_HUB_OFFLINE=1，绕开 hf-mirror 429（数据集已缓存）
  * 每个任务跑完用 verify_eval_artifact.py 机读校验产物，才 touch .done_<marker>
  * 全部成功后自动调用 aggregate_results.py 落最终 summary.json

环境变量:
  OUT_DIR   默认 <root>/results_v2
  BASE_MODEL 默认 <root>/models/Meta-Llama-3-8B
  GPUS      逗号分隔，默认 "0,1"
  WANT      逗号分隔，默认 "mmlu,gsm8k,safety"
  RUNS      逗号分隔，只处理指定 run（冒烟用），默认全部
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
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "results_v2"))
BASE = os.environ.get("BASE_MODEL", os.path.join(ROOT, "models", "Meta-Llama-3-8B"))
GPUS = [g.strip() for g in os.environ.get("GPUS", "0,1").split(",") if g.strip()]
WANT = [w.strip() for w in os.environ.get("WANT", "mmlu,gsm8k,safety").split(",") if w.strip()]
RUNS = [r.strip() for r in os.environ.get("RUNS", "").split(",") if r.strip()]
LIMIT = os.environ.get("LIMIT", "")
FORCE = os.environ.get("FORCE", "0") == "1"
AGGREGATE = os.environ.get("AGGREGATE", "1") == "1"

# marker -> (lm_eval --tasks, num_fewshot, 必须出现在 results 里的键)
SPECS = {
    "mmlu":   ("mmlu", 5, ("mmlu",)),
    "gsm8k":  ("gsm8k", 4, ("gsm8k",)),
    "safety": ("truthfulqa_mc1,truthfulqa_mc2,ifeval", 0,
               ("truthfulqa_mc1", "truthfulqa_mc2", "ifeval")),
}
# 长任务先排，减少双卡尾部空等
ORDER = {"mmlu": 0, "safety": 1, "gsm8k": 2}


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
    run_dirs = sorted(glob.glob(os.path.join(OUT, "*_seed*")))
    if RUNS:
        run_dirs = [os.path.join(OUT, r) for r in RUNS]
    jobs = []
    for run_dir in run_dirs:
        run = os.path.basename(run_dir)
        os.makedirs(run_dir, exist_ok=True)
        parts = run.rsplit("_seed", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        method, seed = parts[0], parts[1]
        adapter = os.path.join(ROOT, "outputs", "adapters", run)
        if not os.path.isfile(os.path.join(adapter, "adapter_config.json")):
            print(f"[{now()}] [SKIP] {run}: adapter missing", flush=True)
            continue
        have = tasks_in(run_dir)
        for marker in WANT:
            if marker not in SPECS:
                continue
            need = SPECS[marker][2]
            done_file = os.path.join(run_dir, f".done_{marker}")
            if not FORCE and os.path.isfile(done_file):
                continue
            if not FORCE and all(t in have for t in need):
                open(done_file, "a").close()  # 已有结果，补写 marker，不重跑
                continue
            tasks, shots, _ = SPECS[marker]
            jobs.append(dict(run=run, method=method, seed=seed, marker=marker,
                             tasks=tasks, shots=shots, run_dir=run_dir, adapter=adapter))
    return jobs


def run_one(job, gpu):
    tasks, shots, marker = job["tasks"], job["shots"], job["marker"]
    run_dir, run = job["run_dir"], job["run"]
    log = os.path.join(run_dir, f"eval_{marker}.log")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "TOKENIZERS_PARALLELISM": "false",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "ADAQ_MODEL_PATH": BASE,
        # nltk 3.10.4 只信任 go-w 的私有目录；显式固定，避免子进程 HOME 差异导致找不到 punkt_tab
        "NLTK_DATA": os.environ.get("NLTK_DATA", "/home/ubuntu/nltk_data"),
    })
    cmd = [
        PY, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={BASE},peft={job['adapter']},dtype=bfloat16",
        "--tasks", tasks,
        "--num_fewshot", str(shots),
        "--batch_size", "auto",
        "--device", "cuda:0",
        "--seed", job["seed"],
        "--output_path", run_dir,
    ]
    if LIMIT:
        cmd += ["--limit", LIMIT]
    t0 = time.time()
    print(f"[{now()}] START gpu{gpu} {run}/{marker} limit={LIMIT or 'full'}", flush=True)
    with open(log, "a") as fh:
        fh.write(f"\n===== REPAIR gpu{gpu} {now()} tasks={tasks} limit={LIMIT or 'full'} =====\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    dt = round(time.time() - t0, 1)
    if rc != 0:
        print(f"[{now()}] FAIL  gpu{gpu} {run}/{marker} rc={rc} {dt}s -> {log}", flush=True)
        return (job, False, f"rc={rc}")
    v = subprocess.call(
        [PY, os.path.join(ROOT, "scripts", "verify_eval_artifact.py"), run_dir, marker]
    )
    if v != 0:
        print(f"[{now()}] FAIL  gpu{gpu} {run}/{marker} artifact-missing {dt}s -> {log}", flush=True)
        return (job, False, "artifact-missing")
    open(os.path.join(run_dir, f".done_{marker}"), "a").close()
    print(f"[{now()}] OK    gpu{gpu} {run}/{marker} {dt}s", flush=True)
    return (job, True, "")


def worker(gpu, queue):
    return [run_one(j, gpu) for j in queue]


def main():
    if not os.path.isdir(BASE):
        print(f"[FATAL] base model not found: {BASE}", file=sys.stderr)
        return 1
    if "safety" in WANT:
        print(f"[{now()}] preflight: safety metric path (GPU-free) ...", flush=True)
        pf_env = os.environ.copy()
        pf_env.update({
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "NLTK_DATA": os.environ.get("NLTK_DATA", "/home/ubuntu/nltk_data"),
        })
        rc_pf = subprocess.call(
            [PY, os.path.join(ROOT, "scripts", "preflight_safety_metrics.py")], env=pf_env
        )
        if rc_pf != 0:
            print(
                f"[{now()}] preflight FAILED rc={rc_pf} -> refuse to launch safety GPU jobs "
                f"(metric path is broken; fix it instead of burning hours)",
                file=sys.stderr, flush=True,
            )
            return 3
        print(f"[{now()}] preflight PASS", flush=True)

    jobs = build_jobs()
    print(f"[{now()}] OUT={OUT} WANT={WANT} GPUS={GPUS} LIMIT={LIMIT or 'full'} jobs={len(jobs)}", flush=True)
    for j in jobs:
        print(f"   {j['run']:22s} {j['marker']}", flush=True)
    if not jobs:
        print("nothing to do", flush=True)
    else:
        # 同一 run 的任务必须在同一队列内串行（避免两个 lm_eval 同写 output_path）
        byrun = defaultdict(list)
        for j in jobs:
            byrun[j["run"]].append(j)
        for r in byrun:
            byrun[r].sort(key=lambda j: ORDER.get(j["marker"], 9))
        groups = sorted(
            byrun.items(),
            key=lambda kv: (min(ORDER.get(j["marker"], 9) for j in kv[1]), -len(kv[1]), kv[0]),
        )
        queues = {g: [] for g in GPUS}
        for i, (_r, js) in enumerate(groups):
            queues[GPUS[i % len(GPUS)]].extend(js)
        results = []
        with ThreadPoolExecutor(max_workers=len(GPUS)) as ex:
            futs = [ex.submit(worker, g, q) for g, q in queues.items()]
            for f in as_completed(futs):
                results.extend(f.result())
        bad = [(j["run"], j["marker"], msg) for j, ok, msg in results if not ok]
        print(f"[{now()}] DONE ok={len(results) - len(bad)} fail={len(bad)}", flush=True)
        for r, m, msg in bad:
            print(f"   FAILED {r}/{m}: {msg}", flush=True)
        if bad:
            return 2
    if AGGREGATE:
        agg_out = os.path.join(OUT, "summary.json")
        print(f"[{now()}] aggregate -> {agg_out}", flush=True)
        rc = subprocess.call(
            [PY, os.path.join(ROOT, "scripts", "aggregate_results.py"),
             "--results-dir", OUT, "--out", agg_out]
        )
        print(f"[{now()}] aggregate exit={rc}", flush=True)
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
