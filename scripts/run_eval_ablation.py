#!/usr/bin/env python3
"""评测 6 个消融策略的 adapter（MMLU 5-shot + GSM8K 4-shot，全量）。

对标主实验 run_eval_missing.py 的口径（HF_HUB_OFFLINE=1, NLTK_DATA 固定, verify 产物）。
区别：adapter 路径是 outputs/ablation_adapters/<strategy>（不是 outputs/adapters/<run>）。

环境变量:
  OUT_DIR    默认 <root>/results_v2_ablation
  STRATEGIES 逗号分隔，默认 "error_only,importance_only,product,uniform,equal_weight,random"
  GPUS       逗号分隔，默认 "0,1"
  LIMIT      每任务样本上限，留空=全量
  FORCE      1=重跑
"""
import glob, json, os, subprocess, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.environ.get("PY", "/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python")
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "results_v2_ablation"))
BASE = os.environ.get("BASE_MODEL", os.path.join(ROOT, "models", "Meta-Llama-3-8B"))
ADAPTER_DIR = os.path.join(ROOT, "outputs", "ablation_adapters")
GPUS = [g.strip() for g in os.environ.get("GPUS", "0,1").split(",") if g.strip()]
STRATEGIES = [s.strip() for s in os.environ.get("STRATEGIES", "error_only,importance_only,product,uniform,equal_weight,random").split(",") if s.strip()]
LIMIT = os.environ.get("LIMIT", "")
FORCE = os.environ.get("FORCE", "0") == "1"

SPECS = {
    "mmlu":  ("mmlu", 5, ("mmlu",)),
    "gsm8k": ("gsm8k", 4, ("gsm8k",)),
}
ORDER = {"mmlu": 0, "gsm8k": 1}

def now(): return datetime.now().strftime("%F %T")

def tasks_in(run_dir):
    have = set()
    for f in glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True):
        try:
            with open(f) as fh: have |= set((json.load(fh).get("results") or {}).keys())
        except (OSError, json.JSONDecodeError): pass
    return have

def build_jobs():
    jobs = []
    for s in STRATEGIES:
        adapter = os.path.join(ADAPTER_DIR, s)
        if not os.path.isfile(os.path.join(adapter, "adapter_config.json")):
            print(f"[{now()}] [SKIP] {s}: adapter missing", flush=True); continue
        run_dir = os.path.join(OUT, s)
        os.makedirs(run_dir, exist_ok=True)
        have = tasks_in(run_dir)
        for marker in SPECS:
            tasks, shots, need = SPECS[marker]
            done_file = os.path.join(run_dir, f".done_{marker}")
            if not FORCE and os.path.isfile(done_file): continue
            if not FORCE and all(t in have for t in need):
                open(done_file, "a").close(); continue
            jobs.append(dict(strategy=s, marker=marker, tasks=tasks, shots=shots, run_dir=run_dir, adapter=adapter))
    return jobs

def run_one(job, gpu):
    tasks, shots, marker = job["tasks"], job["shots"], job["marker"]
    run_dir, strategy = job["run_dir"], job["strategy"]
    log = os.path.join(run_dir, f"eval_{marker}.log")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu, "TOKENIZERS_PARALLELISM": "false",
        "HF_ENDPOINT": "https://hf-mirror.com", "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "ADAQ_MODEL_PATH": BASE,
        "NLTK_DATA": os.environ.get("NLTK_DATA", "/home/ubuntu/nltk_data"),
    })
    cmd = [PY, "-m", "lm_eval", "--model", "hf",
           "--model_args", f"pretrained={BASE},peft={job['adapter']},dtype=bfloat16",
           "--tasks", tasks, "--num_fewshot", str(shots), "--batch_size", "auto",
           "--device", "cuda:0", "--seed", "42", "--output_path", run_dir]
    if LIMIT: cmd += ["--limit", LIMIT]
    t0 = time.time()
    print(f"[{now()}] START gpu{gpu} {strategy}/{marker} limit={LIMIT or 'full'}", flush=True)
    with open(log, "a") as fh:
        fh.write(f"\n===== ABLATION gpu{gpu} {now()} tasks={tasks} limit={LIMIT or 'full'} =====\n"); fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    dt = round(time.time() - t0, 1)
    if rc != 0:
        print(f"[{now()}] FAIL  gpu{gpu} {strategy}/{marker} rc={rc} {dt}s -> {log}", flush=True)
        return (job, False, f"rc={rc}")
    v = subprocess.call([PY, os.path.join(ROOT, "scripts", "verify_eval_artifact.py"), run_dir, marker])
    if v != 0:
        print(f"[{now()}] FAIL  gpu{gpu} {strategy}/{marker} artifact-missing {dt}s", flush=True)
        return (job, False, "artifact-missing")
    open(os.path.join(run_dir, f".done_{marker}"), "a").close()
    print(f"[{now()}] OK    gpu{gpu} {strategy}/{marker} {dt}s", flush=True)
    return (job, True, "")

def main():
    if not os.path.isdir(BASE):
        print(f"[FATAL] base model not found: {BASE}", file=sys.stderr); return 1
    jobs = build_jobs()
    print(f"[{now()}] OUT={OUT} STRATEGIES={STRATEGIES} GPUS={GPUS} LIMIT={LIMIT or 'full'} jobs={len(jobs)}", flush=True)
    for j in jobs: print(f"   {j['strategy']:16s} {j['marker']}", flush=True)
    if not jobs:
        print("nothing to do", flush=True); return 0
    byrun = defaultdict(list)
    for j in jobs: byrun[j["strategy"]].append(j)
    for r in byrun: byrun[r].sort(key=lambda j: ORDER.get(j["marker"], 9))
    groups = sorted(byrun.items(), key=lambda kv: (min(ORDER.get(j["marker"],9) for j in kv[1]), -len(kv[1]), kv[0]))
    queues = {g: [] for g in GPUS}
    for i, (_r, js) in enumerate(groups):
        queues[GPUS[i % len(GPUS)]].extend(js)
    results = []
    with ThreadPoolExecutor(max_workers=len(GPUS)) as ex:
        futs = [ex.submit(lambda g,q: [run_one(j,g) for j in q], g, q) for g, q in queues.items()]
        for f in as_completed(futs): results.extend(f.result())
    bad = [(j["strategy"], j["marker"], msg) for j, ok, msg in results if not ok]
    print(f"[{now()}] DONE ok={len(results)-len(bad)} fail={len(bad)}", flush=True)
    for s, m, msg in bad: print(f"   FAILED {s}/{m}: {msg}", flush=True)
    return 2 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
