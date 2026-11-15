"""P1-B4 补齐种子评测：Qwen2.5-14B，adaq vs qlora，seed 2025/3407（NF4 评测，与 seed42 同口径）。"""
import os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor

PY = "/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python"
BASE = "/home/ubuntu/AdaQ-LoRA"
MODEL = os.path.join(BASE, "models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master")
ADAPTER_DIR = os.path.join(BASE, "outputs/adapters_qwen14b")
SEEDS = [s for s in os.environ.get("SEEDS", "2025 3407").split() if s]
GPUS = [int(x) for x in os.environ.get("GPUS", "0,1").split(",")]
LIMIT = os.environ.get("LIMIT", "")
RUNS = os.environ.get("RUNS", "adaq,qlora").split(",")
SPECS = [("mmlu", 5), ("gsm8k", 4)]
ORDER = {"mmlu": 0, "gsm8k": 1}


def out_root(seed):
    return os.path.join(BASE, f"results_v2_qwen14b_seed{seed}")


def run_one(seed, run, task, nshot, gpu):
    tag = f"s{seed}/{run}/{task}"
    out_dir = os.path.join(out_root(seed), run)
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, f".done_{task}")
    if os.path.exists(done):
        print(f"SKIP {tag} (done)")
        return (tag, "skip")
    adapter = os.path.join(ADAPTER_DIR, f"{run}_seed{seed}")
    if not (os.path.exists(os.path.join(adapter, "adapter_config.json"))
            and os.path.exists(os.path.join(adapter, "adapter_model.safetensors"))):
        print(f"MISS {tag} adapter={adapter}")
        return (tag, "miss")
    cmd = [PY, "-m", "lm_eval", "--model", "hf",
           "--model_args", f"pretrained={MODEL},load_in_4bit=True,bnb_4bit_quant_type=nf4,"
                           f"bnb_4bit_compute_dtype=bfloat16,bnb_4bit_use_double_quant=True,"
                           f"dtype=bfloat16,peft={adapter}",
           "--tasks", task, "--num_fewshot", str(nshot), "--batch_size", "auto",
           "--device", f"cuda:{gpu}", "--seed", "42", "--output_path", out_dir]
    if LIMIT:
        cmd += ["--limit", LIMIT]
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", NLTK_DATA="/home/ubuntu/nltk_data")
    p = subprocess.run(cmd, env=env, cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open(os.path.join(out_dir, f"eval_{task}.log"), "wb") as f:
        f.write(p.stdout)
    if p.returncode == 0:
        open(done, "w").write("ok\n")
        print(f"DONE {tag} rc=0")
        return (tag, "ok")
    print(f"FAIL {tag} rc={p.returncode}")
    return (tag, "fail")


def main():
    specs = [(s, r, t, n) for s in SEEDS for r in RUNS for t, n in SPECS]
    specs.sort(key=lambda x: (x[0], ORDER[x[2]]))
    jobs = [(s, r, t, n, GPUS[i % len(GPUS)]) for i, (s, r, t, n) in enumerate(specs)]
    print(f"jobs={len(jobs)} seeds={SEEDS} runs={RUNS} gpus={GPUS}")
    with ThreadPoolExecutor(max_workers=len(GPUS)) as ex:
        results = list(ex.map(lambda j: run_one(*j), jobs))
    ok = sum(1 for _, s in results if s == "ok")
    skip = sum(1 for _, s in results if s == "skip")
    print(f"summary ok={ok} skip={skip} total={len(results)}")
    sys.exit(0 if ok + skip == len(results) else 1)


if __name__ == "__main__":
    main()
