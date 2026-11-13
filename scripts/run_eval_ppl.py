"""P2-C2: PPL 评测（wikitext word_perplexity），4 方法 BF16 + adaq/qlora NF4 口径。"""
import os, sys, subprocess, json
from concurrent.futures import ThreadPoolExecutor

PY = "/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python"
BASE = "/home/ubuntu/AdaQ-LoRA"
MODEL = os.path.join(BASE, "models/Meta-Llama-3-8B")
ADAPTER_DIR = os.path.join(BASE, "outputs/adapters")
OUT = os.environ.get("OUT_DIR", os.path.join(BASE, "results_v2_ppl"))
GPUS = [int(x) for x in os.environ.get("GPUS", "0,1").split(",")]
LIMIT = os.environ.get("LIMIT", "")
# 默认 4 方法 BF16（与主实验一致口径）+ adaq_nf4/qlora_nf4 量化口径
RUNS = os.environ.get("RUNS", "qlora,dora,adalora,adaq,qlora_nf4,adaq_nf4").split(",")

def run_one(run, gpu):
    out_dir = os.path.join(OUT, run)
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, ".done_ppl")
    if os.path.exists(done):
        print(f"SKIP {run}")
        return (run, "skip")
    nf4 = "_nf4" in run
    base_method = run.replace("_nf4", "")
    args = f"pretrained={MODEL},dtype=bfloat16"
    if nf4:
        args = f"pretrained={MODEL},load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_compute_dtype=bfloat16,bnb_4bit_use_double_quant=True,dtype=bfloat16"
    # 基线方法没有 adapter（qlora/dora/adalora/adaq 才有）
    adapter = os.path.join(ADAPTER_DIR, f"{base_method}_seed42")
    if os.path.exists(adapter):
        args += f",peft={adapter}"
    cmd = [PY, "-m", "lm_eval", "--model", "hf", "--model_args", args,
           "--tasks", "wikitext", "--batch_size", "auto",
           "--device", f"cuda:{gpu}", "--seed", "42", "--output_path", out_dir]
    if LIMIT:
        cmd += ["--limit", LIMIT]
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    p = subprocess.run(cmd, env=env, cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open(os.path.join(out_dir, "eval_ppl.log"), "wb") as f:
        f.write(p.stdout)
    if p.returncode == 0:
        open(done, "w").write("ok\n")
        print(f"DONE {run} rc=0")
        return (run, "ok")
    print(f"FAIL {run} rc={p.returncode}")
    return (run, "fail")

def main():
    jobs = [(r, GPUS[i % len(GPUS)]) for i, r in enumerate(RUNS)]
    print(f"jobs={len(jobs)} runs={RUNS}")
    with ThreadPoolExecutor(max_workers=len(GPUS)) as ex:
        results = list(ex.map(lambda j: run_one(*j), jobs))
    ok = sum(1 for _, s in results if s == "ok")
    print(f"summary ok={ok}/{len(results)}")
    sys.exit(0 if ok == len(results) else 1)

if __name__ == "__main__":
    main()
