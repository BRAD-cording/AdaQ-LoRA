#!/bin/bash
# ab_batch.sh -- stop the slow (batch=1) eval chains and measure eval throughput
# at a fixed larger batch size, using an adapter whose training already finished.
set -uo pipefail
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
MODEL=models/Meta-Llama-3-8B-int4rtn-g128
ADAPTER=outputs/phase3_rtn/adaq_s1

# --- stop the running chains (pattern built at runtime to avoid pkill self-match)
PATCHAIN='chain_rtn_power''2'
PATEVAL='lm_eva''l'
pkill -f "$PATCHAIN" && echo "[ab] killed chain" || echo "[ab] no chain"
pkill -f "$PATEVAL" && echo "[ab] killed lm_eval" || echo "[ab] no lm_eval"
sleep 15
echo "[ab] gpu state:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

run_probe() {
  local bs="$1" limit="$2"
  local out="/tmp/ab_bs${bs}"
  rm -rf "$out"; mkdir -p "$out"
  local t0 t1
  t0=$(date +%s)
  env CUDA_VISIBLE_DEVICES=0 "$PY" -m lm_eval --model hf \
    --model_args "pretrained=${MODEL},peft=${ADAPTER},trust_remote_code=True,dtype=bfloat16" \
    --tasks gsm8k --batch_size "$bs" --num_fewshot 5 --limit "$limit" \
    --output_path "$out" > "/tmp/ab_bs${bs}.log" 2>&1
  local rc=$?
  t1=$(date +%s)
  echo "[ab] batch=${bs} limit=${limit} rc=${rc} elapsed=$((t1 - t0))s"
  "$PY" - "$out" <<'PYEOF'
import json, glob, sys
fs = glob.glob(sys.argv[1] + '/**/results_*.json', recursive=True)
if fs:
    r = json.load(open(sorted(fs)[-1]))['results']['gsm8k']
    print('     strict=%.4f flex=%.4f' % (r['exact_match,strict-match'], r['exact_match,flexible-extract']))
else:
    print('     no results json')
PYEOF
}

echo "[ab] === probe batch=16, limit=48 ==="
run_probe 16 48
echo "[ab] === probe batch=64, limit=48 ==="
run_probe 64 48
