#!/usr/bin/env bash
# chain_f18_rtn.sh — F18：INT4-RTN-g128 PTQ 基线评测（等 GPU 释放后跑 mmlu+gsm8k）。
#
# 量化 round-trip 由 run_f18_rtn_baseline.py --quantize 在 CPU 完成（无 GPU 争用），本脚本只评测。
# RTN 模型 = 对称 INT4-g128 round-trip 后的普通 bf16 模型 → 当 pretrained 评测（无 peft、无 load_in_4bit），
# 测的是 INT4-RTN-g128 量化误差对精度的影响 = 真 PTQ 锚点（对照 base_nf4dq_noadapter / baseline_bf16）。
#
# gate：等 "[bf16-baseline] DONE"（GPU0 释放）；用 GPU0。
# 启动：nohup bash scripts/chain_f18_rtn.sh >> logs/f18_rtn_eval.log 2>&1 &
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
LOCK=/tmp/adaq_f18_rtn.lock
exec 9>"$LOCK"; flock 9
BASE=/home/ubuntu/AdaQ-LoRA
RTN_MODEL=$BASE/models/Meta-Llama-3-8B-int4rtn-g128
OUT=$BASE/results_v2_rtn/base_int4rtn_g128
BF16LOG=$BASE/logs/baseline_bf16.log
LOG=$BASE/logs/f18_rtn_eval.log
mkdir -p "$OUT" "$BASE/logs"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 NLTK_DATA=/home/ubuntu/nltk_data HF_ENDPOINT=https://hf-mirror.com

# 等 bf16 完成（GPU0 释放）
while ! grep -q "\[bf16-baseline\] DONE" "$BF16LOG" 2>/dev/null; do
  echo "[f18-eval] $(date -u '+%F %T') waiting for bf16-baseline DONE ..."; sleep 300
done

# 确保 RTN 模型已量化（CPU 量化应已完成；幂等兜底）
if [ ! -f "$RTN_MODEL/config.json" ]; then
  echo "[f18-eval] $(date -u '+%F %T') RTN 模型缺失，先跑 CPU 量化 round-trip"
  $PY scripts/run_f18_rtn_baseline.py --quantize --verify >> "$LOG" 2>&1
fi
# provenance 拷进 results（models/ 被 gitignore，provenance 须随 results 入库）
cp -f "$RTN_MODEL/rtn_provenance.json" "$OUT/rtn_provenance.json" 2>/dev/null || true

echo "[f18-eval] $(date -u '+%F %T') start INT4-RTN-g128 eval (mmlu 5shot + gsm8k 4shot, GPU0)"
for TASKSPEC in "mmlu:5" "gsm8k:4"; do
  T=${TASKSPEC%%:*}; N=${TASKSPEC##*:}
  if [ -f "$OUT/.done_${T}" ]; then echo "[f18-eval] $T done, skip"; continue; fi
  echo "[f18-eval] $(date -u '+%F %T') START $T (${N}-shot)"
  CUDA_VISIBLE_DEVICES=0 $PY -m lm_eval --model hf \
    --model_args pretrained=$RTN_MODEL,dtype=bfloat16 \
    --tasks $T --num_fewshot $N --batch_size auto --device cuda:0 --seed 42 \
    --output_path "$OUT/$T" >> "$LOG" 2>&1
  rc=$?
  nj=$(find "$OUT/$T" -name 'results_*.json' 2>/dev/null | wc -l)
  if [ "$rc" = 0 ] && [ "$nj" -ge 1 ]; then touch "$OUT/.done_${T}"; echo "[f18-eval] $(date -u '+%F %T') OK $T rc=0 results=$nj"
  else echo "[f18-eval] $(date -u '+%F %T') FAIL $T rc=$rc results=$nj（不 touch .done）"; fi
done

# 刷台账 + 提交（只 results，不含模型权重）
$PY scripts/build_results_ledger.py > /dev/null 2>&1 || true
git add results_v2_rtn results_ledger.csv 2>/dev/null
git commit -q -m "F18 INT4-RTN-g128 PTQ 基线评测结果（results_v2_rtn/base_int4rtn_g128，mmlu+gsm8k）" 2>/dev/null && echo "[f18-eval] committed $(git rev-parse --short HEAD)"
echo "[f18-rtn] DONE  (results: $OUT)"
