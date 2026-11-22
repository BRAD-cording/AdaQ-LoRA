#!/bin/bash
# eval_mmlu_b4.sh -- MMLU 5-shot for the INT4-RTN seeds, at batch 4.
# batch 16 OOMs on loglikelihood (56168 requests, 22.4 GiB demanded); batch 4 fits.
# usage: bash scripts/eval_mmlu_b4.sh <gpu> <method> <seed>...
set -uo pipefail
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
MODEL=models/Meta-Llama-3-8B-int4rtn-g128
OUTROOT=results_v2_rtn_b16
GPU=${1:?gpu}
METHOD=${2:?method}
shift 2
LOG="logs/rtn_mmlu_b4_gpu${GPU}_${METHOD}.log"

echo "[mmlu] $(date -u +%FT%T) START gpu=${GPU} method=${METHOD} seeds=[$*]" | tee -a "$LOG"

for s in "$@"; do
  ADP="outputs/phase3_rtn/${METHOD}_s${s}"
  RES="${OUTROOT}/${METHOD}_s${s}/mmlu"
  if [ -f "${RES}/.done" ]; then
    echo "[mmlu] skip ${METHOD}_s${s}" | tee -a "$LOG"; continue
  fi
  if [ ! -f "${ADP}/adapter_model.safetensors" ]; then
    echo "[mmlu] NO-ADAPTER ${METHOD}_s${s}" | tee -a "$LOG"; continue
  fi
  mkdir -p "$RES"
  for bs in 4 2; do
    echo "[mmlu] $(date -u +%FT%T) EVAL ${METHOD}_s${s} bs=${bs}" | tee -a "$LOG"
    env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" -u -m lm_eval --model hf \
      --model_args "pretrained=${MODEL},peft=${ADP},trust_remote_code=True,dtype=bfloat16" \
      --tasks mmlu --batch_size "${bs}" --num_fewshot 5 \
      --output_path "$RES" >> "$LOG" 2>&1
    if find "$RES" -name 'results_*.json' | grep -q .; then
      touch "${RES}/.done"
      echo "[mmlu] $(date -u +%FT%T) OK ${METHOD}_s${s} bs=${bs}" | tee -a "$LOG"
      break
    fi
    tail -c 3000 "$LOG" | grep -q 'out of memory' || break
    echo "[mmlu] $(date -u +%FT%T) OOM bs=${bs}, retry smaller" | tee -a "$LOG"
  done
done
echo "[mmlu] $(date -u +%FT%T) ALL-DONE gpu=${GPU} method=${METHOD}" | tee -a "$LOG"
