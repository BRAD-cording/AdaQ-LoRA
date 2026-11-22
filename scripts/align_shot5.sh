#!/bin/bash
# align_shot5.sh -- re-score GSM8K at 5-shot for every arm still on 4-shot.
#
# v2: the output directory carries a slug of the adapter root, because NF4 and
# INT8 both have runs named <method>_s<seed>; without it the two trees collide.
#
# usage: bash scripts/align_shot5.sh <gpu> <adapter_root> <method> <seed>...
set -uo pipefail
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
GPU=${1:?gpu}
ROOT=${2:?adapter root}
METHOD=${3:?method}
shift 3
SLUG=$(echo "$ROOT" | sed -e 's|outputs/||' -e 's|/|_|g')
LOG="logs/align5_gpu${GPU}_${METHOD}.log"
mkdir -p logs

for s in "$@"; do
  ADP="${ROOT}/${METHOD}_s${s}"
  RES="results_shot5/${SLUG}_${METHOD}_s${s}"
  if [ -f "${RES}/.done_gsm8k5" ]; then
    echo "[align5] $(date -u +%FT%T) SKIP ${METHOD}_s${s}" | tee -a "$LOG"; continue
  fi
  if [ ! -f "${ADP}/adapter_model.safetensors" ]; then
    echo "[align5] $(date -u +%FT%T) NO-ADAPTER ${ADP}" | tee -a "$LOG"; continue
  fi
  mkdir -p "$RES"
  echo "[align5] $(date -u +%FT%T) EVAL gsm8k-5shot ${SLUG}/${METHOD}_s${s}" | tee -a "$LOG"
  env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" -u -m lm_eval --model hf \
    --model_args "pretrained=models/Meta-Llama-3-8B,peft=${ADP},trust_remote_code=True,dtype=bfloat16" \
    --tasks gsm8k --batch_size 16 --num_fewshot 5 \
    --output_path "$RES" >> "$LOG" 2>&1
  if find "$RES" -name 'results_*.json' | grep -q .; then
    touch "${RES}/.done_gsm8k5"
    echo "[align5] $(date -u +%FT%T) OK ${SLUG}/${METHOD}_s${s}" | tee -a "$LOG"
  else
    echo "[align5] $(date -u +%FT%T) FAIL ${SLUG}/${METHOD}_s${s}" | tee -a "$LOG"
  fi
  sleep 3
done
echo "[align5] $(date -u +%FT%T) ALL-DONE gpu=${GPU} method=${METHOD}" | tee -a "$LOG"
