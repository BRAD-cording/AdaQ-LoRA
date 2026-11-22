#!/bin/bash
# chain_rtn_power.sh -- extend the INT4-RTN seed count for the GSM8K cell.
# Reuses the exact protocol of the existing 3 seeds (see outputs/phase3_rtn/*):
#   model  = models/Meta-Llama-3-8B-int4rtn-g128 (pre-quantized, --no-quant)
#   train  = OpenHermes-2.5 first 2000, 3 epochs, bs 1, grad-accum 16, lr 2e-4, cosine 0.03
#   eval   = lm_eval gsm8k, 5-shot, bf16, adapter via peft=
# Idempotent: skips any run whose adapter / .done_gsm8k already exists.
# usage: bash scripts/chain_rtn_power.sh <gpu_id> <method> <seed> [<seed> ...]
set -uo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
MODEL=models/Meta-Llama-3-8B-int4rtn-g128
GPU=${1:?gpu id}
METHOD=${2:?method}
shift 2
SEEDS="$*"
LOG="logs/rtn_power_gpu${GPU}_${METHOD}.log"
mkdir -p logs

echo "[rtn-power] $(date -u +%FT%T) START gpu=${GPU} method=${METHOD} seeds=[${SEEDS}]" | tee -a "$LOG"

for s in $SEEDS; do
  OUT="outputs/phase3_rtn/${METHOD}_s${s}"
  RES="results_v2_rtn/${METHOD}_s${s}"

  if [ -f "${RES}/.done_gsm8k" ]; then
    echo "[rtn-power] $(date -u +%FT%T) SKIP ${METHOD}_s${s} (eval already done)" | tee -a "$LOG"
    continue
  fi

  if [ ! -f "${OUT}/adapter_model.safetensors" ]; then
    mkdir -p "$OUT"
    echo "[rtn-power] $(date -u +%FT%T) TRAIN ${METHOD}_s${s}" | tee -a "$LOG"
    env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" scripts/train.py \
      --model "$MODEL" \
      --method "$METHOD" \
      --seed "$s" \
      --output-dir "$OUT" \
      --no-quant \
      --batch-size 1 \
      --grad-accum 16 \
      --epochs 3 \
      --dataset-size 2000 \
      --lr 2e-4 \
      >> "$LOG" 2>&1
    if [ ! -f "${OUT}/adapter_model.safetensors" ]; then
      echo "[rtn-power] $(date -u +%FT%T) TRAIN-FAIL ${METHOD}_s${s}" | tee -a "$LOG"
      continue
    fi
    echo "[rtn-power] $(date -u +%FT%T) TRAIN-OK ${METHOD}_s${s}" | tee -a "$LOG"
  fi

  mkdir -p "$RES"
  echo "[rtn-power] $(date -u +%FT%T) EVAL ${METHOD}_s${s}" | tee -a "$LOG"
  env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" -u -m lm_eval --model hf \
    --model_args "pretrained=${MODEL},peft=${OUT},trust_remote_code=True,dtype=bfloat16" \
    --tasks gsm8k \
    --batch_size auto \
    --output_path "$RES" \
    --num_fewshot 5 \
    >> "$LOG" 2>&1

  NJ=$(find "$RES" -name 'results_*.json' 2>/dev/null | wc -l)
  if [ "$NJ" -ge 1 ]; then
    touch "${RES}/.done_gsm8k"
    echo "[rtn-power] $(date -u +%FT%T) EVAL-OK ${METHOD}_s${s} results=${NJ}" | tee -a "$LOG"
  else
    echo "[rtn-power] $(date -u +%FT%T) EVAL-FAIL ${METHOD}_s${s} results=0" | tee -a "$LOG"
  fi

  sleep 3
done

echo "[rtn-power] $(date -u +%FT%T) ALL-DONE gpu=${GPU} method=${METHOD}" | tee -a "$LOG"
