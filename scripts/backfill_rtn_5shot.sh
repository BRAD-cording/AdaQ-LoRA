#!/bin/bash
# backfill_rtn_5shot.sh -- re-evaluate the INT4-RTN runs that were scored with the
# wrong few-shot setting, so that every seed shares one protocol.
#
# Finding that motivates this: results_v2_rtn/*/ results show num_fewshot=4 for
# seeds 42 and 3407 but num_fewshot=5 for seed 2025 (and only some runs have MMLU).
# Mixing 4-shot and 5-shot across paired seeds inflates the paired variance.
#
# This script:
#   1. waits until both chain_rtn_power2.sh jobs have finished and a GPU has >=12 GiB free
#   2. moves each legacy result dir aside (kept, never deleted)
#   3. re-evaluates gsm8k + mmlu at 5-shot for the four affected runs
#   4. writes .done_both markers
# usage: bash scripts/backfill_rtn_5shot.sh <gpu_id> <method> <seed> [<seed> ...]
set -uo pipefail

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
MODEL=models/Meta-Llama-3-8B-int4rtn-g128
GPU=${1:?gpu id}
METHOD=${2:?method}
shift 2
SEEDS="$*"
LOG="logs/rtn_5shot_backfill_gpu${GPU}_${METHOD}.log"
mkdir -p logs

echo "[backfill] $(date -u +%FT%T) START gpu=${GPU} method=${METHOD} seeds=[${SEEDS}]" | tee -a "$LOG"

# 1. wait for the training chains to drain, then for free VRAM
for _ in $(seq 1 240); do
  if pgrep -f 'chain_rtn_powe[r]2' >/dev/null 2>&1; then
    sleep 300
    continue
  fi
  FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | awk -F', ' -v g="${GPU}" '$1==g{print $2}')
  if [ -n "${FREE:-}" ] && [ "${FREE}" -ge 12000 ]; then
    echo "[backfill] $(date -u +%FT%T) chains drained, gpu${GPU} free=${FREE}MiB -> go" | tee -a "$LOG"
    break
  fi
  echo "[backfill] $(date -u +%FT%T) waiting gpu${GPU} free=${FREE:-?}MiB" | tee -a "$LOG"
  sleep 300
done

for s in $SEEDS; do
  OUT="outputs/phase3_rtn/${METHOD}_s${s}"
  RES="results_v2_rtn/${METHOD}_s${s}"

  if [ -f "${RES}/.done_both_5shot" ]; then
    echo "[backfill] $(date -u +%FT%T) SKIP ${METHOD}_s${s} (already 5-shot)" | tee -a "$LOG"
    continue
  fi
  if [ ! -f "${OUT}/adapter_model.safetensors" ]; then
    echo "[backfill] $(date -u +%FT%T) NO-ADAPTER ${METHOD}_s${s}" | tee -a "$LOG"
    continue
  fi

  # keep the legacy (mixed few-shot) results as evidence
  if [ -d "$RES" ] && [ "$(ls -A "$RES" 2>/dev/null)" ]; then
    LEGACY="${RES}_legacy_mixedfewshot_$(date -u +%Y%m%dT%H%M%S)"
    mv "$RES" "$LEGACY" && echo "[backfill] $(date -u +%FT%T) moved legacy -> ${LEGACY}" | tee -a "$LOG"
  fi
  mkdir -p "$RES"

  echo "[backfill] $(date -u +%FT%T) EVAL ${METHOD}_s${s} (mmlu+gsm8k 5-shot)" | tee -a "$LOG"
  env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" -u -m lm_eval --model hf \
    --model_args "pretrained=${MODEL},peft=${OUT},trust_remote_code=True,dtype=bfloat16" \
    --tasks mmlu,gsm8k \
    --batch_size auto \
    --output_path "$RES" \
    --num_fewshot 5 \
    >> "$LOG" 2>&1

  NJ=$(find "$RES" -name 'results_*.json' 2>/dev/null | wc -l)
  if [ "$NJ" -ge 1 ]; then
    touch "${RES}/.done_both_5shot"
    echo "[backfill] $(date -u +%FT%T) EVAL-OK ${METHOD}_s${s} results=${NJ}" | tee -a "$LOG"
  else
    echo "[backfill] $(date -u +%FT%T) EVAL-FAIL ${METHOD}_s${s}" | tee -a "$LOG"
  fi
  sleep 5
done

echo "[backfill] $(date -u +%FT%T) ALL-DONE gpu=${GPU} method=${METHOD}" | tee -a "$LOG"
