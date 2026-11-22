#!/bin/bash
# chain_rtn_power2.sh -- extend INT4-RTN seed count for the GSM8K cell,
# with a GPU-free-memory gate so it waits for other tenants instead of OOMing.
#
# Protocol is byte-identical to the existing 3 seeds (verified from
# outputs/phase3_rtn/adaq_s42/training_args.bin + training_stats.json):
#   model   models/Meta-Llama-3-8B-int4rtn-g128 (pre-quantized, --no-quant)
#   data    teknium/OpenHermes-2.5 train[:2000], packing, max_length 2048
#   3 epochs, per-device batch 1, grad-accum 16, lr 2e-4, cosine, warmup 0.03
#   budget 512 per-layer, target modules q_proj,v_proj, ranks.json
#   eval    lm_eval gsm8k, 5-shot, bf16, peft=<adapter>
# peak VRAM of a training run is 19.8 GiB, hence the >=21000 MiB gate.
# usage: bash scripts/chain_rtn_power2.sh <gpu_id> <method> <seed> [<seed> ...]
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
LOG="logs/rtn_power2_gpu${GPU}_${METHOD}.log"
mkdir -p logs

gpu_free() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' -v g="${GPU}" '$1==g{print $2}'
}

wait_gpu() {
  local need=${1:-21000} max_wait=${2:-43200} waited=0 free
  while true; do
    free=$(gpu_free)
    if [ -n "${free:-}" ] && [ "${free}" -ge "${need}" ]; then
      echo "[rtn-power2] $(date -u +%FT%T) gpu${GPU} free=${free}MiB >= ${need}MiB -> go" | tee -a "$LOG"
      return 0
    fi
    if [ "${waited}" -ge "${max_wait}" ]; then
      echo "[rtn-power2] $(date -u +%FT%T) TIMEOUT waiting gpu${GPU} (free=${free:-?}MiB)" | tee -a "$LOG"
      return 1
    fi
    echo "[rtn-power2] $(date -u +%FT%T) waiting gpu${GPU} free=${free:-?}MiB < ${need}MiB (${waited}s)" | tee -a "$LOG"
    sleep 120
    waited=$((waited + 120))
  done
}

echo "[rtn-power2] $(date -u +%FT%T) START gpu=${GPU} method=${METHOD} seeds=[${SEEDS}]" | tee -a "$LOG"

for s in $SEEDS; do
  OUT="outputs/phase3_rtn/${METHOD}_s${s}"
  RES="results_v2_rtn/${METHOD}_s${s}"

  if [ -f "${RES}/.done_gsm8k" ]; then
    echo "[rtn-power2] $(date -u +%FT%T) SKIP ${METHOD}_s${s} (eval done)" | tee -a "$LOG"
    continue
  fi

  # one retry after an out-of-memory failure, each attempt behind the gate
  for attempt in 1 2; do
    if [ -f "${OUT}/adapter_model.safetensors" ]; then break; fi
    wait_gpu 21000 43200 || break
    mkdir -p "$OUT"
    echo "[rtn-power2] $(date -u +%FT%T) TRAIN ${METHOD}_s${s} attempt=${attempt}" | tee -a "$LOG"
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
    [ -f "${OUT}/adapter_model.safetensors" ] && break
    if tail -c 4000 "$LOG" | grep -q 'OutOfMemoryError'; then
      echo "[rtn-power2] $(date -u +%FT%T) OOM ${METHOD}_s${s} attempt=${attempt}" | tee -a "$LOG"
      sleep 60
    else
      echo "[rtn-power2] $(date -u +%FT%T) TRAIN-FAIL ${METHOD}_s${s} (non-OOM)" | tee -a "$LOG"
      break
    fi
  done

  if [ ! -f "${OUT}/adapter_model.safetensors" ]; then
    echo "[rtn-power2] $(date -u +%FT%T) GIVEUP ${METHOD}_s${s}" | tee -a "$LOG"
    continue
  fi
  echo "[rtn-power2] $(date -u +%FT%T) TRAIN-OK ${METHOD}_s${s}" | tee -a "$LOG"

  wait_gpu 12000 7200 || continue
  mkdir -p "$RES"
  echo "[rtn-power2] $(date -u +%FT%T) EVAL ${METHOD}_s${s} (gsm8k 5-shot)" | tee -a "$LOG"
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
    echo "[rtn-power2] $(date -u +%FT%T) EVAL-OK ${METHOD}_s${s} results=${NJ}" | tee -a "$LOG"
  else
    echo "[rtn-power2] $(date -u +%FT%T) EVAL-FAIL ${METHOD}_s${s}" | tee -a "$LOG"
  fi
  sleep 5
done

echo "[rtn-power2] $(date -u +%FT%T) ALL-DONE gpu=${GPU} method=${METHOD}" | tee -a "$LOG"
