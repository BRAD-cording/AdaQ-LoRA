#!/bin/bash
# chain_rtn_b16.sh -- one uniform protocol for every INT4-RTN seed:
#   train (if the adapter is missing) -> gsm8k 5-shot batch16 -> (optionally) mmlu 5-shot batch16
#
# Why: the legacy results mixed num_fewshot 4 and 5 across paired seeds and were
# produced at batch_size=1 (68 min per gsm8k eval). batch_size=16 is the largest
# that fits in 24 GB alongside a 16 GB bf16 model (32 OOMs), and cuts an eval to
# about 17 min. All results therefore go to results_v2_rtn_b16/ so that the
# legacy files stay untouched as evidence.
#
# usage: MMLU_SEEDS="42 3407 2025" bash scripts/chain_rtn_b16.sh <gpu> <method> <seed>...
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
GPU=${1:?gpu id}
METHOD=${2:?method}
shift 2
SEEDS="$*"
MMLU_SEEDS="${MMLU_SEEDS:-}"
LOG="logs/rtn_b16_gpu${GPU}_${METHOD}.log"
mkdir -p logs "$OUTROOT"

gpu_free() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' -v g="${GPU}" '$1==g{print $2}'
}

wait_free() {
  local need=$1 max=${2:-43200} waited=0 free
  while true; do
    free=$(gpu_free)
    if [ -n "${free:-}" ] && [ "${free}" -ge "${need}" ]; then return 0; fi
    [ "${waited}" -ge "${max}" ] && { echo "[b16] TIMEOUT gpu${GPU} free=${free:-?}"; return 1; }
    sleep 180
    waited=$((waited + 180))
  done
}

echo "[b16] $(date -u +%FT%T) START gpu=${GPU} method=${METHOD} seeds=[${SEEDS}] mmlu_seeds=[${MMLU_SEEDS}]" | tee -a "$LOG"

for s in $SEEDS; do
  ADP="outputs/phase3_rtn/${METHOD}_s${s}"
  RES="${OUTROOT}/${METHOD}_s${s}"

  # ---- train if needed
  if [ ! -f "${ADP}/adapter_model.safetensors" ]; then
    for attempt in 1 2 3; do
      wait_free 21000 14400 || break
      mkdir -p "$ADP"
      echo "[b16] $(date -u +%FT%T) TRAIN ${METHOD}_s${s} attempt=${attempt}" | tee -a "$LOG"
      env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" scripts/train.py \
        --model "$MODEL" --method "$METHOD" --seed "$s" --output-dir "$ADP" \
        --no-quant --batch-size 1 --grad-accum 16 --epochs 3 --dataset-size 2000 --lr 2e-4 \
        >> "$LOG" 2>&1
      [ -f "${ADP}/adapter_model.safetensors" ] && break
      tail -c 3000 "$LOG" | grep -q 'OutOfMemoryError' || break
      echo "[b16] $(date -u +%FT%T) OOM during train, waiting" | tee -a "$LOG"
      sleep 300
    done
  fi
  if [ ! -f "${ADP}/adapter_model.safetensors" ]; then
    echo "[b16] $(date -u +%FT%T) SKIP ${METHOD}_s${s} (no adapter)" | tee -a "$LOG"
    continue
  fi

  # ---- gsm8k, 5-shot, batch 16 (fallback 8 on OOM)
  if [ ! -f "${RES}/gsm8k/.done" ]; then
    for bs in 16 8; do
      wait_free 18000 7200 || break
      mkdir -p "${RES}/gsm8k"
      echo "[b16] $(date -u +%FT%T) EVAL gsm8k ${METHOD}_s${s} bs=${bs}" | tee -a "$LOG"
      env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" -u -m lm_eval --model hf \
        --model_args "pretrained=${MODEL},peft=${ADP},trust_remote_code=True,dtype=bfloat16" \
        --tasks gsm8k --batch_size "${bs}" --num_fewshot 5 \
        --output_path "${RES}/gsm8k" >> "$LOG" 2>&1
      if find "${RES}/gsm8k" -name 'results_*.json' | grep -q .; then
        touch "${RES}/gsm8k/.done"
        echo "[b16] $(date -u +%FT%T) OK gsm8k ${METHOD}_s${s} bs=${bs}" | tee -a "$LOG"
        break
      fi
      tail -c 3000 "$LOG" | grep -q 'out of memory' || break
      echo "[b16] $(date -u +%FT%T) gsm8k OOM bs=${bs}, retry smaller" | tee -a "$LOG"
    done
  fi

  # ---- mmlu, 5-shot, batch 16 (only for the seeds that need it)
  case " ${MMLU_SEEDS} " in
    *" ${s} "*)
      if [ ! -f "${RES}/mmlu/.done" ]; then
        wait_free 18000 7200 || continue
        mkdir -p "${RES}/mmlu"
        echo "[b16] $(date -u +%FT%T) EVAL mmlu ${METHOD}_s${s} bs=16" | tee -a "$LOG"
        env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" -u -m lm_eval --model hf \
          --model_args "pretrained=${MODEL},peft=${ADP},trust_remote_code=True,dtype=bfloat16" \
          --tasks mmlu --batch_size 16 --num_fewshot 5 \
          --output_path "${RES}/mmlu" >> "$LOG" 2>&1
        if find "${RES}/mmlu" -name 'results_*.json' | grep -q .; then
          touch "${RES}/mmlu/.done"
          echo "[b16] $(date -u +%FT%T) OK mmlu ${METHOD}_s${s}" | tee -a "$LOG"
        else
          echo "[b16] $(date -u +%FT%T) FAIL mmlu ${METHOD}_s${s}" | tee -a "$LOG"
        fi
      fi
      ;;
  esac
  sleep 3
done

echo "[b16] $(date -u +%FT%T) ALL-DONE gpu=${GPU} method=${METHOD}" | tee -a "$LOG"
