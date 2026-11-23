#!/bin/bash
# int8_fix.sh -- retrain INT8 seeds 7 and 100 with the gradient accumulation the
# other INT8 seeds used.
#
# Finding: run_int8_seeds_p2_v2.sh passed --batch-size 1 but not --grad-accum, so
# those two seeds silently trained at accumulation 4 instead of 16 -- a four-fold
# difference in effective batch and hence in optimiser steps. Their GSM8K scores
# sit 12 points below the other three seeds, which is a configuration artefact,
# not seed noise.
#
# Old adapters are left untouched; retrained ones go to <name>_ga16 and results
# to results_int8fix/.
# usage: bash scripts/int8_fix.sh <gpu> <method>
set -uo pipefail
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
MODEL=models/Meta-Llama-3-8B-int8rtn
GPU=${1:?gpu}
METHOD=${2:?method}
LOG="logs/int8fix_gpu${GPU}_${METHOD}.log"
mkdir -p logs

for s in 7 100; do
  ADP="outputs/phase3_int8/${METHOD}_s${s}_ga16"
  RES="results_int8fix/${METHOD}_s${s}"
  if [ ! -f "${ADP}/adapter_model.safetensors" ]; then
    mkdir -p "$ADP"
    echo "[int8fix] $(date -u +%FT%T) TRAIN ${METHOD}_s${s} ga=16" | tee -a "$LOG"
    env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" scripts/train.py \
      --model "$MODEL" --method "$METHOD" --seed "$s" --output-dir "$ADP" \
      --no-quant --batch-size 1 --grad-accum 16 --epochs 3 --dataset-size 2000 --lr 2e-4 \
      >> "$LOG" 2>&1
    [ -f "${ADP}/adapter_model.safetensors" ] || { echo "[int8fix] TRAIN-FAIL s${s}" | tee -a "$LOG"; continue; }
    echo "[int8fix] $(date -u +%FT%T) TRAIN-OK s${s}" | tee -a "$LOG"
  fi
  for task in gsm8k mmlu; do
    if [ "$task" = "gsm8k" ]; then BS=16; else BS=2; fi
    if [ ! -f "${RES}/${task}/.done" ]; then
      mkdir -p "${RES}/${task}"
      echo "[int8fix] $(date -u +%FT%T) EVAL ${task} s${s} bs=${BS}" | tee -a "$LOG"
      env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" -u -m lm_eval --model hf \
        --model_args "pretrained=${MODEL},peft=${ADP},trust_remote_code=True,dtype=bfloat16" \
        --tasks "$task" --batch_size "$BS" --num_fewshot 5 \
        --output_path "${RES}/${task}" >> "$LOG" 2>&1
      if find "${RES}/${task}" -name 'results_*.json' | grep -q .; then
        touch "${RES}/${task}/.done"
        echo "[int8fix] $(date -u +%FT%T) OK ${task} s${s}" | tee -a "$LOG"
      else
        echo "[int8fix] $(date -u +%FT%T) FAIL ${task} s${s}" | tee -a "$LOG"
      fi
    fi
  done
done
echo "[int8fix] $(date -u +%FT%T) ALL-DONE gpu=${GPU} method=${METHOD}" | tee -a "$LOG"
