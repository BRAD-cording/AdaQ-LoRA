#!/bin/bash
# extreme.sh -- train and evaluate the maximum-contrast allocation arm (NF4).
# q_proj r=14, v_proj r=2, budget unchanged at 512: a deliberately stronger
# intervention than the paper's own (q=12, v=4). If this still moves nothing,
# the null cannot be blamed on a weak manipulation.
# usage: bash scripts/extreme.sh <gpu> <seed> [<seed> ...]
set -uo pipefail
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
GPU=${1:?gpu}
shift
LOG="logs/extreme_gpu${GPU}.log"
mkdir -p logs

for s in "$@"; do
  ADP="outputs/phase3_nf4_extreme/adaq_s${s}"
  RES="results_extreme/adaq_s${s}"
  if [ ! -f "${ADP}/adapter_model.safetensors" ]; then
    mkdir -p "$ADP"
    echo "[extreme] $(date -u +%FT%T) TRAIN seed=${s}" | tee -a "$LOG"
    env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" scripts/train.py \
      --model models/Meta-Llama-3-8B --method adaq --seed "$s" --output-dir "$ADP" \
      --ranks-file configs/ranks_extreme.json --budget 512 \
      --batch-size 1 --grad-accum 16 --epochs 3 --dataset-size 2000 --lr 2e-4 \
      >> "$LOG" 2>&1
    [ -f "${ADP}/adapter_model.safetensors" ] || { echo "[extreme] TRAIN-FAIL seed=${s}" | tee -a "$LOG"; continue; }
    echo "[extreme] $(date -u +%FT%T) TRAIN-OK seed=${s}" | tee -a "$LOG"
  fi
  for task in gsm8k mmlu; do
    if [ "$task" = "gsm8k" ]; then BS=16; else BS=4; fi
    if [ ! -f "${RES}/${task}/.done" ]; then
      mkdir -p "${RES}/${task}"
      echo "[extreme] $(date -u +%FT%T) EVAL ${task} seed=${s} bs=${BS}" | tee -a "$LOG"
      env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" -u -m lm_eval --model hf \
        --model_args "pretrained=models/Meta-Llama-3-8B,peft=${ADP},trust_remote_code=True,dtype=bfloat16" \
        --tasks "$task" --batch_size "$BS" --num_fewshot 5 \
        --output_path "${RES}/${task}" >> "$LOG" 2>&1
      if find "${RES}/${task}" -name 'results_*.json' | grep -q .; then
        touch "${RES}/${task}/.done"
        echo "[extreme] $(date -u +%FT%T) OK ${task} seed=${s}" | tee -a "$LOG"
      else
        echo "[extreme] $(date -u +%FT%T) FAIL ${task} seed=${s}" | tee -a "$LOG"
      fi
    fi
  done
  sleep 3
done
echo "[extreme] $(date -u +%FT%T) ALL-DONE gpu=${GPU}" | tee -a "$LOG"
