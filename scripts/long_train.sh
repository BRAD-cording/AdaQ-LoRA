#!/bin/bash
# long_train.sh -- the strongest threat to this audit is training volume: 63
# packed steps is a light fine-tune, and rank allocation is a capacity question
# that only bites when the task needs capacity. This reruns the primary
# comparison (QLoRA vs AdaQ, NF4) at 10 epochs instead of 3 (~210 packed steps)
# so the null can be re-examined under a heavier schedule.
#
# usage: bash scripts/long_train.sh <gpu> <seed>...
set -uo pipefail
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
GPU=${1:?gpu}
shift
LOG="logs/longtrain_gpu${GPU}.log"
mkdir -p logs

for s in "$@"; do
  for meth in adaq qlora; do
    ADP="outputs/long_train/${meth}_s${s}"
    RES="results_long_train/${meth}_s${s}"
    if [ ! -f "${ADP}/adapter_model.safetensors" ]; then
      mkdir -p "$ADP"
      echo "[longtrain] $(date -u +%FT%T) TRAIN ${meth}_s${s} epochs=10" | tee -a "$LOG"
      "$PY" scripts/safe_run.py --gpu "$GPU" --min-free 20000 --log "$LOG" -- \
        env CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train.py --model models/Meta-Llama-3-8B --method "$meth" \
        --seed "$s" --output-dir "$ADP" \
        --batch-size 1 --grad-accum 16 --epochs 10 --dataset-size 2000 --lr 2e-4 \
        >> "$LOG" 2>&1
      [ -f "${ADP}/adapter_model.safetensors" ] || { echo "[longtrain] TRAIN-FAIL ${meth}_s${s}" | tee -a "$LOG"; continue; }
      echo "[longtrain] $(date -u +%FT%T) TRAIN-OK ${meth}_s${s}" | tee -a "$LOG"
    fi
    for task in gsm8k mmlu; do
      if [ "$task" = "gsm8k" ]; then BS=16; else BS=2; fi
      if [ ! -f "${RES}/${task}/.done" ]; then
        mkdir -p "${RES}/${task}"
        echo "[longtrain] $(date -u +%FT%T) EVAL ${task} ${meth}_s${s}" | tee -a "$LOG"
        "$PY" scripts/safe_run.py --gpu "$GPU" --min-free 18000 --log "$LOG" -- \
          env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u -m lm_eval --model hf \
          --model_args "pretrained=models/Meta-Llama-3-8B,peft=${ADP},trust_remote_code=True,dtype=bfloat16" \
          --tasks "$task" --batch_size "$BS" --num_fewshot 5 \
          --output_path "${RES}/${task}" >> "$LOG" 2>&1
        find "${RES}/${task}" -name 'results_*.json' | grep -q . && touch "${RES}/${task}/.done" \
          && echo "[longtrain] $(date -u +%FT%T) OK ${task} ${meth}_s${s}" | tee -a "$LOG"
      fi
    done
  done
done
echo "[longtrain] $(date -u +%FT%T) ALL-DONE gpu=${GPU}" | tee -a "$LOG"
