#!/bin/bash
# bf16_lora.sh -- the missing control: bf16 base + LoRA, no quantisation.
#
# Why it matters: the manuscript argues that MMLU tracks quantisation damage
# rather than adapter quality, comparing bf16 base (0.6536) with NF4 QLoRA
# (0.6533). That comparison confounds quantisation with fine-tuning, because
# there is no bf16-plus-LoRA arm. This script trains that arm with the same
# recipe (uniform ranks, batch 1 x 16, 3 epochs, lr 2e-4) on the same seeds, so
# the three-way decomposition becomes possible:
#   bf16 base  ->  bf16 + LoRA  ->  NF4 + LoRA
#
# usage: bash scripts/bf16_lora.sh <gpu> <seed>...
set -uo pipefail
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
GPU=${1:?gpu}
shift
LOG="logs/bf16lora_gpu${GPU}.log"
mkdir -p logs

for s in "$@"; do
  ADP="outputs/bf16_lora/qlora_s${s}"
  RES="results_bf16_lora/qlora_s${s}"
  if [ ! -f "${ADP}/adapter_model.safetensors" ]; then
    mkdir -p "$ADP"
    echo "[bf16] $(date -u +%FT%T) TRAIN qlora_s${s} (bf16, no quant)" | tee -a "$LOG"
    "$PY" scripts/safe_run.py --gpu "$GPU" --min-free 20000 --log "$LOG" -- \
      env CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train.py \
      --model models/Meta-Llama-3-8B --method qlora --seed "$s" --output-dir "$ADP" \
      --no-quant --batch-size 1 --grad-accum 16 --epochs 3 --dataset-size 2000 --lr 2e-4 \
      >> "$LOG" 2>&1
    [ -f "${ADP}/adapter_model.safetensors" ] || { echo "[bf16] TRAIN-FAIL s${s}" | tee -a "$LOG"; continue; }
    echo "[bf16] $(date -u +%FT%T) TRAIN-OK s${s}" | tee -a "$LOG"
  fi
  for task in gsm8k mmlu; do
    if [ "$task" = "gsm8k" ]; then BS=16; else BS=2; fi
    if [ ! -f "${RES}/${task}/.done" ]; then
      mkdir -p "${RES}/${task}"
      echo "[bf16] $(date -u +%FT%T) EVAL ${task} s${s}" | tee -a "$LOG"
      "$PY" scripts/safe_run.py --gpu "$GPU" --min-free 18000 --log "$LOG" -- \
        env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u -m lm_eval --model hf \
        --model_args "pretrained=models/Meta-Llama-3-8B,peft=${ADP},trust_remote_code=True,dtype=bfloat16" \
        --tasks "$task" --batch_size "$BS" --num_fewshot 5 \
        --output_path "${RES}/${task}" >> "$LOG" 2>&1
      find "${RES}/${task}" -name 'results_*.json' | grep -q . && touch "${RES}/${task}/.done" \
        && echo "[bf16] $(date -u +%FT%T) OK ${task} s${s}" | tee -a "$LOG"
    fi
  done
done
echo "[bf16] $(date -u +%FT%T) ALL-DONE gpu=${GPU}" | tee -a "$LOG"
