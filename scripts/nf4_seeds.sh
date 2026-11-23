#!/bin/bash
# nf4_seeds.sh -- add NF4 GSM8K seeds so the weakest cell reaches n>=10.
#
# The 5-shot re-score left NF4 GSM8K at n=3 with SD 1.57 pp and a CI of
# [-2.91,+4.88]; a two-point equivalence claim there needs roughly ten seeds.
# Same recipe as outputs/phase3_main: bnb NF4, batch 4 x accumulation 4, 3 epochs,
# lr 2e-4, then GSM8K 5-shot at batch 16.
# usage: bash scripts/nf4_seeds.sh <gpu> <seed>...
set -uo pipefail
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
GPU=${1:?gpu}
shift
LOG="logs/nf4fill_gpu${GPU}.log"
mkdir -p logs

for s in "$@"; do
  for meth in adaq qlora; do
    ADP="outputs/phase3_main_fill/${meth}_s${s}"
    RES="results_nf4fill/${meth}_s${s}"
    if [ ! -f "${ADP}/adapter_model.safetensors" ]; then
      mkdir -p "$ADP"
      echo "[nf4fill] $(date -u +%FT%T) TRAIN ${meth}_s${s}" | tee -a "$LOG"
      env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" scripts/train.py \
        --model models/Meta-Llama-3-8B --method "$meth" --seed "$s" --output-dir "$ADP" \
        --batch-size 1 --grad-accum 16 --epochs 3 --dataset-size 2000 --lr 2e-4 \
        >> "$LOG" 2>&1
      [ -f "${ADP}/adapter_model.safetensors" ] || { echo "[nf4fill] TRAIN-FAIL ${meth}_s${s}" | tee -a "$LOG"; continue; }
      echo "[nf4fill] $(date -u +%FT%T) TRAIN-OK ${meth}_s${s}" | tee -a "$LOG"
    fi
    if [ ! -f "${RES}/gsm8k/.done" ]; then
      mkdir -p "${RES}/gsm8k"
      echo "[nf4fill] $(date -u +%FT%T) EVAL gsm8k ${meth}_s${s}" | tee -a "$LOG"
      env CUDA_VISIBLE_DEVICES="${GPU}" "$PY" -u -m lm_eval --model hf \
        --model_args "pretrained=models/Meta-Llama-3-8B,peft=${ADP},trust_remote_code=True,dtype=bfloat16" \
        --tasks gsm8k --batch_size 16 --num_fewshot 5 \
        --output_path "${RES}/gsm8k" >> "$LOG" 2>&1
      find "${RES}/gsm8k" -name 'results_*.json' | grep -q . && touch "${RES}/gsm8k/.done" \
        && echo "[nf4fill] $(date -u +%FT%T) OK ${meth}_s${s}" | tee -a "$LOG"
    fi
  done
done
echo "[nf4fill] $(date -u +%FT%T) ALL-DONE gpu=${GPU}" | tee -a "$LOG"
