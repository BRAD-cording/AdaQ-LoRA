#!/bin/bash
set -uo pipefail
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

MODEL="models/Meta-Llama-3-8B-int8rtn"
PEFT_DIR="outputs/phase3_int8"
OUT_DIR="results_v2_int8"
mkdir -p "$OUT_DIR" logs

for method in adaq qlora; do
  for seed in 42 3407 2025; do
    ADAPTER="${PEFT_DIR}/${method}_s${seed}"
    RESULT="${OUT_DIR}/${method}_s${seed}"
    if [ -f "${RESULT}/results_mmlu.json" ] && [ -f "${RESULT}/results_gsm8k.json" ]; then
      echo "[int8-eval] $(date -u +%FT%T) SKIP ${method}_s${seed}" | tee -a logs/int8_eval.log
      continue
    fi
    mkdir -p "$RESULT"
    echo "[int8-eval] $(date -u +%FT%T) EVAL ${method}_s${seed}" | tee -a logs/int8_eval.log
    
    env CUDA_VISIBLE_DEVICES=1 python -m lm_eval \
      --model hf --model_args pretrained="$MODEL",dtype=bfloat16,peft="$ADAPTER" \
      --tasks mmlu --num_fewshot 5 --batch_size auto \
      --output_path "$RESULT" --log_samples 2>&1 | tail -5
    
    env CUDA_VISIBLE_DEVICES=1 python -m lm_eval \
      --model hf --model_args pretrained="$MODEL",dtype=bfloat16,peft="$ADAPTER" \
      --tasks gsm8k --num_fewshot 4 --batch_size auto \
      --output_path "$RESULT" --log_samples 2>&1 | tail -5
    
    python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()" 2>/dev/null || true
    echo "[int8-eval] $(date -u +%FT%T) DONE ${method}_s${seed}" | tee -a logs/int8_eval.log
    sleep 3
  done
done
echo "[int8-eval] $(date -u +%FT%T) ALL DONE" | tee -a logs/int8_eval.log