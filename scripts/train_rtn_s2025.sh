#!/bin/bash
set -uo pipefail
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p logs

echo "[rtn-s2025] $(date -u +%FT%T) START" | tee -a logs/rtn_s2025.log

RTN_MODEL="models/Meta-Llama-3-8B-int4rtn-g128"

for method in adaq qlora; do
  OUTDIR="outputs/phase3_rtn/${method}_s2025"
  if [ -f "${OUTDIR}/adapter_model.safetensors" ]; then
    echo "[rtn-s2025] $(date -u +%FT%T) SKIP ${method}_s2025 (exists)" | tee -a logs/rtn_s2025.log
    continue
  fi
  echo "[rtn-s2025] $(date -u +%FT%T) TRAIN ${method}_s2025" | tee -a logs/rtn_s2025.log
  
  env CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    --model "$RTN_MODEL" --method "$method" --seed 2025 \
    --output "$OUTDIR" --no-quant --batch_size 1 \
    2>&1 | tail -5
  
  if [ -f "${OUTDIR}/adapter_model.safetensors" ]; then
    echo "[rtn-s2025] $(date -u +%FT%T) DONE ${method}_s2025" | tee -a logs/rtn_s2025.log
  else
    echo "[rtn-s2025] $(date -u +%FT%T) FAIL ${method}_s2025" | tee -a logs/rtn_s2025.log
  fi
  sleep 3
done

echo "[rtn-s2025] $(date -u +%FT%T) ALL DONE" | tee -a logs/rtn_s2025.log
