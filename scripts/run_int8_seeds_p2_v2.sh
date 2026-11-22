#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA

export HF_DATASETS_OFFLINE=1
LOG="logs/int8_seeds_p2.log"
echo "[int8-p2] $(date -u +%Y-%m-%dT%H:%M:%S) RESTART P2 INT8 (offline mode)" | tee "$LOG"

export CUDA_VISIBLE_DEVICES=0
MODEL="models/Meta-Llama-3-8B-int8rtn"

for method in adaq qlora; do
  for seed in 7 100; do
    OUTDIR="outputs/phase3_int8/${method}_s${seed}"
    # Skip if adapter already exists
    if [ -f "$OUTDIR/adapter_model.safetensors" ]; then
      echo "[int8-p2] $(date -u +%Y-%m-%dT%H:%M:%S) SKIP ${method}_s${seed} (exists)" | tee -a "$LOG"
      continue
    fi
    echo "[int8-p2] $(date -u +%Y-%m-%dT%H:%M:%S) TRAIN ${method}_s${seed}" | tee -a "$LOG"
    python -u scripts/train.py --model "$MODEL" --method "$method" --seed "$seed" --output-dir "$OUTDIR" --no-quant --batch-size 1 >> "$LOG" 2>&1
    rc=$?
    [ $rc -ne 0 ] && echo "[int8-p2] $(date -u +%Y-%m-%dT%H:%M:%S) FAIL ${method}_s${seed}" | tee -a "$LOG" && exit $rc
    echo "[int8-p2] $(date -u +%Y-%m-%dT%H:%M:%S) OK ${method}_s${seed}" | tee -a "$LOG"
  done
done
echo "[int8-p2] $(date -u +%Y-%m-%dT%H:%M:%S) ALL DONE" | tee -a "$LOG"
