#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA

export HF_DATASETS_OFFLINE=1
LOG="logs/rtn_eval_p1.log"
echo "[rtn-eval] $(date -u +%Y-%m-%dT%H:%M:%S) RESTART RTN eval (offline mode)" | tee "$LOG"

export CUDA_VISIBLE_DEVICES=1
MODEL="models/Meta-Llama-3-8B-int4rtn-g128"

for method in adaq qlora; do
  for seed in 42 3407 2025; do
    ADAPTER="outputs/phase3_rtn/${method}_s${seed}"
    OUTDIR="results_v2_rtn/${method}_s${seed}"
    mkdir -p "$OUTDIR"
    echo "[rtn-eval] $(date -u +%Y-%m-%dT%H:%M:%S) EVAL ${method}_s${seed}" | tee -a "$LOG"
    python -u -m lm_eval --model hf --model_args "pretrained=${MODEL},peft=${ADAPTER},trust_remote_code=True,dtype=bfloat16" --tasks mmlu,gsm8k --batch_size auto --output_path "$OUTDIR" --num_fewshot 5 --log_samples >> "$LOG" 2>&1
    rc=$?
    [ $rc -ne 0 ] && echo "[rtn-eval] $(date -u +%Y-%m-%dT%H:%M:%S) FAIL ${method}_s${seed} rc=$rc" | tee -a "$LOG" && exit $rc
    echo "[rtn-eval] $(date -u +%Y-%m-%dT%H:%M:%S) OK ${method}_s${seed}" | tee -a "$LOG"
  done
done
echo "[rtn-eval] $(date -u +%Y-%m-%dT%H:%M:%S) ALL DONE" | tee -a "$LOG"
