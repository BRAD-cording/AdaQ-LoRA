#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

LOG="logs/gpu1_eval.log"
echo "[gpu1] $(date -u +%Y-%m-%dT%H:%M:%S) START Phase1 GSM8K re-evals" | tee "$LOG"

RTN_MODEL="models/Meta-Llama-3-8B-int4rtn-g128"

# RTN GSM8K re-evals
for job in "qlora_s3407" "qlora_s42"; do
  ADAPTER="outputs/phase3_rtn/${job}"
  OUTDIR="results_v2_rtn/${job}"
  echo "[gpu1] $(date -u +%Y-%m-%dT%H:%M:%S) GSM8K ${job}" | tee -a "$LOG"
  python -u -m lm_eval --model hf --model_args "pretrained=${RTN_MODEL},peft=${ADAPTER},trust_remote_code=True,dtype=bfloat16" --tasks gsm8k --batch_size auto --output_path "$OUTDIR" --num_fewshot 4 >> "$LOG" 2>&1
  rc=$?
  [ $rc -ne 0 ] && echo "[gpu1] FAIL gsm8k ${job} rc=$rc" | tee -a "$LOG" && exit $rc
  echo "[gpu1] $(date -u +%Y-%m-%dT%H:%M:%S) OK gsm8k ${job}" | tee -a "$LOG"
done

echo "[gpu1] $(date -u +%Y-%m-%dT%H:%M:%S) START Phase2 INT8 evals" | tee -a "$LOG"

INT8_MODEL="models/Meta-Llama-3-8B-int8rtn"
for job in "adaq_s100" "qlora_s100"; do
  ADAPTER="outputs/phase3_int8/${job}"
  OUTDIR="results_v2_int8/${job}"
  mkdir -p "$OUTDIR"
  echo "[gpu1] $(date -u +%Y-%m-%dT%H:%M:%S) EVAL ${job}" | tee -a "$LOG"
  python -u -m lm_eval --model hf --model_args "pretrained=${INT8_MODEL},peft=${ADAPTER},trust_remote_code=True,dtype=bfloat16" --tasks mmlu,gsm8k --batch_size auto --output_path "$OUTDIR" --num_fewshot 5 >> "$LOG" 2>&1
  rc=$?
  [ $rc -ne 0 ] && echo "[gpu1] FAIL ${job} rc=$rc" | tee -a "$LOG" && exit $rc
  echo "[gpu1] $(date -u +%Y-%m-%dT%H:%M:%S) OK ${job}" | tee -a "$LOG"
done

echo "[gpu1] $(date -u +%Y-%m-%dT%H:%M:%S) ALL DONE" | tee -a "$LOG"