#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

LOG="logs/gpu0_eval_v2.log"
echo "[gpu0] $(date -u +%Y-%m-%dT%H:%M:%S) START GSM8K adaq_s42" | tee "$LOG"

RTN_MODEL="models/Meta-Llama-3-8B-int4rtn-g128"
python -u -m lm_eval --model hf --model_args "pretrained=${RTN_MODEL},peft=outputs/phase3_rtn/adaq_s42,trust_remote_code=True,dtype=bfloat16" --tasks gsm8k --batch_size auto --output_path results_v2_rtn/adaq_s42 --num_fewshot 4 >> "$LOG" 2>&1
rc=$?
[ $rc -ne 0 ] && echo "[gpu0] FAIL gsm8k adaq_s42 rc=$rc" | tee -a "$LOG" && exit $rc
echo "[gpu0] $(date -u +%Y-%m-%dT%H:%M:%S) OK gsm8k adaq_s42" | tee -a "$LOG"

echo "[gpu0] $(date -u +%Y-%m-%dT%H:%M:%S) START INT8 evals" | tee -a "$LOG"
INT8_MODEL="models/Meta-Llama-3-8B-int8rtn"
for job in "adaq_s7" "qlora_s7"; do
  ADAPTER="outputs/phase3_int8/${job}"
  OUTDIR="results_v2_int8/${job}"
  mkdir -p "$OUTDIR"
  echo "[gpu0] $(date -u +%Y-%m-%dT%H:%M:%S) EVAL ${job}" | tee -a "$LOG"
  python -u -m lm_eval --model hf --model_args "pretrained=${INT8_MODEL},peft=${ADAPTER},trust_remote_code=True,dtype=bfloat16" --tasks mmlu,gsm8k --batch_size auto --output_path "$OUTDIR" --num_fewshot 5 >> "$LOG" 2>&1
  rc=$?
  [ $rc -ne 0 ] && echo "[gpu0] FAIL ${job} rc=$rc" | tee -a "$LOG" && exit $rc
  echo "[gpu0] $(date -u +%Y-%m-%dT%H:%M:%S) OK ${job}" | tee -a "$LOG"
done

echo "[gpu0] $(date -u +%Y-%m-%dT%H:%M:%S) ALL DONE" | tee -a "$LOG"