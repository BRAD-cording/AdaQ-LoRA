#!/bin/bash
set -uo pipefail
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu
cd /home/ubuntu/AdaQ-LoRA
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p logs results_v2_rtn

echo "[rtn+retry] $(date -u +%FT%T) START" | tee -a logs/rtn_retry.log

# === PHASE 1: RTN eval (INT4-RTN model at bf16, NO bnb quantization) ===
# CRITICAL: adapters were trained on INT4-RTN model loaded at bf16
# Must eval on the SAME base model
RTN_BASE="models/Meta-Llama-3-8B-int4rtn-g128"
RTN_PEFT="outputs/phase3_rtn"

for method in adaq qlora; do
  for seed in 42 3407; do
    ADAPTER="${RTN_PEFT}/${method}_s${seed}"
    RESULT="results_v2_rtn/${method}_s${seed}"
    
    if [ -f "${RESULT}/.done_gsm8k" ]; then
      echo "[rtn+retry] $(date -u +%FT%T) SKIP ${method}_s${seed}" | tee -a logs/rtn_retry.log
      continue
    fi
    
    mkdir -p "$RESULT"
    echo "[rtn+retry] $(date -u +%FT%T) EVAL ${method}_s${seed}" | tee -a logs/rtn_retry.log
    
    env CUDA_VISIBLE_DEVICES=1 python -m lm_eval \
      --model hf --model_args pretrained="$RTN_BASE",peft="$ADAPTER",dtype=bfloat16 \
      --tasks mmlu --num_fewshot 5 --batch_size auto \
      --output_path "$RESULT" 2>&1 | tail -3
    
    env CUDA_VISIBLE_DEVICES=1 python -m lm_eval \
      --model hf --model_args pretrained="$RTN_BASE",peft="$ADAPTER",dtype=bfloat16 \
      --tasks gsm8k --num_fewshot 4 --batch_size auto \
      --output_path "$RESULT" 2>&1 | tail -3
    
    touch "${RESULT}/.done_gsm8k"
    echo "[rtn+retry] $(date -u +%FT%T) DONE ${method}_s${seed}" | tee -a logs/rtn_retry.log
    sleep 3
  done
done

echo "[rtn+retry] $(date -u +%FT%T) RTN PHASE DONE" | tee -a logs/rtn_retry.log

# === PHASE 2: adaq_s42 chat-eval retry (NF4 model, correct) ===
NF4_MODEL="models/Meta-Llama-3-8B"
ADAQ_ADAPTER="outputs/phase3_main/adaq_s42"
RETRY_RESULT="results_v2_phase3_chat/adaq_s42"

if [ -f "${RETRY_RESULT}/.done_gsm8k_chat" ]; then
  echo "[rtn+retry] $(date -u +%FT%T) SKIP adaq_s42 retry (already done)" | tee -a logs/rtn_retry.log
else
  echo "[rtn+retry] $(date -u +%FT%T) RETRY adaq_s42 chat-eval" | tee -a logs/rtn_retry.log
  mkdir -p "$RETRY_RESULT"
  
  env CUDA_VISIBLE_DEVICES=1 python -m lm_eval \
    --model hf --model_args pretrained="$NF4_MODEL",peft="$ADAQ_ADAPTER",load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16 \
    --tasks gsm8k --num_fewshot 4 --apply_chat_template --fewshot_as_multiturn \
    --batch_size auto --device cuda:0 --seed 42 \
    --output_path "$RETRY_RESULT" 2>&1 | tail -3
  
  touch "${RETRY_RESULT}/.done_gsm8k_chat"
  echo "[rtn+retry] $(date -u +%FT%T) RETRY DONE adaq_s42" | tee -a logs/rtn_retry.log
fi

echo "[rtn+retry] $(date -u +%FT%T) ALL DONE" | tee -a logs/rtn_retry.log
