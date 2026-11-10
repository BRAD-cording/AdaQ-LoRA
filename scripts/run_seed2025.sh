#!/usr/bin/env bash
# 等 GPU 空闲后跑 seed 2025
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
while pgrep -f "lm_eval.*seed.*(42|3407)" > /dev/null 2>&1; do sleep 60; done
sleep 30
OUT_DIR=results_v2 nohup bash scripts/eval_worker.sh 0 "qlora dora adalora adaq" "2025" > logs/eval_gpu0_seed2025.log 2>&1 &
