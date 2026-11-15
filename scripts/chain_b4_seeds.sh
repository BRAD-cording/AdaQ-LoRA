#!/bin/bash
# P1-B4 补齐种子全链路：训练(2025,3407) → 评测 → 聚合
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
LOG=logs/chain_b4_seeds.log
exec 9>/tmp/adaq_b4_seeds.lock
flock -n 9 || { echo "[chain-b4-seeds] 已有实例在跑，退出 $(date -u)"; exit 0; }

echo "=== [chain-b4-seeds] START $(date -u) ===" | tee -a "$LOG"
/bin/bash scripts/run_b4_qwen14b_seeds.sh 2>&1 | tee -a "$LOG"
echo "=== [chain-b4-seeds] 训练阶段结束 $(date -u) ===" | tee -a "$LOG"

/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python scripts/run_eval_b4_qwen14b_seeds.py 2>&1 | tee -a "$LOG"
echo "=== [chain-b4-seeds] 评测阶段结束 $(date -u) ===" | tee -a "$LOG"

/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python scripts/aggregate_b4_3seeds.py 2>&1 | tee -a "$LOG"
echo "=== [chain-b4-seeds] DONE $(date -u) ===" | tee -a "$LOG"
