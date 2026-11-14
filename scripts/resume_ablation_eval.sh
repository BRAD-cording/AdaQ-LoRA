#!/usr/bin/env bash
# 评测幂等恢复：tmux 会话 adaq_ablation_eval 不存在则重启全量评测
set -u
cd /home/ubuntu/AdaQ-LoRA
if tmux has-session -t adaq_ablation_eval 2>/dev/null; then
  echo "[$(date '+%F %T')] 评测会话已存在，跳过"
  exit 0
fi
echo "[$(date '+%F %T')] 重启消融评测"
tmux new-session -d -s adaq_ablation_eval \
  "cd /home/ubuntu/AdaQ-LoRA && OUT_DIR=results_v2_ablation GPUS=0,1 \
   /home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python scripts/run_eval_ablation.py \
   >> logs/ablation_eval_master.log 2>&1"
