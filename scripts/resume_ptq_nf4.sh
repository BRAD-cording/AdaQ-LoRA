#!/bin/bash
# P0-B1 NF4 加载对比评测的 tmux 幂等启动器（防 VMware 断电）
set -u
cd /home/ubuntu/AdaQ-LoRA
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
OUT=results_v2_ptq_nf4
SESS=adaq_ptq_nf4

if tmux has-session -t "$SESS" 2>/dev/null; then
    echo "[$(date +%F\ %T)] session $SESS already running, skip"
    exit 0
fi
tmux new-session -d -s "$SESS" \
  "OUT_DIR=$OUT GPUS=0,1 LIMIT= $PY scripts/run_eval_ptq_nf4.py | tee logs/ptq_nf4_master.log"
echo "[$(date +%F\ %T)] started session $SESS"
