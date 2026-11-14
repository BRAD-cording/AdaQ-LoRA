#!/usr/bin/env bash
# 消融训练监控：每 3min 检查一次，写进度
cd /home/ubuntu/AdaQ-LoRA
for i in $(seq 1 40); do
  ts=$(date '+%F %T')
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')
  nproc=$(ps -ef | grep "train.py --method adaq" | grep -v grep | wc -l)
  step=$(grep -ohE "[0-9]+/21" logs/ablation_*.log 2>/dev/null | tail -1)
  echo "[$ts] gpu=($gpu) nproc=$nproc step=$step" >> logs/monitor_ablation.log
  sleep 180
done
