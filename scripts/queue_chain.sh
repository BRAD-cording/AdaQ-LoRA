#!/bin/bash
# queue_chain.sh -- after the adaq alignment finishes on each GPU, continue with
# the qlora alignment and then the dose-response (extreme) arm.
cd /home/ubuntu/AdaQ-LoRA || exit 1

nohup setsid bash -c 'while pgrep -f "align_shot5.sh 0" >/dev/null; do sleep 120; done; \
  bash scripts/align_shot5.sh 0 outputs/phase3_main qlora 42 3407 2025; \
  bash scripts/extreme.sh 0 42 3407' \
  > logs/launch_chain0.log 2>&1 < /dev/null &

nohup setsid bash -c 'while pgrep -f "align_shot5.sh 1" >/dev/null; do sleep 120; done; \
  bash scripts/align_shot5.sh 1 outputs/phase3_int8 qlora 42 3407 2025 7 100; \
  bash scripts/extreme.sh 1 2025' \
  > logs/launch_chain1.log 2>&1 < /dev/null &

sleep 8
echo "=== running ==="
ps -eo pid,cmd | grep -E 'align_shot5|extreme\.sh' | grep -v grep | cut -c1-100
