#!/bin/bash
# chain1.sh -- wait for the adaq alignment on GPU1, then run qlora alignment, then the extreme arm.
cd /home/ubuntu/AdaQ-LoRA || exit 1
LOG=logs/chain1.log
echo "[chain1] $(date -u +%FT%T) waiting for align_shot5 gpu1" >> "$LOG"
while pgrep -f 'align_shot[5].sh 1' >/dev/null; do sleep 120; done
echo "[chain1] $(date -u +%FT%T) start qlora INT8 alignment" >> "$LOG"
bash scripts/align_shot5.sh 1 outputs/phase3_int8 qlora 42 3407 2025 7 100
echo "[chain1] $(date -u +%FT%T) start extreme arm" >> "$LOG"
bash scripts/extreme.sh 1 2025
echo "[chain1] $(date -u +%FT%T) ALL-DONE" >> "$LOG"
