#!/bin/bash
# chain0.sh -- wait for the adaq alignment on GPU0, then run qlora alignment, then the extreme arm.
cd /home/ubuntu/AdaQ-LoRA || exit 1
LOG=logs/chain0.log
echo "[chain0] $(date -u +%FT%T) waiting for align_shot5 gpu0" >> "$LOG"
while pgrep -f 'align_shot[5].sh 0' >/dev/null; do sleep 120; done
echo "[chain0] $(date -u +%FT%T) start qlora NF4 alignment" >> "$LOG"
bash scripts/align_shot5.sh 0 outputs/phase3_main qlora 42 3407 2025
echo "[chain0] $(date -u +%FT%T) start extreme arm" >> "$LOG"
bash scripts/extreme.sh 0 42 3407
echo "[chain0] $(date -u +%FT%T) ALL-DONE" >> "$LOG"
