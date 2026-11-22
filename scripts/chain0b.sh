#!/bin/bash
# chain0b.sh -- resume GPU0 work after the OOM interruption.
# The earlier failure was caused by launching an eval while the previous eval's
# process still held ~14 GiB; this version waits for actual free VRAM before
# every stage, not merely for the process to exit.
cd /home/ubuntu/AdaQ-LoRA || exit 1
LOG=logs/chain0b.log
echo "[chain0b] $(date -u +%FT%T) start" >> "$LOG"

wait_free() {
  local need=${1:-20000} waited=0 f
  while true; do
    f=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
    if [ -n "${f:-}" ] && [ "$f" -ge "$need" ]; then
      echo "[chain0b] $(date -u +%FT%T) gpu0 free=${f}MiB" >> "$LOG"
      return 0
    fi
    sleep 30
    waited=$((waited + 30))
    [ "$waited" -ge 3600 ] && { echo "[chain0b] $(date -u +%FT%T) TIMEOUT free=${f:-?}" >> "$LOG"; return 1; }
  done
}

wait_free 20000 || exit 1
echo "[chain0b] $(date -u +%FT%T) qlora NF4 alignment" >> "$LOG"
bash scripts/align_shot5.sh 0 outputs/phase3_main qlora 42 3407 2025

wait_free 20000 || exit 1
echo "[chain0b] $(date -u +%FT%T) extreme arm" >> "$LOG"
bash scripts/extreme.sh 0 42 3407

echo "[chain0b] $(date -u +%FT%T) ALL-DONE" >> "$LOG"
