#!/bin/bash
# status.sh -- single source of truth for the RTN seed-extension run state.
cd /home/ubuntu/AdaQ-LoRA || exit 1
echo "=== now ==="
date -u '+%F %T UTC'
echo "=== chain processes ==="
ps -eo pid,etime,stat,cmd | grep 'chain_rtn_powe[r]2' | grep -v grep | cut -c1-110
echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "=== logs (mtime + tail) ==="
for f in logs/rtn_power2_gpu0_adaq.log logs/rtn_power2_gpu1_qlora.log; do
  if [ -f "$f" ]; then
    echo "--- $f  ($(stat -c %y "$f" | cut -c1-19))"
    tail -5 "$f"
  else
    echo "--- $f MISSING"
  fi
done
echo "=== new adapter dirs ==="
ls -d outputs/phase3_rtn/*_s{1,2,3,4,5,6,7,100,1234} 2>/dev/null | head -20
echo "=== eval done markers ==="
find results_v2_rtn -name '.done_gsm8k' 2>/dev/null | wc -l
