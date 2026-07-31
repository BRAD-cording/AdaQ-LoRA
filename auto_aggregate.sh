#!/usr/bin/env bash
# Wait for evaluation to complete, then run aggregation
cd /mnt/AdaQ-LoRA

LOG=logs/auto_aggregate.log
echo "$(date '+%H:%M:%S') Waiting for evaluation to complete..." | tee "$LOG"

# Wait for eval process to finish
while pgrep -f "run_eval_batch" > /dev/null 2>&1 || pgrep -f "lm_eval" > /dev/null 2>&1; do
    sleep 30
done

echo "$(date '+%H:%M:%S') Evaluation complete! Starting aggregation..." | tee -a "$LOG"

# Run aggregation
python3 scripts/aggregate_results.py --baseline qlora --out results/summary.json 2>&1 | tee -a "$LOG"

echo "$(date '+%H:%M:%S') Pipeline complete! Results in results/summary.json" | tee -a "$LOG"
