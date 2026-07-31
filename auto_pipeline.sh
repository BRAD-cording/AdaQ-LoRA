#!/usr/bin/env bash
# Auto-pipeline: Wait for training to complete, then run eval + aggregate
# Runs in background, automatically proceeds to steps 4-6

set -euo pipefail
cd /mnt/AdaQ-LoRA

LOG=logs/auto_pipeline.log
echo "$(date '+%H:%M:%S') Auto-pipeline started, waiting for batch_train to finish..." | tee "$LOG"

# Wait for training to complete
while pgrep -f "batch_train" > /dev/null 2>&1; do
    sleep 30
done

echo "$(date '+%H:%M:%S') Training complete! Starting evaluation..." | tee -a "$LOG"

# Count completed adapters with epochs=3
ADAPTER_COUNT=0
for f in outputs/adapters/*/training_stats.json; do
    ep=$(python3 -c "import json; print(json.load(open('$f')).get('epochs',0))" 2>/dev/null || echo 0)
    if [ "$ep" = "3" ]; then
        ADAPTER_COUNT=$((ADAPTER_COUNT + 1))
    fi
done
echo "$(date '+%H:%M:%S') Found $ADAPTER_COUNT fully trained adapters" | tee -a "$LOG"

if [ "$ADAPTER_COUNT" -lt 4 ]; then
    echo "$(date '+%H:%M:%S') ERROR: Need at least 4 adapters. Found $ADAPTER_COUNT" | tee -a "$LOG"
    exit 1
fi

# List all trained adapters
echo "" | tee -a "$LOG"
echo "=== Trained Adapters ===" | tee -a "$LOG"
for f in outputs/adapters/*/training_stats.json; do
    dir=$(dirname "$f")
    method=$(basename "$dir")
    ep=$(python3 -c "import json; print(json.load(open('$f')).get('epochs','?'))" 2>/dev/null)
    vram=$(python3 -c "import json; print(json.load(open('$f')).get('peak_vram_gb','?'))" 2>/dev/null)
    echo "  $method: epochs=$ep, VRAM=${vram}GB" | tee -a "$LOG"
done

# Step 4: Run evaluation
echo "" | tee -a "$LOG"
echo "$(date '+%H:%M:%S') === Step 4: Evaluation ===" | tee -a "$LOG"
export QUICK=1
export QUICK_LIMIT=25
export EVAL_MODE=adapter
bash scripts/run_eval_batch.sh 2>&1 | tee -a "$LOG"

# Step 5: Aggregate results
echo "" | tee -a "$LOG"
echo "$(date '+%H:%M:%S') === Step 5: Aggregation ===" | tee -a "$LOG"
python3 scripts/aggregate_results.py --baseline qlora --out results/summary.json 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "$(date '+%H:%M:%S') === Pipeline Complete! ===" | tee -a "$LOG"
echo "Results: results/summary.json" | tee -a "$LOG"
