#!/usr/bin/env bash
# Combined evaluation + aggregation script
# Runs after batch_train.py completes
# Usage: bash post_train_eval.sh

set -euo pipefail
cd /mnt/AdaQ-LoRA

echo "============================================================"
echo "  AdaQ-LoRA Post-Training: Evaluation + Aggregation"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Check training is complete
if pgrep -f "batch_train" > /dev/null 2>&1; then
    echo "ERROR: batch_train still running. Wait for training to complete first."
    exit 1
fi

# Count completed adapters
ADAPTER_COUNT=$(find outputs/adapters -name "adapter_config.json" | wc -l)
echo "Found ${ADAPTER_COUNT} trained adapters"

if [ "$ADAPTER_COUNT" -lt 4 ]; then
    echo "ERROR: Need at least 4 adapters (1 per method). Found $ADAPTER_COUNT"
    exit 1
fi

# List all adapters
echo ""
echo "=== Trained Adapters ==="
for f in outputs/adapters/*/adapter_config.json; do
    dir=$(dirname "$f")
    method=$(basename "$dir")
    stats_file="$dir/training_stats.json"
    if [ -f "$stats_file" ]; then
        epochs=$(python3 -c "import json; print(json.load(open('$stats_file')).get('epochs','?'))" 2>/dev/null)
        vram=$(python3 -c "import json; print(json.load(open('$stats_file')).get('peak_vram_gb','?'))" 2>/dev/null)
        echo "  ✅ $method (epochs=$epochs, VRAM=${vram}GB)"
    else
        echo "  ⚠️  $method (no stats)"
    fi
done

# Step 4: Run evaluation (QUICK mode, adapter mode, MMLU+GSM8K only)
echo ""
echo "=== Step 4: Evaluation ==="
echo "Mode: adapter (4-bit base + PEFT)"
echo "Tasks: MMLU(5-shot) + GSM8K(4-shot)"
echo "Quick limit: 25 samples per task"
echo ""

export QUICK=1
export QUICK_LIMIT=25
export EVAL_MODE=adapter
bash scripts/run_eval_batch.sh 2>&1

# Step 5: Aggregate results
echo ""
echo "=== Step 5: Aggregation ==="
python3 scripts/aggregate_results.py --baseline qlora --out results/summary.json 2>&1

echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "  Results: results/summary.json"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
