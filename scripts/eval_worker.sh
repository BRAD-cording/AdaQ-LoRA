#!/usr/bin/env bash
# 单卡评测 worker：GPU + 方法列表 + 种子列表
set -uo pipefail
GPU_ID="$1"; shift
METHODS="$1"; shift
SEEDS="$1"; shift
cd /home/ubuntu/AdaQ-LoRA
exec env GPU="${GPU_ID}" METHODS="${METHODS}" SEEDS="${SEEDS}" \
  OUT_DIR="${OUT_DIR:-results_v2}" LIMIT="${LIMIT:-}" TASKSETS="${TASKSETS:-main}" \
  bash scripts/run_eval_unified.sh
