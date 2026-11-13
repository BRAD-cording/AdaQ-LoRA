#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python}"
if tmux has-session -t adaq_ablation_eval_schemeA 2>/dev/null; then
  echo "[resume] tmux adaq_ablation_eval_schemeA 已存在，跳过"
  exit 0
fi
tmux new-session -d -s adaq_ablation_eval_schemeA \
  "OUT_DIR=${PROJECT_ROOT}/results_v2_ablation_schemeA GPUS=0,1 ${PY} ${PROJECT_ROOT}/scripts/run_eval_ablation_schemeA.py | tee ${PROJECT_ROOT}/logs/ablation_eval_schemeA_master.log"
echo "[resume] 已启动 tmux adaq_ablation_eval_schemeA"
