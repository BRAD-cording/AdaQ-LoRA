#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if tmux has-session -t adaq_ablation_schemeA 2>/dev/null; then
  echo "[resume] tmux adaq_ablation_schemeA 已存在，跳过"
  exit 0
fi
tmux new-session -d -s adaq_ablation_schemeA "bash ${PROJECT_ROOT}/scripts/run_ablation_schemeA.sh | tee ${PROJECT_ROOT}/logs/ablation_schemeA_master.log"
echo "[resume] 已启动 tmux adaq_ablation_schemeA"
