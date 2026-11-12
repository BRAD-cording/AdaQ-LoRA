#!/usr/bin/env bash
# 断电自动恢复：@reboot cron 调用。在 tmux 里跑 run_ablation.sh，幂等（已完成策略自动跳过）。
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="adaq_ablation"

# 若 tmux 会话已存在且存活，说明训练仍在跑，不重复启动
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[resume_ablation] tmux 会话 ${SESSION} 已存在，跳过（训练仍在进行）"
  exit 0
fi

# 启动新 tmux 会话，跑 run_ablation.sh
echo "[resume_ablation] 启动 tmux 会话 ${SESSION} $(date '+%F %T')"
tmux new-session -d -s "${SESSION}" "bash ${PROJECT_ROOT}/scripts/run_ablation.sh 2>&1 | tee ${PROJECT_ROOT}/logs/ablation_master.log"
echo "[resume_ablation] 已启动，日志 logs/ablation_master.log"
