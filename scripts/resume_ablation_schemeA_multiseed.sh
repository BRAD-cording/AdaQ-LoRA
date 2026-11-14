#!/usr/bin/env bash
# P1-D4: 方案A消融多种子训练+评测 tmux 幂等启动器（串行版）
# 串行：先 seed3407 完整训完 6 策略，再 seed2025。避免两会话同时抢双卡 OOM。
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS=("3407" "2025")

for s in "${SEEDS[@]}"; do
  tmux_sess="adaq_ablation_s${s}"
  if tmux has-session -t "${tmux_sess}" 2>/dev/null; then
    echo "[resume] ${tmux_sess} 已存在，跳过"
    continue
  fi
  # 等上一个种子训完再启动（除第一个外）
  if [ "$s" != "${SEEDS[0]}" ]; then
    prev="${SEEDS[0]}"
    # 幂等等待：轮询上一个种子的完成标记（6 策略 adapter_config.json 全齐）
    echo "[resume] 等待 seed${prev} 训练完成再启动 seed${s}..."
    while true; do
      done_prev=$(ls "${PROJECT_ROOT}/outputs/ablation_adapters_schemeA_s${prev}"/*/adapter_config.json 2>/dev/null | wc -l)
      if [ "$done_prev" -ge 6 ]; then
        echo "[resume] seed${prev} 已训完 ${done_prev}/6，启动 seed${s}"
        break
      fi
      sleep 60
    done
  fi
  echo "[resume] 启动训练 seed=${s}"
  tmux new-session -d -s "${tmux_sess}" "SEED=${s} bash ${PROJECT_ROOT}/scripts/run_ablation_schemeA_multiseed.sh | tee ${PROJECT_ROOT}/logs/ablation_schemeA_s${s}_master.log"
done

echo "[resume] 训练会话已启动"
