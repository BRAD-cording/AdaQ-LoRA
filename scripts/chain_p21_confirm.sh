#!/bin/bash
# P2-1 R1-confirm 链：仅当主跑 R1=true 触发（预注册 §10.5）。
# gate: logs/p21/.done_p21 存在 且 results_v2_p21/summary.json "R1": true。
# 跑 cost_norm+uniform_pm × 新3seed{1337,7,2024}=6run → eval → aggregate_p21_confirm → commit。
# R2 阴性则本链永不启动（gate 不满足；watchdog should_run 亦跳过）。
# 启动：nohup bash scripts/chain_p21_confirm.sh >> logs/p21/chain_p21_confirm.log 2>&1 &
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
LOGDIR=logs/p21
mkdir -p "$LOGDIR"
LOG=$LOGDIR/chain_p21_confirm.log
exec 9>/tmp/adaq_p21_confirm.lock
flock -n 9 || { echo "[p21-confirm] 已有实例在跑，退出 $(date -u)" | tee -a "$LOG"; exit 0; }
log(){ echo "$@" | tee -a "$LOG"; }
log "=== [p21-confirm] START $(date -u) ==="

# gate：等主跑 P2-1 完成
log "[p21-confirm] 等待主跑 P2-1 完成（logs/p21/.done_p21）..."
while [ ! -f logs/p21/.done_p21 ]; do sleep 300; done
log "[p21-confirm] $(date -u) 主跑完成，检查 R1 判定..."

# 仅 R1=true 才触发 confirm；R2 阴性则记录并退出（不浪费 GPU）
if ! grep -q '"R1": true' results_v2_p21/summary.json 2>/dev/null; then
  log "[p21-confirm] $(date -u) R1≠true（R2 阴性或数据缺）→ confirm 不触发，预注册 §10.5 规定阴性不扩种子。退出。"
  touch logs/p21/.done_p21confirm   # 标记"已决策不跑"，watchdog 不再重启
  exit 0
fi

DM="logs/p21/.done_p21confirm"
if [ -f "$DM" ]; then
  log "[p21-confirm] SKIP (.done_p21confirm 已存在)"
else
  log "=== [p21-confirm] TRAIN cost_norm+uniform_pm × {1337,7,2024} $(date -u) ==="
  SEEDS="1337 7 2024" ARMS="cost_norm uniform_pm" /bin/bash scripts/run_p21_train.sh 2>&1 | tee -a "$LOG"
  log "=== [p21-confirm] EVAL gsm8k $(date -u) ==="
  SEEDS="1337,7,2024" ARMS="cost_norm,uniform_pm" TASKS="gsm8k" "$PY" scripts/run_eval_p21.py 2>&1 | tee -a "$LOG"
  log "=== [p21-confirm] AGGREGATE 6-seed 确证判定 $(date -u) ==="
  "$PY" scripts/aggregate_p21_confirm.py 2>&1 | tee -a "$LOG"
  "$PY" scripts/build_results_ledger.py >/dev/null 2>&1 || true
  git add -A >/dev/null 2>&1 || true
  git commit -q -m "P2-1 R1-confirm 6-seed 确证（cost_norm+uniform_pm ×{1337,7,2024}）[auto] $(date -u +%F)" >/dev/null 2>&1 || true
  touch "$DM"
fi
log "=== [p21-confirm] DONE $(date -u) ==="
