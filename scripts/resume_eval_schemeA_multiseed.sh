#!/usr/bin/env bash
# P1-D4 多种子评测串行链（可重入、可断电恢复）
# 1) 等当前 seed3407 评测进程（若有）结束
# 2) 若 seed3407 未满 12 个 .done_*，跑 seed3407 评测（断点续跑）
# 3) seed2025 同理
# 4) 汇总双种子 -> results_v2_ablation_multiseed/
# 5) 触发 B4 Qwen14B 训练（若尚未产出 adapter_config.json）
set -uo pipefail
ROOT="/home/ubuntu/AdaQ-LoRA"
PY="/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python"
LOCK="/tmp/adaq_eval_chain.lock"
LOG="${ROOT}/logs/chain_eval_multiseed.log"

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "[chain] 另一个 chain 持有锁，退出 $(date -u '+%F %T UTC')"
  exit 0
fi

log() { echo "[chain] $(date -u '+%F %T UTC') $*"; }

wait_eval_procs() {
  while pgrep -f "run_eval_ablation_schemeA_multiseed.py" >/dev/null 2>&1; do
    sleep 120
  done
}

count_done() { find "${ROOT}/results_v2_ablation_schemeA_s$1" -name '.done_*' 2>/dev/null | wc -l; }

run_seed() {
  local seed="$1"
  local n
  n=$(count_done "$seed")
  if [ "$n" -ge 12 ]; then
    log "seed${seed} 已完成 ${n}/12，跳过"
    return 0
  fi
  log "跑 seed${seed} 评测（当前 ${n}/12）"
  cd "$ROOT" || return 1
  SEED="$seed" setsid nohup "$PY" scripts/run_eval_ablation_schemeA_multiseed.py \
    > "logs/eval_schemeA_s${seed}.log" 2>&1 &
  local pid=$!
  log "seed${seed} 评测 pid=${pid}"
  while kill -0 "$pid" 2>/dev/null; do sleep 120; done
  n=$(count_done "$seed")
  log "seed${seed} 评测结束，done=${n}/12"
  if [ "$n" -lt 12 ]; then
    log "WARN seed${seed} 未满 12，需人工复盘（继续后续步骤）"
  fi
}

log "=== chain start ==="
log "等待现有评测进程结束（若有）"
wait_eval_procs

run_seed 3407
run_seed 2025

log "双种子评测结束，汇总"
"$PY" "${ROOT}/scripts/aggregate_ablation_multiseed.py" >> "$LOG" 2>&1 || log "WARN 汇总失败"

if [ -f "${ROOT}/outputs/adapters_qwen14b/adaq_seed42/adapter_config.json" ] && \
   [ -f "${ROOT}/outputs/adapters_qwen14b/qlora_seed42/adapter_config.json" ]; then
  log "B4 训练产物已存在，跳过训练"
else
  log "触发 B4 Qwen14B 训练"
  cd "$ROOT" || exit 1
  setsid nohup bash scripts/run_b4_qwen14b.sh > logs/b4_qwen14b_master.log 2>&1 &
  log "B4 训练 pid=$!"
fi
log "=== chain done ==="
