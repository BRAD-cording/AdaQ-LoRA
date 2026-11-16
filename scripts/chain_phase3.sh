#!/bin/bash
# Phase-3 master 链：等 [bf16-baseline] DONE → D→B→E→C→A，每 phase train→eval→aggregate→ledger→commit。
# 幂等：每 phase .done 标记 + 每 run adapter SKIP + 每评测 .done_<task>；flock 防重入；@reboot 可续。
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
LOG=logs/phase3/chain_phase3.log
mkdir -p logs/phase3
exec 9>/tmp/adaq_phase3.lock
flock -n 9 || { echo "[phase3] 已有实例在跑，退出 $(date -u)" | tee -a "$LOG"; exit 0; }

log(){ echo "$@" | tee -a "$LOG"; }
log "=== [phase3] START $(date -u) ==="

# 1) 等 bf16-baseline 完成（其自身等 chain-b4-seeds DONE）
WAITMARK="logs/baseline_bf16.log"
log "[phase3] 等待 $WAITMARK 出现 [bf16-baseline] DONE ..."
while true; do
  if [ -f "$WAITMARK" ] && grep -q "\[bf16-baseline\] DONE" "$WAITMARK"; then
    log "[phase3] 检测到 bf16-baseline DONE $(date -u)"; break
  fi
  sleep 120
done

# C 依赖 R-曲线 ranks（纯 CPU，幂等）
if [ ! -f configs/rcurve/ranks_adaq_R128.json ]; then
  log "[phase3] 生成 R-曲线 ranks ..."
  "$PY" scripts/gen_rcurve_ranks.py 2>&1 | tee -a "$LOG"
fi

run_phase(){
  local P="$1" tasks="${2:-mmlu,gsm8k}"
  local dm="logs/phase3/.done_${P}"
  if [ -f "$dm" ]; then log "[phase3:$P] SKIP (.done)"; return 0; fi
  log "=== [phase3:$P] TRAIN $(date -u) ==="
  PHASE="$P" /bin/bash scripts/run_phase3_train.sh 2>&1 | tee -a "$LOG"
  log "=== [phase3:$P] EVAL tasks=$tasks $(date -u) ==="
  PHASE="$P" TASKS="$tasks" "$PY" scripts/run_eval_phase3.py 2>&1 | tee -a "$LOG"
  log "=== [phase3:$P] AGGREGATE $(date -u) ==="
  PHASE="$P" "$PY" scripts/aggregate_phase3.py 2>&1 | tee -a "$LOG"
  "$PY" scripts/build_results_ledger.py >/dev/null 2>&1 || true
  git add -A >/dev/null 2>&1 || true
  git commit -q -m "phase3:$P train+eval+aggregate [auto] $(date -u +%F)" >/dev/null 2>&1 || true
  touch "$dm"
  log "=== [phase3:$P] COMPLETE $(date -u) ==="
}

run_phase D gsm8k
run_phase B mmlu,gsm8k
run_phase E mmlu,gsm8k
run_phase C mmlu,gsm8k
run_phase A mmlu,gsm8k

# 汇总所有 phase 报告（目录映射须与 run_eval_phase3.py / aggregate_phase3.py 一致）
{
  echo "# Phase-3 总览（自动拼接）  $(date -u)"
  for P in D B E C A; do
    case "$P" in
      D) d=results_v2_phase3_noise ;;
      B) d=results_v2_phase3_main ;;
      E) d=results_v2_phase3_baselines ;;
      C) d=results_v2_phase3_rcurve ;;
      A) d=results_v2_phase3_main ;;
    esac
    f="$d/REPORT_phase3_${P}.md"
    [ -f "$f" ] && { echo; echo "-----"; cat "$f"; }
  done
} > results_v2_phase3_OVERVIEW.md 2>/dev/null || true
git add -A >/dev/null 2>&1 || true
git commit -q -m "phase3: overview [auto]" >/dev/null 2>&1 || true

log "=== [phase3] DONE $(date -u) ==="
