#!/bin/bash
# P2-1 master 链：等所有前序 GPU 任务真实产物完成（净空双卡）→ gen(skip)→train→eval→aggregate→ledger→commit。
# gate 用真实产物（免疫假 DONE，吸取 F18 gate bug 教训 b9a34bd）：
#   phase3: "[phase3] DONE" in logs/phase3/chain_phase3.log
#   bf16  : results_v2/baseline_bf16_seed*/.done_* 计数 ≥6
#   B8    : "[b8-audit] DONE" in logs/b8_flex_audit.log
#   B5    : results_v2_14b_chattemplate/**/.done_gsm8k_chattemplate 计数 ≥6
#   F18   : "[f18-rtn] DONE" in logs/f18_rtn_eval.log
# 幂等：.done_p21 标记 + 每 run adapter SKIP + 每评测 .done_<task>；flock 防重入；@reboot 可续。
# 启动：nohup bash scripts/chain_p21.sh >> logs/p21/chain_p21.log 2>&1 &
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
LOGDIR=logs/p21
mkdir -p "$LOGDIR"
LOG=$LOGDIR/chain_p21.log
exec 9>/tmp/adaq_p21.lock
flock -n 9 || { echo "[p21] 已有实例在跑，退出 $(date -u)" | tee -a "$LOG"; exit 0; }
log(){ echo "$@" | tee -a "$LOG"; }
log "=== [p21] START $(date -u) ==="

gate_ready(){
  local ok=1 pending=""
  grep -q "\[phase3\] DONE" logs/phase3/chain_phase3.log 2>/dev/null || { ok=0; pending="$pending phase3"; }
  local nbf; nbf=$(find results_v2/baseline_bf16_seed* -name '.done_*' 2>/dev/null | wc -l)
  [ "$nbf" -ge 6 ] || { ok=0; pending="$pending bf16($nbf/6)"; }
  grep -q "\[b8-audit\] DONE" logs/b8_flex_audit.log 2>/dev/null || { ok=0; pending="$pending b8"; }
  local nb5; nb5=$(find results_v2_14b_chattemplate -name '.done_gsm8k_chattemplate' 2>/dev/null | wc -l)
  [ "$nb5" -ge 6 ] || { ok=0; pending="$pending b5($nb5/6)"; }
  grep -q "\[f18-rtn\] DONE" logs/f18_rtn_eval.log 2>/dev/null || { ok=0; pending="$pending f18"; }
  GATE_PENDING="$pending"
  return $((1-ok))
}

log "[p21] 等待所有前序 GPU 任务完成（净空双卡）..."
while ! gate_ready; do
  log "[p21] $(date -u '+%F %T') pending:$GATE_PENDING — sleep 300"; sleep 300
done
log "[p21] $(date -u) 前序全完成 → 双卡净空，开始 P2-1"

DM="logs/p21/.done_p21"
if [ -f "$DM" ]; then
  log "[p21] SKIP (.done_p21 已存在)"; 
else
  # ranks 已由 gen_p21_ranks.py 生成（configs/p21/ranks_{arm}.json）；缺则补跑（纯 CPU 幂等）
  if [ ! -f configs/p21/ranks_cost_norm.json ]; then
    log "[p21] 生成四臂 ranks ..."; "$PY" scripts/gen_p21_ranks.py 2>&1 | tee -a "$LOG"
  fi
  log "=== [p21] TRAIN（4臂×3seed=12run）$(date -u) ==="
  /bin/bash scripts/run_p21_train.sh 2>&1 | tee -a "$LOG"
  log "=== [p21] EVAL（mmlu,gsm8k）$(date -u) ==="
  TASKS="mmlu,gsm8k" "$PY" scripts/run_eval_p21.py 2>&1 | tee -a "$LOG"
  log "=== [p21] AGGREGATE + 预注册判定 $(date -u) ==="
  "$PY" scripts/aggregate_p21.py 2>&1 | tee -a "$LOG"
  "$PY" scripts/build_results_ledger.py >/dev/null 2>&1 || true
  git add -A >/dev/null 2>&1 || true
  git commit -q -m "P2-1 成本归一化：4臂×3seed train+eval+aggregate+预注册判定 [auto] $(date -u +%F)" >/dev/null 2>&1 || true
  touch "$DM"
fi
log "=== [p21] DONE $(date -u) ==="
