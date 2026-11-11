#!/bin/bash
# 收口看门狗：无人值守把交付闭合。
#   有 lm_eval 在跑 -> 等
#   没 lm_eval     -> 跑 finalize_eval.py（闸门 PASS 才生成 FINAL_REPORT.md）
#   rc!=0          -> 若驱动进程已死且无 GPU 任务，才触发幂等恢复；否则继续等
# 关键：canary 与放量之间的空档会拿到 INCOMPLETE(rc=2)，这**不是失败**，继续等即可。
set -o pipefail
cd /home/ubuntu/AdaQ-LoRA || exit 1
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
LOG=logs/finalize_watch.log
STATUS=logs/finalize_watch.status
export NLTK_DATA=/home/ubuntu/nltk_data

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

driver_alive(){
  pgrep -f "repair_safety_v2.sh" > /dev/null 2>&1 && return 0
  pgrep -f "run_eval_missing.py" > /dev/null 2>&1 && return 0
  pgrep -f "repair_all.sh" > /dev/null 2>&1 && return 0
  return 1
}

log "FINALIZE-WATCH START rounds<=200 interval=180s"
recovered=0
for i in $(seq 1 200); do
  if pgrep -f "lm_eval --model" > /dev/null 2>&1; then
    log "r$i 有 lm_eval 在跑 -> 等 180s"
    sleep 180
    continue
  fi
  log "r$i 无 lm_eval -> 收口"
  "$PY" scripts/finalize_eval.py >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    log "r$i FINAL_REPORT 生成成功 rc=0"
    log "FINALIZE-WATCH DONE"
    exit 0
  fi
  if driver_alive; then
    log "r$i 收口未通过 rc=$rc（驱动仍在跑，属预期空档）-> 等 180s"
  elif [ "$recovered" -lt 2 ]; then
    recovered=$((recovered+1))
    log "r$i 收口未通过 rc=$rc 且驱动已死 -> 触发幂等恢复 #$recovered"
    bash scripts/resume_repair.sh >> "$LOG" 2>&1
    log "r$i 恢复脚本已执行，等 120s"
  else
    log "r$i 收口未通过 rc=$rc，恢复已用尽 -> 只等不复活（避免反复烧卡）"
  fi
  sleep 180
done
log "FINALIZE-WATCH TIMEOUT 200 轮"
exit 1
