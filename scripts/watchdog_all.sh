#!/bin/bash
# watchdog_all.sh — 顶层看门狗：确保全链所有实验环节无人值守跑完。
# 填补关键缺口：@reboot cron 只在断电重启时恢复；若某 waiter 运行中被 OOM killer/意外杀死，
#   下游全链会静默卡死数天。本看门狗每 600s 巡检，「未完成 + 进程死」即重启。
# 安全性：每个被管脚本自身 flock + 幂等（adapter/results 齐则 SKIP），重启不会重复跑或撞卡。
# 防风暴：每脚本最多重启 MAXR 次（计数落 logs/.wd_restart_<name>），超限记 WARN 待人工。
# 全完成：6 个 DONE 产物齐 → 生成实验总览 → 看门狗退出。
# 启动：nohup bash scripts/watchdog_all.sh >> logs/watchdog.log 2>&1 &  （+ @reboot cron）
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
LOG=logs/watchdog.log
mkdir -p logs
exec 9>/tmp/adaq_watchdog.lock
flock -n 9 || { echo "[watchdog] 已有实例在跑，退出 $(date -u)" | tee -a "$LOG"; exit 0; }
log(){ echo "[watchdog] $(date -u '+%F %T') $*" | tee -a "$LOG"; }

SCRIPTS="phase3 bf16 b8 b5 f18 p21 p21confirm"
MAXR=5

# DONE 产物检查（返回 0=已完成）——全部用真实产物，免疫日志假 DONE 行
check_done(){
  case "$1" in
    phase3)     grep -q "\[phase3\] DONE" logs/phase3/chain_phase3.log 2>/dev/null ;;
    bf16)       [ "$(find results_v2/baseline_bf16_seed* -name '.done_*' 2>/dev/null | wc -l)" -ge 6 ] ;;
    b8)         grep -q "\[b8-audit\] DONE" logs/b8_flex_audit.log 2>/dev/null ;;
    b5)         [ "$(find results_v2_14b_chattemplate -name '.done_gsm8k_chattemplate' 2>/dev/null | wc -l)" -ge 6 ] ;;
    f18)        grep -q "\[f18-rtn\] DONE" logs/f18_rtn_eval.log 2>/dev/null ;;
    p21)        [ -f logs/p21/.done_p21 ] ;;
    p21confirm) [ -f logs/p21/.done_p21confirm ] || ! grep -q '"R1": true' results_v2_p21/summary.json 2>/dev/null ;;
    *) return 1 ;;
  esac
}
# 进程名（pgrep -f 匹配）
proc_of(){
  case "$1" in
    phase3) echo "chain_phase3.sh" ;;
    bf16)   echo "chain_baseline_bf16.sh" ;;
    b8)     echo "run_b8_flex_audit.sh" ;;
    b5)     echo "run_b5_chattemplate.sh" ;;
    f18)    echo "chain_f18_rtn.sh" ;;
    p21)    echo "chain_p21.sh" ;;
    p21confirm) echo "chain_p21_confirm.sh" ;;
  esac
}
# 重启命令（含各自日志重定向）
restart_of(){
  case "$1" in
    phase3) echo "bash scripts/chain_phase3.sh >> logs/phase3/chain_phase3.log 2>&1" ;;
    bf16)   echo "bash scripts/chain_baseline_bf16.sh >> logs/baseline_bf16.log 2>&1" ;;
    b8)     echo "bash scripts/run_b8_flex_audit.sh >> logs/b8_flex_audit.log 2>&1" ;;
    b5)     echo "bash scripts/run_b5_chattemplate.sh >> logs/b5_chattemplate.log 2>&1" ;;
    f18)    echo "bash scripts/chain_f18_rtn.sh >> logs/f18_rtn_eval.log 2>&1" ;;
    p21)    echo "bash scripts/chain_p21.sh >> logs/p21/chain_p21.log 2>&1" ;;
    p21confirm) echo "bash scripts/chain_p21_confirm.sh >> logs/p21/chain_p21_confirm.log 2>&1" ;;
  esac
}
# p21confirm 仅在 p21 完成且 R1=true 时才该启动（否则永不触发，视为"不需要"=done）
should_run(){
  case "$1" in
    p21confirm)
      [ -f logs/p21/.done_p21 ] && grep -q '"R1": true' results_v2_p21/summary.json 2>/dev/null ;;
    *) return 0 ;;
  esac
}

log "=== START 监控 [$SCRIPTS] 每600s 每脚本≤${MAXR}次重启 ==="
while true; do
  alldone=1
  for s in $SCRIPTS; do
    # 不该跑的（如 p21confirm 在 R2 时）视为已完成
    if ! should_run "$s"; then continue; fi
    if check_done "$s"; then continue; fi
    alldone=0
    if pgrep -f "$(proc_of "$s")" >/dev/null 2>&1; then continue; fi  # 活着（跑或 gate 等待）
    # 未完成 + 未活 = 早死 → 重启（封顶 MAXR）
    cf="logs/.wd_restart_${s}"
    cnt=$(cat "$cf" 2>/dev/null || echo 0)
    if [ "$cnt" -ge "$MAXR" ]; then
      log "WARN $s 已死且重启 ${cnt}≥${MAXR} 次，停重启（需人工查 $(proc_of "$s") 崩溃原因）"
      continue
    fi
    log "RESTART $s（进程死且未完成，第 $((cnt+1))/$MAXR 次重启）"
    nohup $(restart_of "$s") &
    echo $((cnt+1)) > "$cf"
    sleep 8
  done
  if [ "$alldone" -eq 1 ]; then
    log "=== 全部实验环节完成 → 生成总览 ==="
    [ -f scripts/gen_final_overview.py ] && "$PY" scripts/gen_final_overview.py 2>&1 | tee -a "$LOG"
    git add -A >/dev/null 2>&1 || true
    git commit -q -m "全实验环节完成：最终总览 [auto] $(date -u +%F)" >/dev/null 2>&1 || true
    log "=== watchdog EXIT 全完成 $(date -u) ==="
    break
  fi
  sleep 600
done
