#!/usr/bin/env bash
# P1-B4 收尾链：等 Qwen2.5-14B 训练产出 2 个 adapter → 跑 B4 评测(adaq/qlora × mmlu/gsm8k)
# 可重入、可断电恢复；flock 防重入
set -uo pipefail
ROOT="/home/ubuntu/AdaQ-LoRA"
PY="/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python"
LOCK="/tmp/adaq_b4_finalize.lock"
LOG="${ROOT}/logs/b4_finalize.log"
ADIR="${ROOT}/outputs/adapters_qwen14b"

exec 9>"${LOCK}"
if ! flock -n 9; then echo "[b4fin] 锁被占用，退出 $(date -u '+%F %T UTC')"; exit 0; fi
log() { echo "[b4fin] $(date -u '+%F %T UTC') $*"; }

has_adapter() { [ -f "${ADIR}/$1/adapter_config.json" ] && [ -f "${ADIR}/$1/adapter_model.safetensors" ]; }

log "=== b4 finalize start ==="
# 1) 等训练完成（最多等 18h，每 5min 轮询）
deadline=$(( $(date +%s) + 18*3600 ))
while true; do
  if has_adapter adaq_seed42 && has_adapter qlora_seed42; then
    log "两个 adapter 就绪"
    break
  fi
  # 训练进程与 tmux 都消失且 adapter 不齐 => 训练失败/未启动
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "ERROR 等待训练超时，adapter 缺失：$(ls -d ${ADIR}/* 2>/dev/null | tr '\n' ' ')"
    exit 1
  fi
  sleep 300
done

# 2) 跑评测（幂等：.done_<task> 已存在会 SKIP）
log "启动 B4 评测"
cd "$ROOT" || exit 1
"$PY" scripts/run_eval_b4_qwen14b.py >> "${LOG}" 2>&1
rc=$?
log "B4 评测退出 rc=${rc}"
echo "done=$(ls ${ROOT}/results_v2_qwen14b/*/.done_* 2>/dev/null | wc -l)/4" >> "${LOG}"
log "=== b4 finalize done ==="
