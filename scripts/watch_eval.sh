#!/usr/bin/env bash
# 评测看门狗：主驱动退出后，检查是否有缺失的 run，自动补跑；直到全部完成。
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
OUT_DIR="${OUT_DIR:-results_v2}"
LOG="${LOG:-logs/watchdog.log}"
echo "[$(date '+%F %T')] watchdog start" >> "$LOG"

check() {
  # 参数: method seed
  local m="$1" s="$2"
  local d="${OUT_DIR}/${m}_seed${s}"
  [[ -f "${d}/.done_main" ]] && return 0
  return 1
}

for round in 1 2 3 4 5 6 7 8; do
  sleep 600
  missing=()
  for m in qlora dora adalora adaq; do
    for s in 42 3407 2025; do
      check "$m" "$s" || missing+=("$m:$s")
    done
  done
  echo "[$(date '+%F %T')] round=$round missing=${missing[*]:-NONE}" >> "$LOG"
  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "[$(date '+%F %T')] all done" >> "$LOG"
    exit 0
  fi
  # 如果没有 lm_eval 在跑，就补跑缺失部分（两卡平分）
  if ! pgrep -f lm_eval > /dev/null 2>&1; then
    echo "[$(date '+%F %T')] no lm_eval running, restarting missing" >> "$LOG"
    gpu=0
    for item in "${missing[@]}"; do
      m="${item%%:*}"; s="${item##*:}"
      OUT_DIR="${OUT_DIR}" nohup bash scripts/eval_worker.sh "$gpu" "${m}" "${s}" \
        >> "${OUT_DIR}/resume_${m}_${s}.log" 2>&1 &
      gpu=$((1 - gpu))
    done
  fi
done
echo "[$(date '+%F %T')] watchdog exit (timeout)" >> "$LOG"
