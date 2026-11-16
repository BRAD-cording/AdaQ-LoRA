#!/bin/bash
# F2 锚点链（修复版 2026-11-16）：等 phase3 全部收口 → 跑 baseline_bf16（3 eval seeds × mmlu+gsm8k，GPU0 串行）
# 原 bug：run_baseline_bf16.py 用 simple_evaluate(seed=) 在本机 lm_eval TypeError → 6 run 全 rc=1，
#         但旧链无条件打 DONE（假 DONE，e5143fc 提交了空锚点）。
# 修复：①run_baseline_bf16.py 改 CLI 子进程（与 run_eval_ablation.py 一致，已验证）；
#       ②本链改等 [phase3] DONE（GPU 现由 phase3 占用，bf16 排其后免抢卡）；
#       ③DONE 仅在 6/6 .done 齐时打，杜绝假 DONE。
# 注意：baseline_bf16.log 里保留旧假 DONE 行（chain_phase3 reboot 门要 grep 它放行，避免 reboot 死锁）。
exec 9>/tmp/adaq_bf16baseline.lock
flock -n 9 || { echo "[bf16-baseline] another instance holds lock, exit"; exit 0; }
cd /home/ubuntu/AdaQ-LoRA
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
LOG=logs/baseline_bf16.log

echo "[bf16-baseline] waiter(restart) started $(date -u) — 改等 [phase3] DONE（GPU 由 phase3 占用）" >> "$LOG"
# 1) 等 phase3 全部收口
while ! grep -q "\[phase3\] DONE" logs/phase3/chain_phase3.log 2>/dev/null; do
  sleep 120
done
echo "[bf16-baseline] phase3 done, starting anchors $(date -u)" >> "$LOG"

# 2) 3 eval seed × 2 任务，GPU0 串行（mmlu ~25min / gsm8k ~76min，共 ~5h）
FAIL=0
for S in 42 3407 2025; do
  for T in mmlu gsm8k; do
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com \
      "$PY" scripts/run_baseline_bf16.py --seed "$S" --task "$T" >> "$LOG" 2>&1 || FAIL=$((FAIL+1))
  done
done

# 3) 验收：6/6 .done 齐才算成功
DONE_CNT=0
for S in 42 3407 2025; do for T in mmlu gsm8k; do
  [ -f "results_v2/baseline_bf16_seed$S/.done_$T" ] && DONE_CNT=$((DONE_CNT+1))
done; done
echo "[bf16-baseline] done markers: $DONE_CNT/6 (fail this round=$FAIL) $(date -u)" >> "$LOG"

if [ "$DONE_CNT" -eq 6 ]; then
  "$PY" scripts/build_results_ledger.py >> "$LOG" 2>&1
  git add results_v2/baseline_bf16_seed42 results_v2/baseline_bf16_seed3407 results_v2/baseline_bf16_seed2025 results_ledger.csv scripts/run_baseline_bf16.py scripts/chain_baseline_bf16.sh 2>/dev/null
  git commit -m "F2: baseline_bf16 锚点行 REAL（BF16 干净基座无 adapter，3 eval seeds × mmlu+gsm8k）[auto]

- 修复 e5143fc 假 DONE：旧 run_baseline_bf16.py simple_evaluate(seed=) TypeError 全失败
- 改 CLI 子进程路径（与 run_eval_ablation.py 一致），6/6 .done 齐才提交
- 主表 BF16 锚点行补齐（review/fair-baseline.md F2、jcr-gate P0）" >> "$LOG" 2>&1 || echo "[bf16-baseline] nothing to commit" >> "$LOG"
  echo "[bf16-baseline] DONE $(date -u)" >> "$LOG"
  echo "[bf16-baseline] DONE $(date -u)" >> logs/chain_b4_seeds.log
else
  echo "[bf16-baseline] INCOMPLETE ($DONE_CNT/6) — 不打 DONE；修后重跑 chain_baseline_bf16.sh $(date -u)" >> "$LOG"
fi
