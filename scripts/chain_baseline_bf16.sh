#!/bin/bash
# F2 后续链：等 B4 三种子链收口 → 跑 baseline_bf16 锚点（3 eval seeds × mmlu+gsm8k，GPU0 串行）
# 幂等：flock /tmp/adaq_bf16baseline.lock；.done_<task> 跳过已完成
exec 9>/tmp/adaq_bf16baseline.lock
flock -n 9 || { echo "[bf16-baseline] another instance holds lock, exit"; exit 0; }
cd /home/ubuntu/AdaQ-LoRA
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
LOG=logs/baseline_bf16.log

echo "[bf16-baseline] waiter started $(date -u)" >> "$LOG"
# 1) 等 B4 链 DONE（哨兵标记与 pwsh-3c1a7ed157e94596 一致）
while ! grep -q "chain-b4-seeds] DONE" logs/chain_b4_seeds.log 2>/dev/null; do
  sleep 120
done
echo "[bf16-baseline] b4 chain done, starting anchors $(date -u)" >> "$LOG"

# 2) 3 个 eval seed × 2 任务，GPU0 串行（每 mmlu ~25min / gsm8k ~76min，共 ~5h）
for S in 42 3407 2025; do
  for T in mmlu gsm8k; do
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com \
      "$PY" scripts/run_baseline_bf16.py --seed "$S" --task "$T" >> "$LOG" 2>&1 \
      && echo "[bf16-baseline] ok seed=$S task=$T $(date -u)" >> "$LOG" \
      || echo "[bf16-baseline] FAIL seed=$S task=$T rc=$? $(date -u)" >> "$LOG"
  done
done

# 3) 刷新台账并提交
"$PY" scripts/build_results_ledger.py >> "$LOG" 2>&1
git add results_v2/baseline_bf16_seed42 results_v2/baseline_bf16_seed3407 results_v2/baseline_bf16_seed2025 results_ledger.csv scripts/run_baseline_bf16.py scripts/chain_baseline_bf16.sh 2>/dev/null
git commit -m "F2: baseline_bf16 锚点行（BF16 干净基座无 adapter，3 eval seeds × mmlu+gsm8k）

- 主表 12 run 一直缺无-adapter 锚点（review/fair-baseline.md F2、jcr-gate P0）
- NF4 表已示四个微调臂 MMLU 全部低于 baseline_nf4(.6364)，缺锚点则『四方法打平』与『四方法同样无用』不可区分
- 产物: results_v2/baseline_bf16_seed{42,3407,2025}/，台账已刷新" >> "$LOG" 2>&1 || echo "[bf16-baseline] nothing to commit" >> "$LOG"

echo "[bf16-baseline] DONE $(date -u)" >> "$LOG"
echo "[bf16-baseline] DONE $(date -u)" >> logs/chain_b4_seeds.log
