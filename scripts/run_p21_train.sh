#!/bin/bash
# P2-1 成本归一化秩分配训练。用法: bash scripts/run_p21_train.sh
# 4 臂 × 3 seed = 12 run（train.py α=2r，375 优化步，--no-packing）。
#   cost_norm  r_l ∝ score_l/c_l   budget512  configs/p21/ranks_cost_norm.json   （主判据臂）
#   raw_score  r_l ∝ score_l       budget512  configs/p21/ranks_raw_score.json   （= AdaQ 现状对照）
#   uniform    r_l 常数            budget512  configs/p21/ranks_uniform.json     （等秩和基线）
#   uniform_pm r_l 常数            budget576  configs/p21/ranks_uniform_pm.json  （参数匹配条件臂，触发于 cost_norm 1.0838×>1.05）
# Scheme B per-layer 守恒；双卡 2 进程一波；adapter_config.json+safetensors 齐则 SKIP（幂等）。
# F4 预算强约束/F15 α=2r 由 train.py 内部执行。预注册 docs/preregistration-p21-costnorm.md（4107552+448330a）。
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
export HF_ENDPOINT=https://hf-mirror.com
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
MODEL=/home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B
LOGDIR=logs/p21
mkdir -p "$LOGDIR"
COMMON="--max-length 1024 --batch-size 4 --grad-accum 4 --epochs 3 --dataset-size 2000 --lr 2e-4 --no-packing"
SEEDS3="42 3407 2025"

JOBS=()   # 每条 "OUTDIR|ARGS"
for s in $SEEDS3; do
  JOBS+=("outputs/p21/cost_norm_s${s}|--method adaq --seed $s --rank-scheme per_layer --ranks-file configs/p21/ranks_cost_norm.json --budget 512")
  JOBS+=("outputs/p21/raw_score_s${s}|--method adaq --seed $s --rank-scheme per_layer --ranks-file configs/p21/ranks_raw_score.json --budget 512")
  JOBS+=("outputs/p21/uniform_s${s}|--method adaq --seed $s --rank-scheme per_layer --ranks-file configs/p21/ranks_uniform.json --budget 512")
  JOBS+=("outputs/p21/uniform_pm_s${s}|--method adaq --seed $s --rank-scheme per_layer --ranks-file configs/p21/ranks_uniform_pm.json --budget 576")
done

train_one() {
  local out="$1" args="$2" gpu="$3"
  local name; name=$(basename "$out")
  local log="$LOGDIR/${name}.log"
  mkdir -p "$out"
  if [ -f "$out/adapter_config.json" ] && [ -f "$out/adapter_model.safetensors" ]; then
    echo "[SKIP] p21/$name 已完成"; return 0
  fi
  echo "[TRAIN] p21/$name gpu$gpu $(date -u)"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train.py $args $COMMON --model "$MODEL" --output-dir "$out" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$out/adapter_config.json" ]; then
    echo "[DONE] p21/$name rc=0 $(date -u)"
  else
    echo "[FAIL] p21/$name rc=$rc (log $log) $(date -u)"
  fi
  return $rc
}

n=${#JOBS[@]}; i=0
echo "=== [p21-train] $n jobs $(date -u) ==="
while [ $i -lt $n ]; do
  IFS='|' read -r out0 args0 <<< "${JOBS[$i]}"
  j=$((i+1))
  if [ $j -lt $n ]; then
    IFS='|' read -r out1 args1 <<< "${JOBS[$j]}"
    train_one "$out0" "$args0" 0 & p0=$!
    train_one "$out1" "$args1" 1 & p1=$!
    wait $p0 || true; wait $p1 || true
    i=$((i+2))
  else
    train_one "$out0" "$args0" 0 || true
    i=$((i+1))
  fi
done
echo "=== [p21-train] 全部派发完 $(date -u) ==="
ls -1d outputs/p21/*/adapter_config.json 2>/dev/null | wc -l
