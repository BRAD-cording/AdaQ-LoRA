#!/bin/bash
# Phase-3 训练分发器。用法: PHASE=D|B|E|C|A bash scripts/run_phase3_train.sh
# 正典配方（统一所有臂，修 F1/F2/F3 三表三口径）：
#   seq1024 + batch4 + accum4 + --no-packing + 3ep + 2000样本 + lr2e-4 → 375 优化步
# 双卡 2 进程一波，adapter_config.json+safetensors 齐则 SKIP（幂等）。
# F4 预算强约束/F15 α=2r/F25 adalora 动态调度 均由 train.py 内部执行。
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
export HF_ENDPOINT=https://hf-mirror.com
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
MODEL=/home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B
LOGDIR=logs/phase3
mkdir -p "$LOGDIR"
PHASE="${PHASE:?need PHASE=D|B|E|C|A}"
COMMON="--max-length 1024 --batch-size 4 --grad-accum 4 --epochs 3 --dataset-size 2000 --lr 2e-4 --no-packing"
SEEDS3="42 3407 2025"
SEEDS2="42 3407"

JOBS=()   # 每条 "OUTDIR|ARGS"
case "$PHASE" in
  D)  # 噪声底：product 配置 × 3 seed × 3 repeat = 9 run（同配置多次→分离种子间/非确定性方差）
    for s in $SEEDS3; do for r in 0 1 2; do
      JOBS+=("outputs/phase3_noise/product_s${s}_r${r}|--method adaq --seed $s --rank-scheme per_module_type --ranks-file configs/ablation_ranks/ranks_product.json --budget 512")
    done; done ;;
  B)  # adalora 动态调度（F25），直接落 phase3_main 作主表 adalora 臂
    for s in $SEEDS3; do
      JOBS+=("outputs/phase3_main/adalora_s${s}|--method adalora --seed $s --base-rank 8 --budget 512")
    done ;;
  E)  # 新基线：LoftQ + 参数匹配 QLoRA(r9,budget576)
    for s in $SEEDS3; do
      JOBS+=("outputs/phase3_baselines/loftq_s${s}|--method qlora_loftq --seed $s --base-rank 8 --budget 512")
      JOBS+=("outputs/phase3_baselines/qlorapm_s${s}|--method qlora --seed $s --base-rank 9 --budget 576")
    done ;;
  C)  # R-曲线最小集：R∈{128,512,1024}×{adaq(per_layer),qlora}×seed{42,3407}=12run
    for R in 128 512 1024; do for s in $SEEDS2; do
      JOBS+=("outputs/phase3_rcurve/adaq_R${R}_s${s}|--method adaq --seed $s --rank-scheme per_layer --ranks-file configs/rcurve/ranks_adaq_R${R}.json --budget $R")
      JOBS+=("outputs/phase3_rcurve/qlora_R${R}_s${s}|--method qlora --seed $s --base-rank $((R/64)) --budget $R")
    done; done ;;
  A)  # 主表重训：qlora/dora/adaq ×3seed（adalora 已由 B 落 phase3_main，SKIP）
    for s in $SEEDS3; do
      JOBS+=("outputs/phase3_main/qlora_s${s}|--method qlora --seed $s --base-rank 8 --budget 512")
      JOBS+=("outputs/phase3_main/dora_s${s}|--method dora --seed $s --base-rank 8 --budget 512")
      JOBS+=("outputs/phase3_main/adaq_s${s}|--method adaq --seed $s --rank-scheme per_module_type --ranks-file ranks.json --budget 512")
      JOBS+=("outputs/phase3_main/adalora_s${s}|--method adalora --seed $s --base-rank 8 --budget 512")
    done ;;
  *) echo "unknown PHASE=$PHASE"; exit 2 ;;
esac

train_one() {
  local out="$1" args="$2" gpu="$3"
  local name; name=$(basename "$out")
  local log="$LOGDIR/${PHASE}_${name}.log"
  mkdir -p "$out"
  if [ -f "$out/adapter_config.json" ] && [ -f "$out/adapter_model.safetensors" ]; then
    echo "[SKIP] $PHASE/$name 已完成"; return 0
  fi
  echo "[TRAIN] $PHASE/$name gpu$gpu $(date -u)"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train.py $args $COMMON --model "$MODEL" --output-dir "$out" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$out/adapter_config.json" ]; then
    echo "[DONE] $PHASE/$name rc=0 $(date -u)"
  else
    echo "[FAIL] $PHASE/$name rc=$rc (log $log) $(date -u)"
  fi
  return $rc
}

n=${#JOBS[@]}; i=0
echo "=== [phase3-train:$PHASE] $n jobs $(date -u) ==="
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
echo "=== [phase3-train:$PHASE] 全部派发完 $(date -u) ==="
ls -1d outputs/phase3_*/*/adapter_config.json 2>/dev/null | wc -l
