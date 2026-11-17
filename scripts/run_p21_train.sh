#!/bin/bash
# P2-1 成本归一化秩分配训练（参数化）。
# 主跑: bash scripts/run_p21_train.sh   （SEEDS="42 3407 2025" ARMS="cost_norm raw_score uniform uniform_pm" = 12 run）
# confirm: SEEDS="1337 7 2024" ARMS="cost_norm uniform_pm" bash scripts/run_p21_train.sh  （6 run）
# train.py α=2r，375 优化步，--no-packing；Scheme B per-layer 守恒；双卡 2 进程一波；adapter 齐则 SKIP（幂等）。
# F4 预算强约束/F15 α=2r 由 train.py 内部执行。预注册 4107552+448330a+§10.5。
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
export HF_ENDPOINT=https://hf-mirror.com
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
MODEL=/home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B
LOGDIR=logs/p21
mkdir -p "$LOGDIR"
COMMON="--max-length 1024 --batch-size 4 --grad-accum 4 --epochs 3 --dataset-size 2000 --lr 2e-4 --no-packing"
SEEDS="${SEEDS:-42 3407 2025}"
ARMS="${ARMS:-cost_norm raw_score uniform uniform_pm}"
# 每臂预算（uniform_pm=576 参数匹配条件臂，余=512）
budget_of(){ case "$1" in uniform_pm) echo 576 ;; *) echo 512 ;; esac; }

JOBS=()   # 每条 "OUTDIR|ARGS"
for s in $SEEDS; do
  for arm in $ARMS; do
    bud=$(budget_of "$arm")
    JOBS+=("outputs/p21/${arm}_s${s}|--method adaq --seed $s --rank-scheme per_layer --ranks-file configs/p21/ranks_${arm}.json --budget $bud")
  done
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
echo "=== [p21-train] $n jobs SEEDS=[$SEEDS] ARMS=[$ARMS] $(date -u) ==="
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
