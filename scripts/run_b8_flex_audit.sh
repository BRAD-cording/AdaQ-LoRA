#!/usr/bin/env bash
# run_b8_flex_audit.sh — B8（JCR 阻断项）：14B GSM8K flex 抽取器人工审计的采样阶段。
#
# ★ 重要更正：B8 不是零 GPU 任务。全仓库零 samples_*.jsonl（所有评测从未用 --log_samples），
#   原始生成文本已丢弃 → 必须带 --log_samples 重跑才能审计 flex 抽取是否捕获模型意图。
#   本脚本逐字复刻 run_eval_b4_qwen14b_seeds.py 的 14B NF4 评测命令 + --limit 400 --log_samples。
#
# 审计 adaq_seed42 + qlora_seed42 两臂（+8.87pt flex 信号 = adaq−qlora，必须双臂查差别抽取偏差）。
# 用 GPU1（bf16 baseline 用 GPU0，避免争用）；gate 在 phase3 DONE 后启动。
#
# 启动：nohup bash scripts/run_b8_flex_audit.sh >> logs/b8_flex_audit.log 2>&1 &
# 产出：results_v2_audit_flex/{adaq,qlora}_seed42/samples_gsm8k_*.jsonl + flex_audit_worksheet_*.tsv
# 后续：人工双标注 worksheet（annotA/annotB 列填 Y/N/UNCLEAR）→ audit_flex_extract.py kappa 算 κ≥0.95。
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
LOCK=/tmp/adaq_b8_flex.lock
exec 9>"$LOCK"; flock 9
BASE=/home/ubuntu/AdaQ-LoRA
MODEL=$BASE/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
ADIR=$BASE/outputs/adapters_qwen14b
OUTROOT=$BASE/results_v2_audit_flex
P3LOG=$BASE/logs/phase3/chain_phase3.log
LOG=$BASE/logs/b8_flex_audit.log
mkdir -p "$OUTROOT" "$BASE/logs"
export HF_HUB_OFFLINE=1 NLTK_DATA=/home/ubuntu/nltk_data HF_ENDPOINT=https://hf-mirror.com

# 等 phase3 完成（GPU 释放）
while ! grep -q "\[phase3\] DONE" "$P3LOG" 2>/dev/null; do
  echo "[b8] $(date -u '+%F %T') waiting for phase3 DONE ..."; sleep 300
done
echo "[b8] $(date -u '+%F %T') phase3 done → start 14B flex audit sampling (adaq+qlora seed42, 400 docs each, GPU1)"

for ARM in adaq qlora; do
  ADAPTER=$ADIR/${ARM}_seed42
  if [ ! -f "$ADAPTER/adapter_config.json" ]; then echo "[b8] adapter missing $ADAPTER, skip"; continue; fi
  OUT=$OUTROOT/${ARM}_seed42
  mkdir -p "$OUT"
  if compgen -G "$OUT/samples_gsm8k_*.jsonl" >/dev/null 2>&1; then echo "[b8] $ARM samples exist, skip"; continue; fi
  echo "[b8] $(date -u '+%F %T') START $ARM gsm8k --limit 400 --log_samples"
  CUDA_VISIBLE_DEVICES=1 $PY -m lm_eval --model hf \
    --model_args pretrained=$MODEL,load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_compute_dtype=bfloat16,bnb_4bit_use_double_quant=True,dtype=bfloat16,peft=$ADAPTER \
    --tasks gsm8k --num_fewshot 4 --batch_size auto --device cuda:0 --seed 42 \
    --limit 400 --log_samples --output_path "$OUT" >> "$LOG" 2>&1
  rc=$?
  ns=$(compgen -G "$OUT/samples_gsm8k_*.jsonl" >/dev/null 2>&1 && ls "$OUT"/samples_gsm8k_*.jsonl | wc -l || echo 0)
  echo "[b8] $(date -u '+%F %T') $ARM done rc=$rc samples_files=$ns"
done

echo "[b8] $(date -u '+%F %T') sampling done → inspect schema + build worksheets"
for ARM in adaq qlora; do
  OUT=$OUTROOT/${ARM}_seed42
  compgen -G "$OUT/samples_gsm8k_*.jsonl" >/dev/null 2>&1 || continue
  SF=$(ls "$OUT"/samples_gsm8k_*.jsonl | head -1)
  echo "[b8] === inspect $ARM ($SF) ===" >> "$LOG"
  $PY scripts/audit_flex_extract.py inspect "$SF" >> "$LOG" 2>&1
  $PY scripts/audit_flex_extract.py worksheet "$OUT" --arm "$ARM" --out "$OUT/flex_audit_worksheet_${ARM}.tsv" >> "$LOG" 2>&1
  echo "[b8] worksheet: $OUT/flex_audit_worksheet_${ARM}.tsv"
done
echo "[b8-audit] DONE  (worksheets in $OUTROOT/{adaq,qlora}_seed42/; 人工双标注后跑: $PY scripts/audit_flex_extract.py kappa <annotated.tsv>)"
