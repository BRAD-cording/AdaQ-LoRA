#!/usr/bin/env bash
# run_b5_chattemplate.sh — B5（JCR 阻断项 P0）：14B GSM8K chat-template 复评。
#
# 目的：排除 R1 正向信号（Δ_flex +8.87pt）的「格式漂移 artifact」替代解释。
#   原 14B 评测用 gsm8k.yaml + --num_fewshot 4 但**无 chat-template** → Qwen2.5-Instruct
#   格式错配（strict .18 vs 官方≈.94）。gsm8k README 明示：「Use this task with
#   --fewshot_as_multiturn and --apply_chat_template to replicate Meta's reported performance」。
#   已验证 lm_eval 0.4.13 严格 parse_args 接受这两 flag（非静默忽略）。
#
# 若 chat-template 下 adaq−qlora flex 差仍存 → 格式 artifact 排除，R1 信号更硬；
# 若差缩小/消失 → 原信号是格式 artifact，R1 撤回。
#
# 顺带 --log_samples 捕获 item-level 生成（供 B11 bootstrap CI / TOST 分析，零额外 GPU）。
#
# GPU1（bf16 用 GPU0）；gate 在 B8 采样完成后（GPU1 释放）。
# 启动：nohup bash scripts/run_b5_chattemplate.sh >> logs/b5_chattemplate.log 2>&1 &
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
LOCK=/tmp/adaq_b5.lock
exec 9>"$LOCK"; flock 9
BASE=/home/ubuntu/AdaQ-LoRA
MODEL=$BASE/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
ADIR=$BASE/outputs/adapters_qwen14b
OUTROOT=$BASE/results_v2_14b_chattemplate
B8LOG=$BASE/logs/b8_flex_audit.log
LOG=$BASE/logs/b5_chattemplate.log
mkdir -p "$OUTROOT" "$BASE/logs"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 NLTK_DATA=/home/ubuntu/nltk_data HF_ENDPOINT=https://hf-mirror.com

# gate：等 B8 采样完成（GPU1 释放）；B8 自身 gate 在 phase3 DONE
while ! grep -q "\[b8-audit\] DONE" "$B8LOG" 2>/dev/null; do
  echo "[b5] $(date -u '+%F %T') waiting for b8-audit DONE ..."; sleep 300
done
echo "[b5] $(date -u '+%F %T') b8 done → start 14B chat-template GSM8K (adaq+qlora × seed42/2025/3407, GPU1)"

for ARM in adaq qlora; do
  for SEED in 42 2025 3407; do
    ADAPTER=$ADIR/${ARM}_seed${SEED}
    if [ ! -f "$ADAPTER/adapter_config.json" ]; then echo "[b5] MISS $ARM seed$SEED adapter"; continue; fi
    OUT=$OUTROOT/${ARM}_seed${SEED}
    mkdir -p "$OUT"
    if [ -f "$OUT/.done_gsm8k_chattemplate" ]; then echo "[b5] $ARM seed$SEED done, skip"; continue; fi
    echo "[b5] $(date -u '+%F %T') START $ARM seed$SEED gsm8k +chat_template +fewshot_as_multiturn +log_samples"
    CUDA_VISIBLE_DEVICES=1 $PY -m lm_eval --model hf \
      --model_args pretrained=$MODEL,load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_compute_dtype=bfloat16,bnb_4bit_use_double_quant=True,dtype=bfloat16,peft=$ADAPTER \
      --tasks gsm8k --num_fewshot 4 --apply_chat_template --fewshot_as_multiturn \
      --batch_size auto --device cuda:0 --seed 42 --log_samples --output_path "$OUT" >> "$LOG" 2>&1
    rc=$?
    nj=$(find "$OUT" -name 'results_*.json' 2>/dev/null | wc -l)
    if [ "$rc" = "0" ] && [ "$nj" -ge 1 ]; then
      touch "$OUT/.done_gsm8k_chattemplate"; echo "[b5] $(date -u '+%F %T') OK $ARM seed$SEED rc=0 results=$nj"
    else
      echo "[b5] $(date -u '+%F %T') FAIL $ARM seed$SEED rc=$rc results=$nj（不 touch .done）"
    fi
  done
done

# 汇总：chat-template flex vs 原非-chat flex（原值来自 results_v2_qwen14b*/summary）
echo "[b5] $(date -u '+%F %T') 6 run 完成 → 生成对比报告"
$PY - <<'PYEOF' >> "$LOG" 2>&1
import json, glob, os
OUTROOT="results_v2_14b_chattemplate"
def flex_of(d):
    fs=glob.glob(os.path.join(d,"**","results_*.json"),recursive=True)
    if not fs: return None
    r=json.load(open(sorted(fs)[-1]))
    g=r.get("results",{}).get("gsm8k",{})
    return g.get("exact_match,flexible-extract"), g.get("exact_match,strict-match")
rows=[]
for arm in ["adaq","qlora"]:
    for seed in [42,2025,3407]:
        d=os.path.join(OUTROOT,f"{arm}_seed{seed}")
        fl,st=flex_of(d) if os.path.isdir(d) else (None,None)
        rows.append((arm,seed,fl,st))
# 原非-chat flex（results_v2_qwen14b*）
orig={}
for arm in ["adaq","qlora"]:
    for seed in [42,2025,3407]:
        base="results_v2_qwen14b" if seed==42 else f"results_v2_qwen14b_seed{seed}"
        ds=glob.glob(os.path.join(base,arm,"**","results_*.json"),recursive=True)
        if ds:
            r=json.load(open(sorted(ds)[-1])); g=r.get("results",{}).get("gsm8k",{})
            orig[(arm,seed)]=g.get("exact_match,flexible-extract")
with open(os.path.join(OUTROOT,"REPORT_14B_chattemplate.md"),"w") as f:
    f.write("# B5：14B GSM8K chat-template 复评（排除格式漂移 artifact）\n\n")
    f.write("命令：gsm8k --num_fewshot 4 --apply_chat_template --fewshot_as_multiturn（gsm8k README 推荐，复刻官方口径）；NF4 评测同原。\n\n")
    f.write("| arm | seed | 原 flex(无chat) | chat flex | chat strict | Δflex(chat−原) |\n|---|---|---|---|---|---|\n")
    for arm,seed,fl,st in rows:
        o=orig.get((arm,seed))
        fs=f"{fl:.4f}" if fl is not None else "—"
        ss=f"{st:.4f}" if st is not None else "—"
        os_=f"{o:.4f}" if o is not None else "—"
        d=f"{(fl-o)*100:+.2f}pt" if (fl is not None and o is not None) else "—"
        f.write(f"| {arm} | {seed} | {os_} | {fs} | {ss} | {d} |\n")
    # 配对差 chat-template
    f.write("\n## chat-template 下 adaq−qlora 配对差（flex）\n")
    diffs=[]
    for seed in [42,2025,3407]:
        a=dict((r[0],r) for r in rows if r[1]==seed).get("adaq")
        q=dict((r[0],r) for r in rows if r[1]==seed).get("qlora")
        if a and q and a[2] is not None and q[2] is not None:
            dd=(a[2]-q[2])*100; diffs.append(dd)
            f.write(f"- seed{seed}: {dd:+.2f}pt\n")
    if diffs:
        mean=sum(diffs)/len(diffs); same=all(d>0 for d in diffs) or all(d<0 for d in diffs)
        f.write(f"\n均值 {mean:+.2f}pt，{'3/3 同号' if same else '符号不一致'}。\n")
        f.write("\n**判定**：chat-template 下若差仍存且同号 → 格式 artifact 排除，R1 信号更硬；若缩小/消失 → 原 +8.87pt 系格式 artifact，R1 撤回。（噪声底 N=4.09pt 仍适用）\n")
print("REPORT_14B_chattemplate.md 生成")
PYEOF
echo "[b5-chattemplate] DONE  (report: $OUTROOT/REPORT_14B_chattemplate.md)"
