# results_v2 状态说明（2026-11-11）

本目录是论文主表口径：bf16 基座 + PEFT adapter、MMLU 全量 14042 / GSM8K 全量 1319、
TruthfulQA(mc1/mc2)+IFEval 0-shot、4 方法 × 3 种子 = 12 run。

## 当前状态：全量补跑中

`scripts/run_eval_missing.py` 正在后台双卡补跑（日志 `logs/repair_*.log`）。

- `summary.pre-repair.json`：**修复前的残缺假表，严禁引用**。它记录了 v2.0 marker
  缺陷造成的后果——12 个 run 无一同时具备 MMLU+GSM8K，聚合却 exit 0。
- `summary.json`：修复后的表。BATCH1(main) 跑完后为 2 任务完整版；
  BATCH2(safety) 跑完后为 4 任务完整版（**最终版**）。

## 如何判断 summary.json 是否可用（机读，别靠肉眼）

```
python - <<'PY'
import json
d=json.load(open("results_v2/summary.json"))
a=d["_audit"]
assert a["mismatch_task_count"]==0, a["sample_len_mismatch"]
for m in ("qlora","dora","adalora","adaq"):
    for t in ("mmlu/acc","gsm8k/exact_match,strict-match",
              "truthfulqa_mc1/acc","truthfulqa_mc2/acc","ifeval/inst_level_strict_acc"):
        assert d[m][t]["n_seeds"]==3, (m,t,d[m][t]["n_seeds"])
print("OK: 4 方法 × 3 种子 × 5 指标齐全，口径一致")
PY
```

每个 `<method>_seed<seed>/` 下应有 `.done_mmlu` / `.done_gsm8k` / `.done_safety` 三个标记，
且 `verify_eval_artifact.py` 对每个 marker 返回 0。

## 缺陷复盘

见 `docs/eval-protocol-v2.md` §5。一句话：v2.0 把 mmlu 与 gsm8k 写成同一个
`marker=main`，第一个任务成功后第二个被 SKIP；hf-mirror 429 导致的 mmlu 失败
又被后续 gsm8k 的成功 touch 同一 marker 掩盖。v2.1 改为每任务独立 marker +
产物机读校验 + 离线模式。
