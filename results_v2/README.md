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

## 7. 运行期判据与 ETA 基准（2026-11-11 补）

**唯一权威判据**（按可靠性排序，`check_eval_status.py` 是交付闸门）：

1. `logs/repair_main.log` / `logs/repair_safety.log` 中 `[... ] OK    gpuN` 行数
2. `.done_mmlu` / `.done_gsm8k` / `.done_safety` 各 12 个
3. 新增的 `results_*.json`（时间戳文件，补跑不会覆盖旧文件）
4. `python scripts/check_eval_status.py` → exit 0 / `VERDICT: PASS`

**禁止当判据的伪信号**：`eval_*.log` 里 tqdm 进度条用 `\r` 重绘，整段是**一个超长行**；
`grep -oE '...[0-9]+%' | tail -1` 会拿到任意历史帧，既不是当前进度也不能算 ETA。
曾据此误判"10 分钟只涨 1%"，实为测量假象。看进度只看 `OK` 行。

**历史耗时基准**（本机实测，2× RTX 4090D，bf16 基座 + PEFT，全量）：

| 任务 | 实测 | 出处 |
|:---|:---|:---|
| MMLU 全量 14042 | 27min24s | `qlora_seed42/eval_main.log` 05:55:49→06:23:13 |
| GSM8K 全量 1319 | 约 22min | 历史批次日志 |
| 冒烟 LIMIT=8（mmlu+gsm8k+safety 三任务） | 243s（88.8+95.0+58.9） | `logs/smoke_v21.log` |

按 12 任务 / 双卡串行估算：BATCH1 约 2.5–3h，BATCH2 因 IFEval 541 题生成约 3–5h，
总计 5–8h（估算，非承诺）。

**卡住判据**：超过 40min 无新增 `OK` 行且 `nvidia-smi` util 掉到 0 → 查对应
`eval_*.log` 尾部；529/429/OOM 会体现在 `FAIL ` 行，脚本会 `exit 2` 且不 touch marker。
不要用 `logs/repair_all.status` 里的 `ALL DONE` 文案判断成功——批次失败时它也会写；
以 `check_eval_status.py` 的机读结果为准。

### 补：诊断命令自身的陷阱（2026-11-11 实测）

曾在一条诊断命令里同时 grep「当前补跑日志」和「历史基准日志 qlora_seed42/eval_main.log」，
输出混在一起，把历史日志的 `56168/56168 [21:31<00:00, 43.48it/s]` 误读成
"当前首批快出结果"。**教训：核对进度时 grep 范围必须只含本次补跑的日志文件；
历史日志只能单独查、且必须显式标注"基准"**。

取当前真实最后一帧的正确做法（tqdm 用 `\r` 重绘，整段是一个超长行）：

```
python - <<'PY'
import re
p="results_v2/adalora_seed3407/eval_mmlu.log"
pat=re.compile(r"Running (?:loglikelihood|generate_until) requests:\s*(\d+)%\|[^\r]*?(\d+)/(\d+) \[([0-9:]+)<([0-9:]+),\s*([\d.]+)it/s\]")
f=pat.findall(open(p,errors="ignore").read().replace("\r","\n"))[-1]
print(f)   # ('22','12193','56168','10:53','17:12','42.61')
PY
```

实测（21:05 UTC）：当前两个全量 MMLU 各 21–22%、约 40 it/s、剩余 17–19min；
历史单卡全量 MMLU 平均 43.48 it/s、推理 21:31、wall 27min24s。
双卡并发下 40 it/s 属正常量级，**无需干预加速**，不要为了"看起来更快"去重跑。

## 8. 补跑前的链路预验证（2026-11-11 轮 3，隔离目录，不碰运行中的 results_v2）

补跑会产出「旧残缺 json + 新补跑 json」共存的目录，聚合器必须正确合并。
在 `/tmp/agg_sim*` 用真实产物做过预验证，结论：

- `verify_eval_artifact.py` 对**全量**产物 7 组测试全过：历史全量 mmlu/gsm8k 正向
  exit 0；只有 gsm8k 的 run 查 mmlu、只有 mmlu 的 run 查 gsm8k、未跑 safety 的 run
  查 safety、不存在目录 → 全部 exit 1（即 v2.0 的残缺能被抓住）。
- `aggregate_results.py` 同一 run 目录多份 `results_*.json` → **按 task 键合并全部文件**，
  同一 (run, task) 取 mtime 最新。注入 decoy(acc=0.9999, mtime 最旧) 后仍取真实
  0.6532，去重生效。
- 明确用 3 份真实产物（mmlu/gsm8k/safety）合并，`qlora` 一次拿到 5 个指标：
  mmlu/acc, gsm8k/exact_match,strict-match, truthfulqa_mc1/acc, truthfulqa_mc2/acc,
  ifeval/inst_level_strict_acc —— `ALL_5_METRICS_PRESENT: True`。
- 踩过的坑（仅诊断脚本，非产物）：冒烟目录有 3 份 results json，`find | head -1`
  会拿到 mmlu 那份，误以为 safety 指标缺失。**按 task 内容筛文件**，不按文件名猜。

结论：补跑产物落地后直接跑 `aggregate_results.py` 即可，无需人工拼接。
