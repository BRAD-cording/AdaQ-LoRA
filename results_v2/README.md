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

## 9. 卡住时的确定性恢复（2026-11-11 轮 4）

`logs/repair_all.sh` 是 `setsid nohup` 后台自跑。若远端 OOM、SSH 断连或机器重启
把进程干掉，`repair_all.status` 不会有 `ALL DONE`，goal 续跑轮会在空等里耗尽预
算（实测 4 轮 / 约 3–4 分钟一轮，30 轮约 2h，而补跑剩 5–8h）。

**卡住判据**：超过 40min 无新增 `OK` 行、无新增 `results_*.json`、GPU util 掉 0，
且 `check_eval_status.py` 仍 `INCOMPLETE`。

**一条命令恢复**（幂等，脚本只写不自动调度，活着时不做任何改动）：

```
bash scripts/resume_repair.sh
```

它先 `pgrep` 确认无活跃补跑进程，再跑一次 `check_eval_status.py`；仍 INCOMPLETE
才用 `run_eval_missing.py`（WANT=mmlu,gsm8k,safety，双卡，跑完自动聚合）幂等补
缺失任务。**不删产物、不重置 marker、不改评分器/测试/CI**，补跑产物是新增时间戳
json，聚合器按 mtime 去重。

严禁在补跑进程仍存活时手工重启 `repair_all.sh` —— 会造成两个 `lm_eval` 同写
同一 `output_path`。


---

## 10. 全量混合产物聚合只读预演（Round 7 追加，排除交付返工）

轮 3 的合并预演用的是 `results_smoke/`（LIMIT=8 冒烟产物），**不是**「历史全量产物 + 全量补跑产物」的混合。
本次用真实 `results_v2` 做只读预演，输出写到 `/tmp/agg_preview/summary.json`，**未碰** `results_v2/summary.json`：

| 项 | 实测值 |
| --- | --- |
| 命令 | `aggregate_results.py --results-dir results_v2 --out /tmp/agg_preview/summary.json` |
| 退出码 | `0` |
| `_audit.mismatch_task_count` | `0` |
| `_audit.sample_len_mismatch` | `[]` |
| `_audit.duplicate_files` | `[]` |
| `n_files / n_runs` | `14 / 12`（预演时点） |

**排除「假绿」**：`norm_keys` 只丢非数值键，`results[task].sample_len` 是数值故被保留；
实测 `mmlu: {'14042': 10 份}`、`gsm8k: {'1319': 4 份}` —— 值真实存在且各自统一，
`mismatch_task_count=0` 是**真绿**，不是「全 None 所以审计不触发」的假绿。
注意顶层 `n-samples` 对 mmlu 是 `null`、对 gsm8k 是字典，**不能拿它当口径判据**，判据只认 `results[task].sample_len`。

**结论**：BATCH1(12 个 mmlu/gsm8k) 与 BATCH2(12 个 safety) 产物全落地后，只要补跑沿用全量口径，
`aggregate_results.py` 不会因历史/补跑混合误报 `exit 2`；`check_eval_status.py` 的 `mismatch_task_count==0` 判据可达。
`repair_all.sh` 仅 `set -o pipefail`、无 `set -e`，BATCH1 的非零退出**不会掐断** BATCH2，批次链接健壮。


---

## 11. 统计功效边界（交付诚实红线，Round 10 追加）

`summary.json` 的 `significance_test` 采用 **Wilcoxon signed-rank (paired by seed)**，
`aggregate_results.py` 的内置门槛是 **≥6 对配对值**才计算有效 p 值；不足时只输出 note。

**实测（本轮对当前 summary.json）**：全部任务均为
`{"note": "仅 2 对配对值；Wilcoxon 需 >=6 对才有效", "n_pairs": 2}`。
实验矩阵固定 3 个种子 `[42, 3407, 2025]`，即使全量补齐、每方法 3 seed 全配对，
**最多也只有 3 对 < 6** —— 这是设计约束，不是数据缺失，补跑不会改变它。

**因此交付措辞的硬边界**：
- 只能给**描述性** `mean ± std (n=3)`，**禁止**「显著优于 / SOTA / p<0.05 / 统计显著」等推断性表述；
- 显著性一律写「未检验 / 功效不足（3 seeds < 6 配对，Wilcoxon 不适用）」；
- 与 §6「MMLU 四方法差异在种子波动内、不能当方法有效性主证据」叠加：
  **AdaQ 的方法有效性主张目前没有统计证据支撑**，论文需扩到 ≥6 种子或改用预注册的检验并重新设计实验。

聚合器已自动输出这条 note（无需改代码、不改评分器/测试/CI）；本节只把「不能说的」固化成交付口径，
防止下游把 `significance_test` 字段误读成「已做显著性检验且通过」。

## §12 BATCH2 safety 全灭复盘（2026-11-11）

### 现象
`logs/repair_safety.log`：12/12 `FAIL rc=1`，每个耗时 3979.8–9865.2s；
`repair_all.status` 仍写 `BATCH2 exit=2 (summary.json 完整表已落)` + `ALL DONE` —— **文案与事实相反**，
再次印证「status 文件不作判据，只认 OK 行 + 机读闸门」。

### 根因（12/12 同一根因，逐字证据）
```
results_v2/<run>/eval_safety.log:
  [nltk_data] Error loading punkt_tab: <urlopen error [Errno 104] Connection reset by peer>
  Downloaded punkt_tab on rank 0            <-- 假成功：download 失败也打印这句
  ...
  LookupError:
    Resource 'punkt_tab' not found.
    Attempted to load 'tokenizers/punkt_tab/english/'
```
链路：`lm_eval/tasks/ifeval/instructions_util.py:36 download_nltk_resources()`
→ `:45 nltk.data.find("tokenizers/punkt_tab")` 找不到 → `:48 nltk.download("punkt_tab")`
→ 需要 `raw.githubusercontent.com`（本机 `Errno 104`，实测 `000ERR`）→ 失败仍 `:49 print("Downloaded ...")`
→ 生成阶段跑完 1–2.7h 后，算 metric 时 `:1692 count_sentences` / `:1587 nltk.word_tokenize` 抛 LookupError
→ 进程 rc=1，**predictions 全在内存里丢失**，02:34 之后落盘产物 = 0。

排除项（均 0/12）：`Killed` / `CUDA out of memory` / `Traceback` 之外的 CUDA 错误。

### 为什么 LIMIT=8 冒烟没抓到
冒烟只跑 8 条 IFEval，恰好没命中调用 `word_tokenize` / `count_sentences` 的指令；
`instructions_util` import 成功 ≠ metric 计算成功。**冒烟粒度不足 = 没冒烟。**

### 修复（只补数据，不改 lm_eval 源码/评分逻辑）
1. 从 `ghproxy.net` / `gh-proxy.com` 拉 `punkt_tab.zip`(4.3MB) + `punkt.zip`(13.9MB)
   （官方 `raw.githubusercontent.com` 与 hf-mirror 对应路径均不可用）
2. 解压到 `/home/ubuntu/nltk_data/tokenizers/`（不是 `nltk_data/` 根，否则 nltk 找不到 `tokenizers/` 前缀）
3. `chmod -R go-w`：nltk 3.10.4 拒读 group/other-writable 目录
4. `scripts/preflight_safety_metrics.py`（新增）：无 GPU 跑**全量 541 条** `process_results`，
   校验 4 个 metric 键 + 2 个 nltk 资源 + TaskManager 可解析 → `VERDICT: READY/BLOCKED`
5. `scripts/run_eval_missing.py`：`WANT` 含 safety 时先跑 preflight，`rc!=0` 直接 `return 3` 拒绝启动；
   并显式注入 `NLTK_DATA=/home/ubuntu/nltk_data`
6. `logs/repair_safety_v2.sh`：**2-run canary 门控** → 2/2 产物校验通过才放量剩余 10 run

### 教训（可复用）
- 「下载成功」的 print 不能当证据，要看资源 `nltk.data.find()` 是否真能解析。
- 评测类任务的门控必须打在**最容易崩且最贵的那一段**（这里是 metric，不是 generation），
  且必须用**全量样本**驱动，不能用小 limit 抽样。
- 长任务里任何「跑完才报错」的依赖缺失，都应在启动前用 CPU 侧打靶提前暴露。

## §13 preflight 门控的负向验证（防止「假绿的门」）

门控最危险的失败模式不是报错，是**该报错时却 PASS**。所以对
`scripts/preflight_safety_metrics.py` 做了负向验证。

### 无效的验证构造（记录，避免重犯）
`NLTK_DATA=/tmp/nltk_empty` **不能**隔离——nltk 把 `NLTK_DATA` 当作**追加**搜索目录，
仍能命中 `/home/ubuntu/nltk_data`，结果 PASS（`CLEAN_NEG_EXIT=0`）。
误以为「门失效」，实际是测试构造错。教训：**验证门控要先证明"故障条件真的注入进去了"**。

### 有效的负向验证（monkeypatch `nltk.data.find`）
让 `punkt_tab` 解析抛 `LookupError`，其余资源不变：

| preflight 检查项 | 结果 |
| --- | --- |
| `nltk resources` | CAUGHT `LookupError: Resource 'punkt_tab' not found. [SIMULATED]` |
| `ifeval process_results`（541 条） | CAUGHT `RuntimeError: 64/541 docs failed; first=(1040, 'LookupError', ...)` |
| `ifeval agg_inst_level_acc` | CAUGHT `LookupError` |

独立复核：注入后直接跑 `process_results` 全量 → **64/541 条抛 LookupError**。
=> 单条 metric 计算就会炸，不需要等 `generate_until` 跑完 48 分钟；
**门是真门，不是全 None 式假绿**。

### 顺带查明的覆盖率事实
- 短回答（`One. Two.`）命中分词路径：**64/541**
- 长回答（60 句）：命中 **112/541**
=> 约 1/8–1/5 的样本会真正走 `nltk.word_tokenize` / `count_sentences`。
当年 `--limit 8` 冒烟只跑 8 条，命中概率 ≈ 8×(64/541) ≈ 0.95 条，
**期望不到 1 条**——这就是它必然漏检的定量解释。

## 14. safety 放量阶段的两个结构性风险（均已在启动前排除）

### 14.1 `resume_repair.sh` 的 `alive()` 谓词本来永远不命中

原写法 `pgrep -f "python scripts/run_eval_missing.py"`。真实 cmdline 是

```
/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python -W ignore scripts/run_eval_missing.py
```

中间夹着 `-W ignore`，**模式不匹配**。实测：真实进程数 `1`，旧模式命中 `0`。

后果：canary 通过 → 放量之间的空档里，`alive()` 会误判「无人在跑」，于是
`resume_repair.sh` 拉起**第二个驱动**。两个 `run_eval_missing.py` 同时按
`GPUS=0,1` 抢同一张卡 → 显存打爆或互相拖慢，且两边都会看到「任务缺失」而重复跑。

修复（commit `3651c7b`）：改宽松 basename 模式，并把 `repair_safety_v2.sh`、
`finalize_watch.sh` 一并纳入判活名单：

```bash
pgrep -f "repair_all\.sh"        && return 0
pgrep -f "repair_safety_v2\.sh"  && return 0
pgrep -f "run_eval_missing\.py"  && return 0
pgrep -f "finalize_watch\.sh"    && return 0
pgrep -f "lm_eval --model"       && return 0
```

正向实测（canary 在跑时执行）：输出「补跑仍在运行，不做任何改动」，`RESUME_EXIT=0`，
之后 `run_eval_missing` 进程数仍为 **1**、`lm_eval` 仍为 **2** —— 没有多拉一个驱动。

### 14.2 放量会不会把 canary 那两个 run 再跑一遍？

会问这个问题是因为：`build_jobs()` 的跳过条件有两条（`.done_<marker>` 存在，或
`tasks_in()` 已含全部所需 task）。canary 通过后这两条都成立，理论上会跳过；
但「理论上」不算证据。

沙箱实证（`/tmp/sim_rest`，合成 12 个 run 目录，其中 `qlora_seed42` /
`adalora_seed2025` 标记为 canary 已完成）：

```
SIM_JOBS=10   (期望 10：canary 两 run 必须被跳过)
REPEATED= 无重复
TARGETS= adalora_seed3407, adalora_seed42, adaq_seed2025, adaq_seed3407,
         adaq_seed42, dora_seed2025, dora_seed3407, dora_seed42,
         qlora_seed2025, qlora_seed3407
```

结论：放量阶段精确落在那 10 个未完成 run 上，不重跑 canary。

### 14.3 承载余量（放量前实测）

| 项 | 实测 | 判断 |
| --- | --- | --- |
| 磁盘 `/` | 148G used / 321G avail (32%) | 充足 |
| 单个 run 产物 | 均值 0.1MB | 10 run 约 1MB，可忽略 |
| GPU 显存 | 16072 / 24564 MiB per card | 单卡单进程，未逼近上限 |
| 系统内存 | used 9G / total 125G，available 116G | 充足 |
| `batch_size` 判定 | loglikelihood 32 / generate_until **1** | IFEval 的瓶颈，不调（避免 OOM 与中途换方案） |

**ETA**：`max_gen_toks=1280` 且 generate_until batch=1，单 run 实测速率 2.7–6.7 s/it，
推得单 run 约 48–61 min；canary 2 个 → 放量 10 个 = 5 波 × ~50–60 min ≈ **4.2–5 h**。

## 15. 我在「修 alive()」时自己引入的一个死锁（已修）

§14.1 修完 `alive()` 之后，我把 `finalize_watch.sh` 也加进了判活名单——理由是
「看门狗活着说明流程还在推进」。这条推理在**调用方向**上是错的：

```
finalize_watch.sh  --(驱动猝死时)-->  bash scripts/resume_repair.sh
                                          └─ alive() 里 pgrep "finalize_watch\.sh"
                                             → 命中调用者自己 → alive=YES → 直接退出
```

即：看门狗一旦调用恢复脚本，恢复脚本就会 pgrep 到**正在调用它的看门狗**，判定
「还有人在干活」，于是永远不恢复。整条无人值守链在最关键的那一步静默失效，
而且日志上看起来完全正常（「补跑仍在运行，不做任何改动」）。

修复（commit `adc5f15`）：调用方传 `ADAQ_INVOKED_BY=finalize_watch`，判活时显式排除调用者。

```bash
if [ "${ADAQ_INVOKED_BY:-}" != "finalize_watch" ]; then
  pgrep -f "finalize_watch\.sh" >/dev/null 2>&1 && return 0
fi
```

**语义单测**（驱动全死、仅看门狗存活）：

| 版本 | `alive()` | 结果 |
| --- | --- | --- |
| 带 watchdog 判活（旧） | YES | 永不恢复 ← 死锁 |
| `ADAQ_INVOKED_BY=finalize_watch`（新） | NO | 正确进入恢复分支 |

**恢复分支端到端实证**（沙箱：强制 `alive()=false` + `OUT_DIR=/tmp/sim_rest2`
+ `RUNS=__none__`，绝不起 GPU 任务）：

```
[14:21:02] 检测到无活跃补跑进程，检查交付判据…
run 完整度: 24/36 任务组齐全
VERDICT: INCOMPLETE 继续等待/补跑
[14:21:02] 判据仍 INCOMPLETE，幂等重启缺失任务（双卡）。
→ preflight PASS → OUT=/tmp/sim_rest2 WANT=['safety'] jobs=0
→ nothing to do
```

真实 GPU 任务数全程保持 `lm_eval=2 / run_eval_missing=1` 未受影响；
真实 `resume_repair.sh` 无 `SANDBOX TEST` 残留（`grep -c` = 0）。

看门狗已用新脚本重启（旧 PID 3829581 → 新 PID 3839641）：bash 读的是脚本字节码，
**不重启就还是旧逻辑**，这一点和改配置必须重启同理。
