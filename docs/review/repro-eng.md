# AdaQ-LoRA 投稿前评审 · 训练效率与可复现性工程（repro-eng）

- 评审人角色：训练效率与可复现性工程专家（显存预算 / 数值不变性 / 断点续训 / 实验台账 / 第三方可重跑性）
- 评审日期：2026-11-15（UTC）
- 仓库：/home/ubuntu/AdaQ-LoRA @ HEAD `94ee08c15ad93d476068237f3c5545261f5d8fa3`
- 评审方式：**全程只读实读实核**（git log/show、pip list、torch.load 读 training_args.bin、safetensors 张量形状、crontab -l、ps、logs 全文、site-packages 源码对照）。评审期间 14B 三种子评测链（chain_b4_seeds.sh，PID 1196644）正在运行，未做任何干扰。
- 配套评审：docs/review/jcr-gate.md、docs/review/rank-budget.md（本文不重复其结论）。

---

## 0. 结论（Verdict）

## **MAJOR REVISION（大修）**

一句话理由：**数字是诚实的**（results_v2/FINAL_REPORT.md L44-46 自我禁止「显著优于/SOTA/p<0.05」表述，主表四方法差异确实落在种子噪声内），**评测侧重跑链扎实**（2026-11 用 lm_eval 0.4.13 对 7 月 adapter 统一重评，且有 verify_eval_artifact.py 机读门）；但**「第三方拿 repo+HEAD 重跑出主表（8B）每个训练数字」不成立**——主实验的真实训练配置（use_liger_kernel=True、max_seq_length=None≈8K 打包、peft 0.19.1 时代环境、/mnt 原机器硬件）与仓库内任何已提交代码/文档**互相矛盾或完全缺失**，全部结果文件与 14B 编排脚本不在 git；断点续训数值等价性**零验证**（现存 smoke 日志是 no-op）。这些属于可补救的台账/文档/流程缺口（多数不需重训），故不是 reject；但缺口位于「每张表每个数字可复核」这一投稿硬要求上，远超 minor。

---

## 1. 证据基础（本文所有断言的实测来源）

| # | 证据对象 | 实测手段 | 关键结果 |
|---|---------|---------|---------|
| E1 | git 历史 | `git log/show -s` | 首次入库 `950ccc4` = **2026-11-10**，晚于主实验训练（2026-07-31）约 3.5 个月 |
| E2 | 主实验 adapter | `outputs/adapters/*/adapter_config.json` mtime + 字段 | 全部 **2026-07-31 02:29**，`peft_version: 0.19.1`（主实验唯一存活的版本记录） |
| E3 | 训练配置真值 | `torch.load(.../training_args.bin, weights_only=False)` | 见 §2 Q1/Q3 三族对照表 |
| E4 | rank 实效 | safetensors `lora_A` 形状实测 | 主 adaq q[12,4096]/v[4,4096]（peft 0.19.1）；14B adaq q[7,5120]/v[4,5120]、消融 per_layer 64 条、schemeA q[10]/v[4]（peft 0.21.2）**两代版本 rank_pattern 均实证生效**；adalora lora_A=[16,4096]=init_r 未剪枝 |
| E5 | 现行环境 | `pip list`（py3.13） | peft **0.21.2** / transformers **4.57.6** / torch **2.11.0+cu130** / trl **0.12.0** / lm_eval **0.4.13** / numpy 2.4.3 / liger_kernel 0.5.10 |
| E6 | peft 匹配语义 | `site-packages/peft/utils/other.py:1524-1532`、`peft/tuners/lora/model.py:309-312` | `get_pattern_key` = `re.match(rf"(.*\.)?({key})$", key_to_match)` 后缀正则，短名/层级名两种 key 风格均命中 |
| E7 | 种子链 | transformers 4.57.6 `Trainer.train` 源码 | `set_seed(self.args.seed)` 确认调用；`data_seed=None` → shuffle 派生自 seed |
| E8 | 评测种子 | `results_v2/adaq_seed42/eval_main.log:4` | `Setting random seed to 42 \| numpy 42 \| torch 42 \| fewshot 42`（lm_eval 作用域=全部四类 RNG） |
| E9 | crontab / 进程 | `crontab -l`、`ps aux` | AdaQ 8 条 @reboot sleep 阶梯 + **另一项目 AFLAP 7 条共享同机 GPU，无仲裁** |
| E10 | 数据集 revision | `~/.cache/huggingface/hub/datasets--teknium--OpenHermes-2.5/refs/main` | `b82037821055c377bed0d495e72e46de3bc72e84`（仅存在于本机 cache，代码未 pin） |

---

## 2. 六问六答

### Q1 第三方拿 repo+commit 能否重跑每张表每个数字？——**主表：不能；11 月新表：接近能但仍有硬缺口**

**(a) 主实验（8B，results_v2 主表 12 run）的训练侧不可重跑，四个独立原因：**

1. **实际训练代码 ≠ 任何已提交版本。** 首次入库 `950ccc4`（2026-11-10）是事后重建快照：其 train.py 向 SFTConfig 传 `max_length=args.max_length`，而真实产物 `outputs/adapters/adaq_seed42/checkpoint-21/training_args.bin` 实测 `max_seq_length=None`——两者不可能由同一份代码产生（现行 trl 的 SFTConfig 无 `max_length` 字段）。7 月真实入口 `batch_train.py` + `auto_pipeline.sh`（`cd /mnt/AdaQ-LoRA`）虽在 git，但只能证明调用方式（--epochs 3 --dataset-size 2000 --seed ×3），不能证明 train.py 当时的内部行为。
2. **有效打包长度无记录且三处矛盾。** 主实验 `max_seq_length=None` + `packing=True`；由 trainer_state `max_steps=21`（3 epoch × 7 步/epoch × 有效 batch 16 = 112 条打包序列/epoch）反推，平均每序列 ≈8K token（与 Llama-3 `model_max_length=8192`、OpenHermes-2.5 平均 ~450 token/样本自洽）。而 `configs/training_params.yaml` 写 `max_seq_length: 2048`，HEAD train.py 默认 `--max-length 2048`，现行 trl 0.12 对 None 的兜底是 `min(model_max_length, 1024)`（`trl/trainer/sft_trainer.py:305-307`）——**按任何一个重跑都得到不同的数据组织，梯度统计不同，数字必然不同**。
3. **数值路径不同：主实验开着 Liger。** training_args.bin 实测主实验 adaq/qlora 均 `use_liger_kernel=True`；消融与 14B 均 `False`；HEAD train.py 亦为 `False`。Liger 融合算子（fused CE 等）改变数值结果，且 7 月 liger-kernel 版本无任何记录。仓库自己的铁律 3 工具 `scripts/numerical_invariance_check.py` **从未跑过存档**（logs/ 与 docs/ grep 无任何运行输出）。
4. **7 月环境整体缺失。** 唯一存活记录是 adapter_config 里的 `peft_version: 0.19.1`；当时的 transformers/trl/torch/bitsandbytes/datasets/liger 版本、Python 版本、GPU 型号（`/mnt/workspace` 表明是阿里云 PAI 原机器，非现 2×4090D）**零记录**。requirements.txt 全部是 `>=` 下界（自 950ccc4 未改），无 lockfile/environment.yml/Dockerfile/pip freeze 存档。

**(b) 结果文件不在 git，数字本身不可核对。** `.gitignore` 排除 `outputs/ logs/ results/* results_v2/*`，仅 `results_v2/README.md` 与 `summary.json`/`summary.pre-repair.json` 被反选（git status 显示为未跟踪=反选生效但**从未 git add**）。后果：FINAL_REPORT.md、summary.json、每个 run 的 lm-eval 原始 `results_*.json`、trainer_state.json、training_args.bin 全部游离在版本控制外。第三方 clone 后**看不到主表数字，更无法把论文表格对回原始评测产物**。

**(c) 14B 三种子链的编排层不在 git。** `scripts/chain_b4_seeds.sh`、`scripts/run_b4_qwen14b_seeds.sh`、`scripts/run_eval_b4_qwen14b_seeds.py`、`scripts/aggregate_b4_3seeds.py`、`ranks_qwen14b.json` 均未跟踪（git status 17 条未跟踪清单实测）。14B 表（P1-B4）的复现路径 = 未提交脚本 + 未提交 rank 文件。`adaq_precompute.py` 本身在 git ✓（950ccc4，227 行），但 ranks_qwen14b.json 的生成未落盘命令记录。

**(d) 评测侧相对扎实（应予肯定）。** 主表 12 run 的数字 = 2026-11-10~12 用 lm_eval 0.4.13 对 7 月 adapter 的统一重评（`results_v2/*/eval_main.log` 头部含完整 model_args 与四项 seed 日志），`run_eval_missing.py` 有 verify_eval_artifact.py 机读校验后才写 marker（L146-151），`results_v2/README.md` 提供机读自校验片段，`docs/eval-protocol-v2.md` 明确 bf16 基座口径。**给定 adapter，评测数字可重跑**；不可重跑的是 adapter 本身。

**(e) 附带发现（影响表格可信度）：**
- **AdaLoRA 基线未剪枝、2 倍参数**：train.py L52 `total_step=200` 硬编码 >> 主实验实际 `max_steps=21` → 剪枝从未触发；safetensors 实测 lora_A=[16,4096]（init_r），system_metrics.json 实测参数 6,816,768 ≈ 其他方法（3.4-3.8M）的 2 倍；且其训练 loss 3.0891→2.9365 与其他方法 2.0→1.5 明显异态。对比表中 adalora 行必须如实标注此状态，否则构成基线不公平。
- **14B seed42 训练拓扑与另两种子不一致**：seed42 走 `run_b4_qwen14b.sh`（无 CUDA_VISIBLE_DEVICES → device_map auto 双卡切分，training_stats peak_vram=6.67GB），seeds 2025/3407 走单卡（17.6/17.62GB）。同表三种子的系统指标不可比；数值上模型并行切分不改变数学但非位级等同。
- `run_b4_qwen14b.sh` L39/44 训练输出 `| tail -15` → **14B seed42 全量训练日志已丢**（仅 trainer_state.json 25 条 loss 存活）；set -e 下 `rc=$?` 的 FAIL 分支是死代码。
- system_metrics.json `base_params=8,030,261,248` 为硬编码（smoke 实测 8,033,669,120，差 3.4M）。
- 评审时刻 14B 三种子评测仍在进行（seed2025 mmlu done / gsm8k 运行中，seed3407 未开始），REPORT_14B 数字未定稿。

### Q2 seed 传递链是否完整？方差归因是否干净？——**训练链闭环 ✓；评测存在两种并存口径，且任务书所述「评测固定 42」仅对部分表成立**

- **训练侧 ✓**：`batch_train.py`/各 run 脚本 `--seed {42,2025,3407}` → `SFTConfig(seed=args.seed)` → transformers `Trainer.train` 内 `set_seed(self.args.seed)`（E7 源码实证）→ python/numpy/torch RNG + sampler shuffle（data_seed=None 派生自 seed）。training_args.bin 实测各 run seed 字段正确（42/2025 抽查一致）。数据顺序由 seed 决定，packing 在 shuffle 前按固定顺序完成——同 seed 同数据顺序成立（受 Q1 数据集未 pin 的制约：若上游数据集变更，同 seed 也不同数据）。
- **评测侧存在两种口径（并存但未文档化）**：
  - **主表（results_v2 12 run）**：`run_eval_missing.py` L130 `--seed job["seed"]`，从 run 目录名 `{method}_seed{n}` 解析 → **评测 seed = 训练 seed**（42/2025/3407 随 run 变）；`run_eval_unified.sh` L124 同。eval_main.log L4 实证四项 RNG 全部设置。
  - **消融/schemeA/14B**：`run_eval_ablation.py` L78 `--seed 42` 固定（schemeA multiseed 通过 `import run_eval_ablation as base` 复用，同固定 42）；`run_eval_b4_qwen14b_seeds.py`/`run_eval_b4_qwen14b.py` `--seed 42` 固定。
- **方差归因判定**：
  - 14B/schemeA 表：训练 seed 变、评测 seed 固定 42 → **std 干净地归因于训练随机性** ✓（这是正确设计）。
  - 主表：训练与评测 seed 共变 → std 是训练+评测混合方差。lm_eval 默认贪心解码，评测 seed 只影响 few-shot 选样与 batch 分组，量级远小于训练 seed，实践上可接受——但**必须在论文方法节声明**，否则「n=3 std」的语义跨表不一致。
  - `--seed` 作用域实证（E8）：random/numpy/torch/fewshot 四类，覆盖完整。
- **缺口**：数据集 revision 未 pin（E10，仅本机 cache 有 `b820378…`）；模型权重无 revision/checksum（modelscope `snapshots/master` 是移动目标）。

### Q3 轮数边界是否记录且可从 training_stats.json 复核？——**主实验：是 ✓；11 月产物：否（字段退化），需从不在 git 的二进制文件恢复**

| 族 | training_stats.json 字段 | epochs/steps/dataset_size 可核性 |
|---|---|---|
| 主 8B（7 月） | method/seed/budget/peak_vram_gb/**epochs:3/dataset_size:2000** | ✓ 直接可核（epochs 由 batch_train.py L75-79 训练后回填）；steps 从 trainer_state.json（max_steps=21）|
| 消融/schemeA（11 月） | method/seed/budget/peak_vram_gb（**无 epochs/dataset_size**） | ✗ HEAD train.py L264-268 丢弃了这两个字段；只能从 trainer_state.json（max_steps=126, epoch=3.0）+ training_args.bin（num_train_epochs=3）恢复，两者均在 gitignore 的 outputs/ 下 |
| 14B（11 月） | 同上退化 | 同上（max_steps=255）；epochs/dataset_size 只存在于未跟踪的 run_b4_qwen14b_seeds.sh 参数里 |

步数账（实测自洽）：主 8B 21=7×3 步（112 序列/epoch）；消融 126=42×3（672 序列/epoch@1024）；14B 255=85×3（1360 序列/epoch@512）。三族打包长度互不相同且只有 11 月两族落盘于 training_args.bin。**修复方向：把 epochs/dataset_size/max_steps/pack_length 恢复进 training_stats.json 并把每 run 台账（csv/json）提交 git**（见修复清单 F3/F4）。

### Q4 断点续训数值等价性——**机制在位 ✓，等价性零验证 ✗**

- 机制：HEAD train.py L243-254 自动扫 checkpoint-* 取最大编号 resume；11 月起 `save_only_model=False` → checkpoint-110/120/126（消融）与 14B checkpoint 内实测含 optimizer.pt/rng_state.pth/scheduler.pt ✓；主实验 7 月 `save_only_model=True` → checkpoint-7/14/21 内**无 optimizer/rng/scheduler**（实测），当时断电只能从 epoch 边界冷启动（优化器状态丢失）——该风险未成为现实（loss 曲线 21 步连续无断口），但说明主实验根本没有可数值等价的续训能力。
- 验证现状：**没有做过任何真实 kill-resume 等价性测试**。`logs/smoke_resume_test.log`（Nov 12）实测为 no-op：resume checkpoint-3（已完成点）→ `train_runtime 0.0029s, train_loss 0.0`，没有中途 kill、没有对照 loss 序列。`numerical_invariance_check.py` 有 --compare 能力但无运行存档。
- 实际交付产物均为无中断完整 run（b4/ablation 各日志 grep 无 resume 事件；trainer_state loss 序列连续）→ **风险未兑现，但工程主张（"支持断电续训"）目前无证据支撑**，论文/README 若声称断点续训能力必须补验证（修复清单 F6）。

### Q5 无人值守链幂等性竞态——**主链顺序正确 ✓；14B seeds 链回归 ✗；跨链 GPU 竞态窗口存在**

- **正确样板**：`run_eval_missing.py` 先跑 `verify_eval_artifact.py`（机读校验 results_*.json 含目标任务指标键）通过后才 `touch .done_{marker}`（L146-151），并有 preflight（L165-182，safety 任务 GPU 前闸门）。`.done` 语义 = 「产物已验证」而非「命令退出码为 0」。这是从 v2.0 真实事故（.done_main 先 touch → mmlu 429 失败被掩盖，脚本头注释 L5 自述）修出来的，**顺序正确**。
- **回归**：`run_eval_b4_qwen14b_seeds.py` L46-47 仅 `p.returncode==0` 即 touch `.done_{task}`，**不调 verify_eval_artifact.py、无 preflight**——rc=0 但产物缺失/半写时产生假 done marker，且 14B 表将据此聚合。同一仓库里两套标准，新脚本退化到事故前版本。
- **flock 语义**：chain_b4_seeds.sh `flock -n /tmp/adaq_b4_seeds.lock` 防自重入 ✓；但 `finalize_b4_then_eval.sh` 用**另一把锁** `/tmp/adaq_b4_finalize.lock`。两条链都在 crontab @reboot（180s/210s）：同一次重启中，若 seed42 adapter 已存在，finalize 立即启动 2 卡评测，同时 chain 启动 2 卡训练 → **无互斥的 GPU 抢占窗口**（本次实际时序错开未触发，属结构性风险）。
- **同机异构项目**：crontab 实测混有 AFLAP 项目 7 条（@reboot + 每 5/10 分钟周期任务，含 v5_d_supervisor/landing_audit/q2_landing_gate/post_reboot_gpu_check），与 AdaQ 8 条共享同 2 张 GPU，**无机器级仲裁**。宿主随机断电史（已知硬事实）下，一次重启可能同时拉起两个项目的 GPU 任务。
- **良性项**：resume_*.sh 家族 tmux has-session 幂等（TOCTOU 后果 = 第二次 new-session 同名失败，无害）；`resume_ablation_schemeA_multiseed.sh`（a00fe41 入库版）等待判据用单层 glob `ls outputs/.../*/adapter_config.json | wc -l ≥6`，正确排除 checkpoint 子目录 ✓——曾发生 OOM 事故的带 bug 旧版（find -name 误数 checkpoint-30/40/50）**从未入库**，入库版本与 commit 描述一致。
- `resume_eval_schemeA_multiseed.sh` count_done 用 `find -name '.done_*' | wc -l` 阈值 12（6 策略×2 任务），marker 仅在 rc==0 后写 → 计数语义成立；但同样无产物校验。

### Q6 环境固化与版本漂移——**固化完全缺失；rank_pattern 语义漂移已用双版本形状实证排除，但必须锁版本**

- **漂移事实**：主实验 adapter（2026-07-31）`peft_version 0.19.1` vs 现环境 `peft 0.21.2`（E5）。11 月全部新产物（消融/schemeA/14B）已是 0.21.2。**同一论文的主表与消融/14B 表由两代 peft 训练**，这必须在论文实现细节中声明。
- **语义等效实证（排除漂移影响）**：
  - 0.19.1 产物：主 adaq rank_pattern {'q_proj':12,'v_proj':4} → safetensors lora_A q[12,4096]/v[4,4096] ✓ 生效；
  - 0.21.2 产物：14B adaq q[7,5120]/v[4,5120] ✓；消融 per_layer 64 条（layers.0 q[31]/v[20]）✓；schemeA q[10]/v[4] ✓；
  - 机制：0.21.2 `get_pattern_key`（E6）= `re.match(rf"(.*\.)?({key})$", key_to_match)` 后缀匹配，与 train.py L61-69 `_to_pattern_key`（去 `.weight`/前导 `model.`）的注释语义一致，短名与层级名两种 key 风格均正确命中。**两代版本对两种 key 风格的张量形状实证等效 → rank_pattern 语义漂移排除。**
  - 残留风险：未锁版本意味着未来 peft 升级仍可能漂移（0.21.x 后续版本改 get_pattern_key 无从察觉）；且 alpha_pattern 仅 per_layer 方案使用（α=2r），per_module_type 复刻主实验只写 rank_pattern（train.py L87-89 注释）——两方案的 α 语义差异要在论文写清。
- **固化缺失清单**：requirements.txt 全 `>=` 下界；无 environment.yml/lock/Dockerfile/pip freeze 存档；README L87 仅 `pip install -r requirements.txt`；现环境是激进的 py3.13+torch2.11+cu130 组合，与 `>=` 下界相距甚远；HF_ENDPOINT=hf-mirror 只在部分 shell 脚本 export（train.py 内无）→ 第三方直接跑 train.py 还要自配网络与 `ADAQ_MODEL_PATH`（默认指向 /mnt/workspace 原机器路径）。
- **模型/数据 pin 缺失**：models/ gitignore（合理），但两个基座模型无 revision/checksum 记录（Qwen 走 modelscope `snapshots/master` 移动目标）；OpenHermes-2.5 `load_dataset` 不传 revision（本机 cache refs/main=`b82037821055c377bed0d495e72e46de3bc72e84` 是唯一事实）。

---

## 3. 修复清单（按优先级；路径+精确改法）

### P0（投稿前必须，决定「数字可复核」成立与否）

- **F1 结果台账入 git。** `.gitignore`：追加 `!results_v2/FINAL_REPORT.md`、`!results_v2_qwen14b*/summary_3seeds.json`、`!results_v2_qwen14b*/REPORT_14B.md`；然后 `git add results_v2/summary.json results_v2/summary.pre-repair.json results_v2/FINAL_REPORT.md`（前两者反选已生效仅未 add）。同时新建 `results_ledger.csv`（列：run, family, task, metric, value, n_samples, train_seed, eval_seed, lm_eval_version, results_json_path, timestamp）由 `scripts/aggregate_results.py`/`aggregate_b4_3seeds.py` 末尾自动追加，并 git add——把「论文每个数字 → 原始 results_*.json」的映射变成可 diff 的单一台账。
- **F2 未跟踪编排层入 git。** `git add scripts/chain_b4_seeds.sh scripts/run_b4_qwen14b_seeds.sh scripts/run_eval_b4_qwen14b_seeds.py scripts/aggregate_b4_3seeds.py ranks_qwen14b.json`；在 ranks_qwen14b.json 同目录补 `ranks_qwen14b.PROVENANCE.md`（生成命令原文：`python adaq-lora-research-pipeline/scripts/adaq_precompute.py --model models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master --budget 512 --r-min 4 --r-max 128`、日期、模型 checksum）。
- **F3 主实验 provenance 文档。** 新建 `docs/main-run-provenance.md`，逐项落盘 training_args.bin 实测值（save_strategy=epoch、save_only_model=True、max_seq_length=None、use_liger_kernel=True、packing=True、batch 4×ga 4、lr 2e-4 cosine warmup 0.03、paged_adamw_8bit、bf16）、反推打包长度（112 序列/epoch ≈8K）、peft 0.19.1、原机器 `/mnt/AdaQ-LoRA`（阿里云 PAI，GPU 型号由作者回填）、950ccc4 快照与真实产物的 `max_length`/`max_seq_length` 矛盾说明、以及「主表数字 = 7 月 adapter × 2026-11 lm_eval 0.4.13 重评」的两段式血统声明。**同步修正 `configs/training_params.yaml`**：`max_seq_length: 2048` 行加注释「主实验实际为 None（≈8K 打包）；2048 为 HEAD 默认；消融=1024；14B=512」。
- **F4 环境固化。** 现环境 `pip freeze > envs/lock-py313-20261115.txt` 入 git；新建 `envs/lock-july-main-run.txt` 只含可考证项（peft==0.19.1，其余标注 UNKNOWN）；requirements.txt 改为双段：`# 论文数字所用（2026-11 重评/新训练）` 段用 == 精确锁定 peft/transformers/torch/trl/lm_eval/numpy/bitsandbytes/datasets/liger-kernel 十项，`>=` 段仅留给非关键依赖。README「复现」节写明：主表训练环境不可完全重建（F3），可重建的是评测与 11 月各表。
- **F5 training_stats.json 字段恢复。** `scripts/train.py` L264-268 stats dict 增补：`"epochs": args.epochs, "dataset_size": args.dataset_size, "max_seq_length": args.max_length, "effective_batch": args.batch_size*args.grad_accum`，并从 `trainer.state.max_steps`/`global_step` 补 `"max_steps"`；对已完成的 11 月各 run 跑一次性回填脚本（从各自 trainer_state.json/training_args.bin 读取，标注 backfilled=true）。

### P1（强烈建议，决定工程主张可信度）

- **F6 真实续训等价性验证。** 新建 `scripts/tests/test_resume_equivalence.sh`：小配置（100 样本、60 步）跑两次——A 无中断完整 60 步；B 在第 30 步 `kill -9` 后依赖 train.py 自动 resume 跑完；对比 A/B 的 trainer_state loss 序列逐步差（验收：|Δloss|/loss < 1e-3 或记录 bf16 非确定性底噪后给出阈值依据），输出存 `logs/resume_equivalence_<date>.log` 并在 docs 引用。现有 `logs/smoke_resume_test.log` 是 no-op（train_runtime 0.0029s），不得作为等价性证据引用。
- **F7 14B seeds 评测幂等回归修复。** `scripts/run_eval_b4_qwen14b_seeds.py` L46-47：在 `p.returncode==0` 与 `touch .done_{task}` 之间插入 `subprocess.call([PY, "scripts/verify_eval_artifact.py", out_dir, task])`，非 0 则打 FAIL 不写 marker（对齐 run_eval_missing.py L146-151 样板）。同改 `run_eval_b4_qwen14b.py`。
- **F8 跨链 GPU 互斥。** `scripts/finalize_b4_then_eval.sh` 与 `scripts/chain_b4_seeds.sh` 的评测/训练 GPU 段统一改用共享锁 `flock /tmp/adaq_gpu.lock`（阻塞式，非 -n），或在 finalize 启动评测前 `pgrep -f chain_b4_seeds` 让行。README 运维节记录：本机与 AFLAP 项目共享 GPU 且无仲裁，重启风暴时人工介入顺序。
- **F9 数据集/模型 pin。** `scripts/train.py` L174 `load_dataset("teknium/OpenHermes-2.5", split=..., revision="b82037821055c377bed0d495e72e46de3bc72e84")`（E10 实测本机 cache revision）；`docs/main-run-provenance.md` 记录两个基座的 sha256（config.json + 首个 safetensors 分片）与 modelscope/HF 来源路径。
- **F10 seed 口径入文档。** `docs/eval-protocol-v2.md` 增「种子口径」节：主表评测 seed=训练 seed（run_eval_missing.py L130，方差=训练+评测混合，贪心解码下评测分量可忽略）；消融/schemeA/14B 评测 seed 固定 42（方差=纯训练）；14B 表为 NF4 4bit 基座推理 vs 主表 bf16 基座——跨表协议差异在论文各表 caption 注明。

### P2（应修，公平性与卫生）

- **F11 AdaLoRA 基线注记或重跑。** 论文中 adalora 行必须标注「init_r=16 未剪枝（total_step=200 > 实际 21 步），可训练参数 6.8M ≈ 他法 2 倍」；更优做法：train.py L52 改 `total_step=` 实际 max_steps（或从 epochs/steps 计算）后重跑 adalora 三 seed，否则从对比表移除。
- **F12 14B seed42 拓扑一致性。** 论文若报 14B peak_vram 或主张严格同配置：用 `CUDA_VISIBLE_DEVICES=0` 单卡重跑 seed42（run_b4_qwen14b_seeds.sh 路径），或在表中注明 seed42 为双卡切分拓扑、vram 读数（6.67GB）不可比。
- **F13 run_b4_qwen14b.sh 日志与死代码。** L39/44 去掉 `| tail -15` 改 `> "$LOGDIR/b4_${method}_s${seed}.log" 2>&1`（对齐 seeds 版脚本 L31/36），FAIL 分支改 `if ! python ...; then`；L29 SKIP 判据补 `&& adapter_model.safetensors`（对齐 seeds 版 L21）。
- **F14 numerical_invariance_check.py 补跑存档。** 按 docs/training-optimization-plan.md 铁律 3：baseline vs liger、baseline vs flash_attn 各 100 步对照，输出入 logs/ 并在 F3 文档引用（主实验用了 liger，此对照是解释两代数值差异的唯一手段）。
- **F15 aggregate_system_metrics.py 去硬编码。** `base_params=8030261248` 改为运行时从 model config 计算，或注明来源与 smoke 实测差值（8,033,669,120）。

---

## 4. 三句话总评

1. 这个项目的**评测侧重跑工程是真材实料**——统一重评、机读产物校验后才写 marker、报告自我禁止 SOTA/显著性话术、n=3 逐种子明细可查，这套纪律在同类工作里属上游水平。
2. 但**训练侧的复现链条在主实验处断裂**：7 月真实配置（liger=True、≈8K 打包、peft 0.19.1、PAI 原机器）与仓库内所有已提交代码/配置互相矛盾，结果文件与 14B 编排层全部游离在 git 外，第三方今天 clone 仓库无法重建主表任何一个训练数字。
3. 好消息是断点全部可用**台账化手段补救**（F1-F5 均为文档/入库/字段回填，不需重训主实验），完成 P0+P1 后本项目可达「数字可核对、口径可审计、新表可重跑」的可发表复现标准。

---

## 附录 A：关键实测摘录

```text
# training_args.bin（torch.load weights_only=False 实测）
outputs/adapters/adaq_seed42/checkpoint-21        : seed=42 max_seq_length=None save_strategy=EPOCH
  save_only_model=True use_liger_kernel=True packing=True b4×ga4 lr2e-4 cosine warmup0.03
  paged_adamw_8bit bf16 save_steps=500 logging10 workers4 full_determinism=False
outputs/ablation_adapters/equal_weight/ckpt-126   : max_seq_length=1024 save_strategy=STEPS
  save_steps=10 limit=3 save_only_model=False use_liger_kernel=False
outputs/adapters_qwen14b/adaq_seed42/ckpt-255     : b2×ga8 max_seq_length=512 STEPS/10/3/False liger=False
# 步数账：21=7×3(112 序列/epoch)；126=42×3(672@1024)；255=85×3(1360@512)

# safetensors lora_A 形状（rank_pattern 实效铁证）
主 adaq_seed42 (peft0.19.1): q_proj[12,4096] v_proj[4,4096]   # {'q_proj':12,'v_proj':4}
主 adalora_seed42          : lora_A[16,4096]                  # init_r=16 未剪枝（total_step=200>21）
消融 equal_weight (0.21.2) : layers.0 q[31] v[20] / layers.15 q[10] v[4]（64 条 per_layer）
14B adaq (0.21.2)          : q_proj[7,5120] v_proj[4,5120]

# peft 0.21.2 匹配语义（site-packages 源码）
peft/utils/other.py:1524-1532  get_pattern_key: re.match(rf"(.*\.)?({key})$", key_to_match)
peft/tuners/lora/model.py:309-312  r = rank_pattern.get(get_pattern_key(...), lora_config.r)

# 评测 seed 作用域（results_v2/adaq_seed42/eval_main.log:4）
Setting random seed to 42 | Setting numpy seed to 42 | Setting torch manual seed to 42 | Setting fewshot manual seed to 42
model_args: pretrained=/home/ubuntu/AdaQ-LoRA/models/Meta-Llama-3-8B, peft=.../adaq_seed42, dtype=bfloat16   # bf16 基座，无 load_in_4bit

# 环境（pip list 实测，py3.13）
peft 0.21.2 | transformers 4.57.6 | torch 2.11.0+cu130 | trl 0.12.0 | lm_eval 0.4.13
numpy 2.4.3 | bitsandbytes 0.50.2 | accelerate 1.15.0 | datasets 3.6.0 | liger_kernel 0.5.10

# 数据集 revision（本机 cache 实测）
~/.cache/huggingface/hub/datasets--teknium--OpenHermes-2.5/refs/main = b82037821055c377bed0d495e72e46de3bc72e84

# 评审时刻运行状态（只读）
chain_b4_seeds.sh PID 1196644 存活；2×lm_eval 进程（seed2025 gsm8k）；
results_v2_qwen14b_seed2025/{adaq,qlora}/.done_mmlu (Nov 15 15:32/15:33)；seed3407 评测未开始
```

*评审完成：2026-11-15 · 全程只读 · 未干扰任何训练/评测进程*
