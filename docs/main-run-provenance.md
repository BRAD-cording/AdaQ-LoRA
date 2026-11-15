# 主实验血统档案（main-run provenance）

> [2026-11-16 落档] 按 review/repro-eng.md P0 与 review/fair-baseline.md F13 要求，如实记录 8B 主表的**两段式血统**。本档案是诚实披露，不掩饰断裂。

## 1. 两段式时间线（已证实）
- **训练段（2026-07-31 前后）**：12 个 8B adapter（qlora/dora/adalora/adaq × seed42/3407/2025）由**当时代码**训练，与现 repo 代码存在实质差异——training_args.bin 实测 `use_liger_kernel=True`、`max_seq_length=None`（packing 生效，7 步/epoch 反推 ≈8K token/步）、`save_strategy=EPOCH`、`max_steps=21`。现 repo train.py 已多轮演进（save_steps=10、--rank-scheme、seq1024 等），**不能逐位重建当时的训练过程**。
- **评测段（2026-11）**：同一批 adapter 用 lm_eval 0.4.13、BF16 干净基座协议（model_args 无 load_in_4bit）重评，产出 results_v2/。
- 含义：主表数字=「7 月训练产物 × 11 月评测口径」，任何复现者用现 repo 代码重训不会得到逐位相同结果（review/stats-protocol.md F1：21 步对方法比较无功效）。

## 2. 已知血统缺陷（未修复，如实列出）
- adapter_config.json 的 `base_model_name_or_path=/mnt/workspace/.cache/modelscope/...` 在本机不存在（异机训练迁移而来）；全库无基座 safetensors sha256 记录 → **基座同一性只能靠路径约定，无法密码学验证**。
- ranks_qwen14b.json 生成于异机 /mnt/workspace（本机重测 rel_error 差第 4 位小数：.092642 vs .092598）。
- 数据集未 pin revision：HF cache 现存 [cais___mmlu EleutherAI___wikitext_document_level generator google___if_eval openai___gsm8k ]（snapshot 2026-11-16）。
- requirements 历史上全为 >= 无锁；本 commit 起以 requirements.lock.txt（pip freeze，pytorch_gpu env）为准。

## 3. 硬件更正
所有文档中「A10」记载有误：实为 **2×NVIDIA GeForce RTX 4090D 24GB**（nvidia-smi 实测）。本 commit 已批量更正 docs/。

## 4. 版本锚点
- 本档案落档时 git HEAD: b0d4dde3e65583d7d1c5c8cc181e1e271864c27b
- 评测: lm_eval 0.4.13 / peft 0.21.2（主实验训练期为 0.19.1，rank_pattern 后缀正则语义两代一致，已排除漂移）
- 评测种子口径：主表 eval seed=训练 seed（run_eval_missing.py:130）；消融/14B eval seed 固定 42（run_eval_ablation.py:78）→ 混合方差，跨表比较禁止（eval-protocol-v2.md §9.3）。

## 5. 补救路径（GPU 排期待用户拍板）
主表两条路：(a) 按现 repo 代码+锁环境重训 12 run（≥200 步收敛配置）；(b) 保留现主表但降级为「历史产物附录」，主证据链切换到 126 步收敛的消融表+14B 表。评审倾向 (a)（review/stats-protocol.md P0-1）。
