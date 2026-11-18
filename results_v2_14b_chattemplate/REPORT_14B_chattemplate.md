# B5：14B GSM8K chat-template 复评（排除格式漂移 artifact）

命令：gsm8k --num_fewshot 4 --apply_chat_template --fewshot_as_multiturn（gsm8k README 推荐，复刻官方口径）；NF4 评测同原。

| arm | seed | 原 flex(无chat) | chat flex | chat strict | Δflex(chat−原) |
|---|---|---|---|---|---|
| adaq | 42 | 0.6588 | 0.7331 | 0.0000 | +7.43pt |
| adaq | 2025 | 0.5118 | 0.7074 | 0.0000 | +19.56pt |
| adaq | 3407 | 0.5603 | 0.7111 | 0.0008 | +15.09pt |
| qlora | 42 | 0.5254 | 0.7157 | 0.0000 | +19.03pt |
| qlora | 2025 | 0.4215 | 0.7096 | 0.0000 | +28.81pt |
| qlora | 3407 | 0.5178 | 0.7043 | 0.0008 | +18.65pt |

## chat-template 下 adaq−qlora 配对差（flex）
- seed42: +1.74pt
- seed2025: -0.23pt
- seed3407: +0.68pt

均值 +0.73pt，符号不一致。

**判定**：chat-template 下若差仍存且同号 → 格式 artifact 排除，R1 信号更硬；若缩小/消失 → 原 +8.87pt 系格式 artifact，R1 撤回。（噪声底 N=4.09pt 仍适用）
