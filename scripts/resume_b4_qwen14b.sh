#!/bin/bash
# P1-B4 resume: tmux 幂等启动 Qwen2.5-14B 训练
tmux has-session adaq_b4_qwen14b 2>/dev/null && exit 0
tmux new-session -d -s adaq_b4_qwen14b "bash /home/ubuntu/AdaQ-LoRA/scripts/run_b4_qwen14b.sh 2>&1 | tee /home/ubuntu/AdaQ-LoRA/logs/b4_qwen14b.log"
echo "resume_b4: started tmux adaq_b4_qwen14b"
