#!/usr/bin/env bash
# 幂等恢复 AdaQ-LoRA 全量补跑。只写不自动调度；由 goal 续跑轮在检测到卡住时调用。
#
# 背景：logs/repair_all.sh 是 setsid nohup 后台自跑。若远端 OOM / SSH 断连 /
# 机器重启把进程干掉，repair_all.status 不会有 ALL DONE，而 goal 续跑轮会在
# 空等里耗尽。本脚本提供一条命令的确定性恢复。
#
# 安全性：
#   * 先 pgrep 检查 repair_all.sh / run_eval_missing.py 是否还活着，活着就什么都不做
#   * 重启前不删除任何产物、不重置 marker、不改 .gitignore / 评分器 / 测试
#   * run_eval_missing.py 自身按「实际产物里有哪些 task」生成缺失清单，幂等
#   * 补跑产物是新增时间戳 json，不会覆盖旧文件；聚合器按 mtime 去重
set -uo pipefail
cd /home/ubuntu/AdaQ-LoRA
PY=/home/ubuntu/miniconda3/envs/pytorch_gpu/bin/python
STATUS=logs/repair_all.status

alive() {
  pgrep -f "bash logs/repair_all.sh" >/dev/null 2>&1 && return 0
  pgrep -f "python scripts/run_eval_missing.py" >/dev/null 2>&1 && return 0
  pgrep -f "lm_eval --model" >/dev/null 2>&1 && return 0
  return 1
}

if alive; then
  echo "[$(date '+%F %T')] 补跑仍在运行，不做任何改动。状态："
  tail -3 "$STATUS" 2>/dev/null
  exit 0
fi

echo "[$(date '+%F %T')] 检测到无活跃补跑进程，检查交付判据…"
"$PY" scripts/check_eval_status.py > /tmp/resume_check.out 2>&1
rc=$?
head -1 /tmp/resume_check.out; tail -1 /tmp/resume_check.out
if [[ $rc -eq 0 ]]; then
  echo "[$(date '+%F %T')] 判据已 PASS，无需恢复。"
  exit 0
fi

echo "[$(date '+%F %T')] 判据仍 INCOMPLETE，幂等重启缺失任务（双卡）。"
mkdir -p logs
setsid nohup env OUT_DIR=results_v2 WANT=mmlu,gsm8k,safety GPUS=0,1 AGGREGATE=1 \
  "$PY" scripts/run_eval_missing.py >> logs/repair_resume.log 2>&1 < /dev/null &
echo "[$(date '+%F %T')] 已启动 resume pid=$!；日志 logs/repair_resume.log"
echo "下一轮核验：tail -5 logs/repair_resume.log && python scripts/check_eval_status.py"
