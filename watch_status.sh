#!/usr/bin/env bash
watch -n 5 '
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
echo
echo "=== GRPO progress ==="
tail -c 4000 /workspace/fdsp_summary_format/outputs/logs/grpo_train.log | tr "\r" "\n" | grep -E "s/it|it/s" | tail -5
'
