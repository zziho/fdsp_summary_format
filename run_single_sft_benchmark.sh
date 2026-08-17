#!/usr/bin/env bash
# Single-GPU (physical GPU0) SFT baseline with continuous GPU telemetry.
set -uo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-/venv/main/bin/python}"
STEPS="${SINGLE_SFT_STEPS:-25}"
WARMUP="${SINGLE_SFT_WARMUP:-5}"

mkdir -p single_outputs single_reports
rm -rf single_outputs/trainer
rm -f single_outputs/nvidia_smi.csv single_outputs/train.log

export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL="${BASE_MODEL:-/workspace/models/google-gemma-3-12b-it}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export LORA_RANK="${LORA_RANK:-16}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG="single_outputs/train.log"
GPU_CSV="single_outputs/nvidia_smi.csv"

{
  echo "SINGLE_SFT_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "BASE_MODEL=${BASE_MODEL}"
  echo "MAX_LENGTH=${MAX_LENGTH} LORA_RANK=${LORA_RANK}"
  echo "STEPS=${STEPS} EXCLUDED_WARMUP_STEPS=${WARMUP}"
} | tee "${LOG}"

echo "timestamp,index,name,memory.used.MiB,memory.total.MiB,utilization.gpu,power.draw.W" > "${GPU_CSV}"
(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits \
      | awk -F', ' '$2 == 0 { print }' >> "${GPU_CSV}"
    sleep 1
  done
) &
MONITOR_PID=$!

START=$(date +%s)
set +e
"${PYTHON}" train_sft_single.py \
  --max-steps "${STEPS}" \
  --warmup-benchmark-steps "${WARMUP}" \
  2>&1 | tee -a "${LOG}"
EC=${PIPESTATUS[0]}
set -e
END=$(date +%s)

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true

echo "SINGLE_SFT_END $(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT:${EC} ELAPSED_SEC:$((END-START))" \
  | tee -a "${LOG}"

"${PYTHON}" generate_single_sft_report.py \
  --exit-code "${EC}" \
  --elapsed-sec "$((END-START))" \
  --out single_reports/SINGLE_GPU_SFT_작업_과정_및_결과.md \
  | tee -a "${LOG}"

exit "${EC}"
