#!/usr/bin/env bash
# Full overnight pipeline: SFT → merge → GRPO → merge + reports.
# Must be started with nohup/tmux so it survives Cursor disconnect.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-/venv/main/bin/python}"
TORCHRUN="${TORCHRUN:-/venv/main/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BASE_MODEL="${BASE_MODEL:-/workspace/models/google-gemma-3-12b-it}"

export BASE_MODEL
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

mkdir -p outputs/logs reports
# Detach from any controlling terminal / agent session.
cd "$(pwd)"
trap '' HUP

LOG="outputs/logs/full_pipeline.log"
exec >>"${LOG}" 2>&1

echo "============================================================"
echo "PIPELINE_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "BASE_MODEL=${BASE_MODEL}"
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "pid=$$ ppid=$PPID"
echo "============================================================"

# Avoid colliding with a leftover empty adapter dir
if [[ -d outputs/sft_adapter && ! -f outputs/sft_adapter/adapter_config.json ]]; then
  rm -rf outputs/sft_adapter
fi

# ---------- SFT ----------
echo "SFT_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
START=$(date +%s)
set +e
"${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" train_sft.py \
  2>&1 | tee outputs/logs/sft_train.log
EC=${PIPESTATUS[0]}
set -e
END=$(date +%s)
echo "SFT_END $(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT:${EC} ELAPSED_SEC:$((END-START))" | tee -a outputs/logs/sft_train.log
if [[ "${EC}" -ne 0 ]]; then
  echo "SFT failed"; exit "${EC}"
fi
if [[ ! -f outputs/sft_adapter/adapter_config.json ]]; then
  echo "ERROR: missing sft_adapter"; exit 2
fi

"${PYTHON}" extract_metrics.py outputs/logs/sft_train.log -o outputs/sft_metrics.jsonl || true
"${PYTHON}" generate_stage_report.py \
  --stage sft \
  --log outputs/logs/sft_train.log \
  --metrics outputs/sft_metrics.jsonl \
  --out "reports/SFT_작업_과정_및_결과.md" || true

# ---------- SFT merge ----------
echo "MERGE_SFT_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
START=$(date +%s)
"${PYTHON}" merge_adapter.py --stage sft
EC=$?
END=$(date +%s)
echo "MERGE_SFT_END $(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT:${EC} ELAPSED_SEC:$((END-START))"
[[ "${EC}" -eq 0 ]] || exit "${EC}"
echo "" >> "reports/SFT_작업_과정_및_결과.md"
echo "### Merge 결과" >> "reports/SFT_작업_과정_및_결과.md"
echo "- merged path: \`outputs/sft_merged\`" >> "reports/SFT_작업_과정_및_결과.md"
echo "- merge wall-clock: $((END-START))s" >> "reports/SFT_작업_과정_및_결과.md"

# ---------- GRPO ----------
echo "GRPO_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
START=$(date +%s)
set +e
"${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" train_grpo.py \
  2>&1 | tee outputs/logs/grpo_train.log
EC=${PIPESTATUS[0]}
set -e
END=$(date +%s)
echo "GRPO_END $(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT:${EC} ELAPSED_SEC:$((END-START))" | tee -a outputs/logs/grpo_train.log

"${PYTHON}" extract_metrics.py outputs/logs/grpo_train.log -o outputs/grpo_metrics.jsonl || true
"${PYTHON}" generate_stage_report.py \
  --stage grpo \
  --log outputs/logs/grpo_train.log \
  --metrics outputs/grpo_metrics.jsonl \
  --out "reports/GRPO_작업_과정_및_결과.md" || true

if [[ "${EC}" -ne 0 ]]; then
  echo "GRPO failed"; exit "${EC}"
fi
if [[ ! -f outputs/grpo_adapter/adapter_config.json ]]; then
  echo "ERROR: missing grpo_adapter"; exit 4
fi

# ---------- GRPO merge ----------
echo "MERGE_GRPO_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
START=$(date +%s)
"${PYTHON}" merge_adapter.py --stage grpo
EC=$?
END=$(date +%s)
echo "MERGE_GRPO_END $(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT:${EC} ELAPSED_SEC:$((END-START))"
echo "" >> "reports/GRPO_작업_과정_및_결과.md"
echo "### Merge 결과" >> "reports/GRPO_작업_과정_및_결과.md"
echo "- merged path: \`outputs/grpo_merged\`" >> "reports/GRPO_작업_과정_및_결과.md"
echo "- merge wall-clock: $((END-START))s" >> "reports/GRPO_작업_과정_및_결과.md"

echo "PIPELINE_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Reports: reports/SFT_작업_과정_및_결과.md reports/GRPO_작업_과정_및_결과.md"
