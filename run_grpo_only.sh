#!/usr/bin/env bash
# Continue from SFT-merged: GRPO → merge → reports.
# Survives disconnect when started inside tmux.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-/venv/main/bin/python}"
TORCHRUN="${TORCHRUN:-/venv/main/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

export BASE_MODEL="${BASE_MODEL:-/workspace/models/google-gemma-3-12b-it}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Safer than the OOM'd pd=8 run:
# - EOS = <end_of_turn> (set in train_grpo.py) so completions stop early
# - pd=4 microbatch for logprob logits; gen_batch=16 keeps prompts/step=4
# - grad_accum=2 keeps optimizer batch similar while cutting peak VRAM
export GRPO_PER_DEVICE_BATCH="${GRPO_PER_DEVICE_BATCH:-4}"
export GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-2}"
export GRPO_GENERATION_BATCH="${GRPO_GENERATION_BATCH:-16}"
export GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-4}"
export GRPO_EOS_TOKEN="${GRPO_EOS_TOKEN:-<end_of_turn>}"
export GRPO_CHECKPOINT_EVERY="${GRPO_CHECKPOINT_EVERY:-20}"

mkdir -p outputs/logs reports
trap '' HUP

LOG="outputs/logs/grpo_only_pipeline.log"
exec >>"${LOG}" 2>&1

echo "============================================================"
echo "GRPO_PIPELINE_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "BASE_MODEL=${BASE_MODEL}"
echo "GRPO_PER_DEVICE_BATCH=${GRPO_PER_DEVICE_BATCH} GRPO_GRAD_ACCUM=${GRPO_GRAD_ACCUM}"
echo "GRPO_GENERATION_BATCH=${GRPO_GENERATION_BATCH} GRPO_NUM_GENERATIONS=${GRPO_NUM_GENERATIONS}"
echo "GRPO_EOS_TOKEN=${GRPO_EOS_TOKEN} GRPO_CHECKPOINT_EVERY=${GRPO_CHECKPOINT_EVERY}"
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "pid=$$"
echo "============================================================"

if [[ ! -f outputs/sft_merged/config.json ]]; then
  echo "ERROR: missing outputs/sft_merged"; exit 2
fi

# Archive previous failed run artifacts (keep for the report) then clear incomplete adapter.
stamp=$(date -u +%Y%m%dT%H%M%SZ)
if [[ -f outputs/logs/grpo_train.log ]]; then
  mv outputs/logs/grpo_train.log "outputs/logs/grpo_train_failed_${stamp}.log" || true
fi
if [[ -f outputs/grpo_metrics.jsonl ]]; then
  mv outputs/grpo_metrics.jsonl "outputs/grpo_metrics_failed_${stamp}.jsonl" || true
fi
rm -rf outputs/grpo_adapter

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
  --out "reports/GRPO_REPORT.md" || true

if [[ "${EC}" -ne 0 ]]; then
  echo "GRPO failed"; exit "${EC}"
fi
if [[ ! -f outputs/grpo_adapter/adapter_config.json ]]; then
  echo "ERROR: missing grpo_adapter"; exit 4
fi

echo "MERGE_GRPO_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
START=$(date +%s)
"${PYTHON}" merge_adapter.py --stage grpo
EC=$?
END=$(date +%s)
echo "MERGE_GRPO_END $(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT:${EC} ELAPSED_SEC:$((END-START))"
[[ "${EC}" -eq 0 ]] || exit "${EC}"

{
  echo ""
  echo "### Merge 결과"
  echo "- merged path: \`outputs/grpo_merged\`"
  echo "- merge wall-clock: $((END-START))s"
} >> "reports/GRPO_REPORT.md"

echo "GRPO_PIPELINE_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Reports: reports/GRPO_REPORT.md"
