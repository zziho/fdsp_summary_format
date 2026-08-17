#!/usr/bin/env bash
# Wait for the in-flight SFT job, then merge → GRPO → merge + write reports.
# Safe to run under nohup/tmux; does NOT restart SFT.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-/venv/main/bin/python}"
TORCHRUN="${TORCHRUN:-/venv/main/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BASE_MODEL="${BASE_MODEL:-/workspace/models/google-gemma-3-12b-it}"
SFT_PID="${SFT_PID:-14063}"
SFT_TERMINAL_LOG="${SFT_TERMINAL_LOG:-/root/.cursor/projects/workspace/terminals/186481.txt}"

export BASE_MODEL
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
# Keep INFO for first collectives evidence; can be noisy but useful for NVLink proof.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

mkdir -p outputs/logs reports

LOG="outputs/logs/continue_after_sft.log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo "CONTINUE_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "BASE_MODEL=${BASE_MODEL}"
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "Waiting for SFT pid=${SFT_PID} (or train_sft processes) ..."
echo "============================================================"

sft_still_running() {
  if [[ -n "${SFT_PID}" ]] && kill -0 "${SFT_PID}" 2>/dev/null; then
    return 0
  fi
  pgrep -f '/venv/main/bin/python3 -u train_sft.py' >/dev/null 2>&1 && return 0
  pgrep -f 'torchrun .*train_sft.py' >/dev/null 2>&1 && return 0
  return 1
}

while sft_still_running; do
  echo "[wait] SFT still running @ $(date -u +%H:%M:%SZ)"
  sleep 60
done

echo "SFT processes gone @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Capture SFT terminal log if present
if [[ -f "${SFT_TERMINAL_LOG}" ]]; then
  cp -f "${SFT_TERMINAL_LOG}" outputs/logs/sft_train_terminal.txt
  # Prefer a clean training log without metadata if SFT_END present
  if grep -q 'SFT_END' "${SFT_TERMINAL_LOG}"; then
    cp -f "${SFT_TERMINAL_LOG}" outputs/logs/sft_train.log
  else
    cp -f "${SFT_TERMINAL_LOG}" outputs/logs/sft_train.log
  fi
fi

if [[ ! -f outputs/sft_adapter/adapter_config.json ]]; then
  echo "ERROR: SFT finished but outputs/sft_adapter/adapter_config.json missing"
  exit 2
fi
echo "Found SFT adapter: outputs/sft_adapter"

# --- metrics + SFT report (pre-merge) ---
"${PYTHON}" extract_metrics.py outputs/logs/sft_train.log -o outputs/sft_metrics.jsonl || true
"${PYTHON}" generate_stage_report.py \
  --stage sft \
  --log outputs/logs/sft_train.log \
  --metrics outputs/sft_metrics.jsonl \
  --out "reports/SFT_작업_과정_및_결과.md" || true

# --- SFT merge ---
echo "MERGE_SFT_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
START=$(date +%s)
"${PYTHON}" merge_adapter.py --stage sft
EC=$?
END=$(date +%s)
echo "MERGE_SFT_END $(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT:${EC} ELAPSED_SEC:$((END-START))"
if [[ "${EC}" -ne 0 ]]; then
  exit "${EC}"
fi
if [[ ! -d outputs/sft_merged ]]; then
  echo "ERROR: sft_merged missing"
  exit 3
fi

# Refresh SFT report after merge note
"${PYTHON}" generate_stage_report.py \
  --stage sft \
  --log outputs/logs/sft_train.log \
  --metrics outputs/sft_metrics.jsonl \
  --out "reports/SFT_작업_과정_및_결과.md" || true
echo "" >> "reports/SFT_작업_과정_및_결과.md"
echo "### Merge 결과" >> "reports/SFT_작업_과정_및_결과.md"
echo "- merged path: \`outputs/sft_merged\`" >> "reports/SFT_작업_과정_및_결과.md"
echo "- merge wall-clock: $((END-START))s" >> "reports/SFT_작업_과정_및_결과.md"

# Optional short activation-ckpt A/B smoke (2 steps) — skip if SKIP_AB=1
if [[ "${SKIP_AB:-0}" != "1" ]]; then
  echo "AB_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mkdir -p outputs/activation_ckpt_ab
  (
    export OUTPUT_ROOT="/workspace/fdsp_summary_format/outputs/activation_ckpt_ab/on"
    # on = default (activation_checkpointing true in common.fsdp_config)
    START=$(date +%s)
    "${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      train_sft.py --limit 32 --max-steps 2 \
      > outputs/logs/ab_act_ckpt_on.log 2>&1 || true
    echo "AB_ON_ELAPSED $(( $(date +%s) - START ))" | tee -a outputs/activation_ckpt_ab/summary.txt
  )
  echo "AB_END_NOTE: full off-toggle needs code change; recorded ON baseline only (SKIP_AB=0)." \
    | tee -a outputs/activation_ckpt_ab/summary.txt
fi

# --- GRPO ---
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
  echo "GRPO failed with ${EC}; skip GRPO merge"
  exit "${EC}"
fi

if [[ ! -f outputs/grpo_adapter/adapter_config.json ]]; then
  echo "ERROR: grpo_adapter missing after successful-looking exit"
  exit 4
fi

# --- GRPO merge ---
echo "MERGE_GRPO_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
START=$(date +%s)
"${PYTHON}" merge_adapter.py --stage grpo
EC=$?
END=$(date +%s)
echo "MERGE_GRPO_END $(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT:${EC} ELAPSED_SEC:$((END-START))"
"${PYTHON}" generate_stage_report.py \
  --stage grpo \
  --log outputs/logs/grpo_train.log \
  --metrics outputs/grpo_metrics.jsonl \
  --out "reports/GRPO_작업_과정_및_결과.md" || true
echo "" >> "reports/GRPO_작업_과정_및_결과.md"
echo "### Merge 결과" >> "reports/GRPO_작업_과정_및_결과.md"
echo "- merged path: \`outputs/grpo_merged\`" >> "reports/GRPO_작업_과정_및_결과.md"
echo "- merge wall-clock: $((END-START))s" >> "reports/GRPO_작업_과정_및_결과.md"

echo "CONTINUE_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Artifacts: outputs/sft_adapter outputs/sft_merged outputs/grpo_adapter outputs/grpo_merged"
echo "Reports: reports/SFT_작업_과정_및_결과.md reports/GRPO_작업_과정_및_결과.md"
