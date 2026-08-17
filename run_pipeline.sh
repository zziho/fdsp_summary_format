#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-/venv/main/bin/python}"
TORCHRUN="${TORCHRUN:-/venv/main/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Prefer NVLink P2P (and IB if present). Override to 1 only if this host hangs.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

"${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" train_sft.py
"${PYTHON}" merge_adapter.py --stage sft
"${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" train_grpo.py
"${PYTHON}" merge_adapter.py --stage grpo

echo "Pipeline complete: outputs/grpo_merged"
