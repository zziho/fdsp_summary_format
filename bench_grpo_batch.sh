#!/usr/bin/env bash
# Measure GRPO seconds-per-prompt for several per-device batch sizes.
# Each config runs 2 optimizer steps; the last step is the steady-state sample.
set -uo pipefail

cd "$(dirname "$0")"

TORCHRUN="${TORCHRUN:-/venv/main/bin/torchrun}"
PYTHON="${PYTHON:-/venv/main/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CONFIGS="${CONFIGS:-8 16}"

export BASE_MODEL="${BASE_MODEL:-/workspace/models/google-gemma-3-12b-it}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

mkdir -p outputs/logs outputs/bench
RESULT=outputs/bench/grpo_batch_bench.tsv
printf 'per_device\tprompts_per_step\tstep_sec\tsec_per_prompt\tpeak_gpu0_mib\tpeak_gpu1_mib\tstatus\n' > "${RESULT}"

for PD in ${CONFIGS}; do
  LOG="outputs/logs/bench_grpo_pd${PD}.log"
  BENCH_ROOT="/workspace/fdsp_summary_format/outputs/bench/pd${PD}"
  rm -rf "${BENCH_ROOT}"
  mkdir -p "${BENCH_ROOT}"
  # Isolate provenance/adapter writes but reuse the real merged SFT model.
  ln -sfn /workspace/fdsp_summary_format/outputs/sft_merged "${BENCH_ROOT}/sft_merged"

  echo "=== benchmarking per_device=${PD} $(date -u +%H:%M:%SZ) ===" | tee -a outputs/logs/bench_grpo.log
  GRPO_PER_DEVICE_BATCH="${PD}" OUTPUT_ROOT="${BENCH_ROOT}" \
    "${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" train_grpo.py \
      --max-steps 2 > "${LOG}" 2>&1
  EC=$?

  PPS=$(rg -o 'prompts/step=([0-9]+)' -r '$1' "${LOG}" | head -1)
  STEP=$(rg -o '([0-9.]+)s/it' -r '$1' "${LOG}" | tail -1)

  read -r G0 G1 <<<"$("${PYTHON}" - "${BENCH_ROOT}/provenance" <<'PY'
import csv, sys
from pathlib import Path
peak = {}
for path in Path(sys.argv[1]).glob("grpo_nvidia_smi_*.csv"):
    with path.open() as handle:
        for row in csv.DictReader(handle):
            try:
                idx = int(row["index"])
                used = int(row["memory.used.MiB"])
            except (KeyError, TypeError, ValueError):
                continue
            peak[idx] = max(peak.get(idx, 0), used)
print(peak.get(0, 0), peak.get(1, 0))
PY
)"

  STATUS=ok
  if [[ "${EC}" -ne 0 ]]; then
    if rg -qi 'out of memory' "${LOG}"; then STATUS=oom; else STATUS="fail(${EC})"; fi
  fi

  SPP=""
  if [[ -n "${STEP}" && -n "${PPS}" && "${PPS}" -gt 0 ]]; then
    SPP=$(awk -v s="${STEP}" -v p="${PPS}" 'BEGIN{printf "%.1f", s/p}')
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${PD}" "${PPS:-?}" "${STEP:-?}" "${SPP:-?}" "${G0:-?}" "${G1:-?}" "${STATUS}" | tee -a "${RESULT}"

  # Free the 23GB symlink target reference and adapter junk, keep provenance/logs.
  rm -rf "${BENCH_ROOT}/grpo_adapter" "${BENCH_ROOT}/sft_merged"
  sleep 10
done

echo "--- benchmark table ---"
cat "${RESULT}"
echo "BENCH_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
