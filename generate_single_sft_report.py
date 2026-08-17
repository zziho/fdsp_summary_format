#!/usr/bin/env python3
"""Generate the single-GPU SFT benchmark report from captured telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "single_outputs"


def read_events() -> list[dict]:
    path = OUTPUT / "step_metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def read_gpu() -> list[dict]:
    path = OUTPUT / "nvidia_smi.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value, default=0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def fmt(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--elapsed-sec", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    cli = parser.parse_args()

    events = read_events()
    gpu = read_gpu()
    log_path = OUTPUT / "train.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

    config = next((r for r in events if r.get("event") == "config"), {})
    model = next((r for r in events if r.get("event") == "model"), {})
    warmup = int(config.get("excluded_warmup_steps", 5))
    steps = [r for r in events if r.get("event") == "step"]
    measured = [r for r in steps if int(r.get("step", 0)) > warmup]
    step_secs = [number(r.get("sec")) for r in measured if number(r.get("sec")) > 0]
    avg_sec = mean(step_secs) if step_secs else None
    min_sec = min(step_secs) if step_secs else None
    max_sec = max(step_secs) if step_secs else None
    effective_batch = int(config.get("effective_batch_size", 32))
    samples_sec = effective_batch / avg_sec if avg_sec else None

    allocator_peak = max(
        (number(r.get("peak_allocated_mib")) for r in steps), default=0.0
    )
    allocator_reserved = max(
        (number(r.get("peak_reserved_mib")) for r in steps), default=0.0
    )

    gpu_mem = [number(r.get("memory.used.MiB")) for r in gpu]
    gpu_util = [number(r.get("utilization.gpu")) for r in gpu]
    active_util = [u for u in gpu_util if u > 0]
    nvidia_peak = max(gpu_mem, default=0.0)
    avg_util = mean(active_util) if active_util else None

    trainer_logs = [
        r for r in events if r.get("event") == "trainer_log" and "num_tokens" in r
    ]
    token_rate = None
    if len(trainer_logs) >= 2 and step_secs:
        first = next(
            (r for r in trainer_logs if int(r.get("step", 0)) > warmup), None
        )
        last = trainer_logs[-1]
        if first and int(last.get("step", 0)) > int(first.get("step", 0)):
            delta_tokens = number(last.get("num_tokens")) - number(first.get("num_tokens"))
            interval_steps = int(last["step"]) - int(first["step"])
            relevant = [
                number(r["sec"])
                for r in measured
                if int(first["step"]) < int(r["step"]) <= int(last["step"])
            ]
            denominator = sum(relevant)
            if delta_tokens > 0 and denominator > 0 and interval_steps > 0:
                token_rate = delta_tokens / denominator

    final_log = trainer_logs[-1] if trainer_logs else {}
    completed_steps = max((int(r.get("step", 0)) for r in steps), default=0)
    oom = "out of memory" in log.lower()
    stable = cli.exit_code == 0 and completed_steps >= int(config.get("max_steps", 25))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    text = f"""# Single-GPU SFT Baseline — 작업 과정 및 결과

작성일: {now}  
작업 디렉토리: `/workspace/fdsp_summary_format`  
로그: `single_outputs/train.log`  
GPU telemetry: `single_outputs/nvidia_smi.csv`

---

## 1. 목적

FSDP SFT와 동일한 모델·정밀도·LoRA·데이터·sequence length·optimizer·LR·유효 배치를 사용하되,
**FSDP 없이 물리 GPU0 한 장만** 사용해 25 step 단기 benchmark를 수행한다.

- 총 25 optimizer steps
- 최초 {warmup} step은 warm-up으로 제외
- 이후 {max(0, completed_steps - warmup)} step으로 평균 throughput 계산
- 전체 3 epoch 성능 비교가 아니라 **메모리/속도/안정성 baseline**

---

## 2. 동등 조건

| 항목 | FSDP SFT | Single-GPU baseline |
|------|----------|---------------------|
| 모델 | `google/gemma-3-12b-it` | 동일 |
| precision | BF16 / TF32 | 동일 |
| LoRA | r=16, alpha=16 | 동일 |
| 데이터 | `model_input_fixed.jsonl` | 동일 |
| max sequence length | 4096 | 동일 |
| activation checkpointing | FSDP activation ckpt=True | Trainer gradient ckpt=True (단일 GPU 대응) |
| optimizer | adamw_torch_fused | 동일 |
| learning rate | 2e-4, cosine | 동일 |
| per-device batch | 1 | 1 |
| gradient accumulation | 16 | 32 |
| world size | 2 | 1 |
| **effective batch** | 1×16×2 = **32** | 1×32×1 = **32** |
| FSDP | FULL_SHARD | **사용 안 함** |
| GPU | 2×A100-80GB | **GPU0 1×A100-80GB** |

activation checkpointing은 구현 경로가 다르다. FSDP는 layer wrapper 기반이고,
single-GPU는 Transformers gradient checkpointing으로 같은 재계산 trade-off를 적용했다.

---

## 3. 결과

| 필수 기록 | 값 |
|-----------|-----|
| exit code | **{cli.exit_code}** |
| 완료 step | **{completed_steps}/{config.get('max_steps', 25)}** |
| wall-clock | **{cli.elapsed_sec}s** |
| OOM | **{"YES" if oom else "NO"}** |
| 안정 완료 | **{"YES" if stable else "NO"}** |
| Peak VRAM (`nvidia-smi`) | **{nvidia_peak:.0f} MiB / 81920 MiB** |
| Peak allocated (PyTorch) | **{allocator_peak:.0f} MiB** |
| Peak reserved (PyTorch) | **{allocator_reserved:.0f} MiB** |
| 평균 sec/step (warm-up 제외) | **{fmt(avg_sec)} s** |
| min / max sec/step | **{fmt(min_sec)} / {fmt(max_sec)} s** |
| samples/sec (effective samples) | **{fmt(samples_sec, 3)}** |
| tokens/sec (Trainer num_tokens 기준) | **{fmt(token_rate, 1)}** |
| GPU utilization (active samples 평균) | **{fmt(avg_util, 1)}%** |
| 마지막 loss | **{final_log.get('loss', 'N/A')}** |
| 마지막 token accuracy | **{final_log.get('mean_token_accuracy', 'N/A')}** |

모델:
- total params: {model.get('total_params', 'N/A')}
- trainable params: {model.get('trainable_params', 'N/A')} ({fmt(number(model.get('trainable_pct')))}%)

---

## 4. 해석

### VRAM

- Single GPU에는 base parameter 전체가 상주한다.
- FSDP는 base/grad/optimizer state를 두 GPU에 shard하므로 GPU당 상주량을 줄이는 대신 통신 비용이 생긴다.
- LoRA라 optimizer state 자체는 작지만, activation과 BF16 base는 여전히 크다.

### Throughput

- `sec/step`은 유효 배치 32의 optimizer step 기준이다.
- `samples/sec = 32 / 평균 sec/step`.
- `tokens/sec`는 Trainer 누적 `num_tokens` 차이를 동일 구간 wall time으로 나눈 값이다.

### 안정성

- OOM: {"발생" if oom else "없음"}
- benchmark 완료: {"성공" if stable else "미완료"}
- 20-step 측정 구간에서 sec/step 변동 폭: {fmt(min_sec)}–{fmt(max_sec)}s

---

## 5. FSDP SFT와 비교

FSDP SFT 실측:
- Peak VRAM: 약 34GB/GPU
- 평균 step: 약 29–30s
- samples/sec: 1.09
- tokens/sec: 약 1555

Single-GPU 실측:
- Peak VRAM: {nvidia_peak:.0f} MiB
- 평균 step: {fmt(avg_sec)}s
- samples/sec: {fmt(samples_sec, 3)}
- tokens/sec: {fmt(token_rate, 1)}

주의: FSDP 값은 전체 3 epoch, single 값은 warm-up 제외 단기 benchmark다.
동일 유효 배치를 맞췄지만 장기 scheduler 상태와 데이터 순서 구간은 완전히 같지 않다.
"""

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(text, encoding="utf-8")
    print(f"wrote {cli.out}")


if __name__ == "__main__":
    main()
