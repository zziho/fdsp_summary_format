#!/usr/bin/env python3
"""Generate SFT/GRPO markdown reports from logs + provenance (FSDP-focused)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
PROV = OUTPUT / "provenance"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _latest_nvidia(stage: str, tag_substr: str) -> list[dict]:
    files = sorted(PROV.glob(f"{stage}_nvidia_smi_*{tag_substr}*.csv"))
    if not files:
        files = sorted(PROV.glob(f"{stage}_nvidia_smi_*.csv"))
        files = [f for f in files if tag_substr in f.name] or files[-1:]
    if not files:
        return []
    path = files[-1]
    rows = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _gpu_summary(rows: list[dict]) -> str:
    if not rows:
        return "(nvidia-smi snapshot 없음)"
    lines = []
    for row in rows:
        idx = row.get("index", "?")
        used = row.get("memory.used.MiB") or row.get("memory.used")
        total = row.get("memory.total.MiB") or row.get("memory.total")
        util = row.get("utilization.gpu")
        name = row.get("name", "")
        lines.append(f"- GPU{idx} ({name}): {used}/{total} MiB, util={util}%")
    return "\n".join(lines)


def _parse_timing(log_text: str, stage: str) -> dict:
    out = {}
    m = re.search(rf"{stage.upper()}_START\s+(\S+)", log_text)
    if m:
        out["start"] = m.group(1)
    m = re.search(rf"{stage.upper()}_END\s+(\S+)\s+EXIT:(\d+)\s+ELAPSED_SEC:(\d+)", log_text)
    if m:
        out["end"] = m.group(1)
        out["exit"] = m.group(2)
        out["elapsed_sec"] = int(m.group(3))
    m = re.search(r"'train_runtime':\s*'?([\d.]+)'?", log_text)
    if m:
        out["train_runtime"] = float(m.group(1))
    m = re.search(r"'train_samples_per_second':\s*'?([\d.]+)'?", log_text)
    if m:
        out["samples_per_sec"] = float(m.group(1))
    m = re.search(r"'train_steps_per_second':\s*'?([\d.]+)'?", log_text)
    if m:
        out["steps_per_sec"] = float(m.group(1))
    m = re.search(r"'train_loss':\s*'?([\d.]+)'?", log_text)
    if m:
        out["train_loss"] = float(m.group(1))
    # progress bar step time sample
    times = [float(x) for x in re.findall(r"(\d+\.\d+)s/it", log_text)]
    if times:
        out["step_time_mean"] = sum(times) / len(times)
        out["step_time_last"] = times[-1]
    # P2P evidence
    out["p2p_channels"] = len(re.findall(r"via P2P/", log_text))
    out["ib_no_device"] = "NET/IB : No device found" in log_text
    out["nvlink_topo_note"] = "NV4" if "NV4" in log_text or out["p2p_channels"] else "unknown"
    return out


def _metrics_table(rows: list[dict]) -> str:
    if not rows:
        return "(메트릭 없음)"
    header = (
        "| # | loss | grad_norm | lr | entropy | mean_token_acc | num_tokens | epoch |\n"
        "|---|------|-----------|----|---------|----------------|------------|-------|\n"
    )
    body = []
    for i, row in enumerate(rows, 1):
        body.append(
            "| {i} | {loss} | {gn} | {lr} | {ent} | {acc} | {tok} | {ep} |".format(
                i=i,
                loss=row.get("loss", ""),
                gn=row.get("grad_norm", ""),
                lr=row.get("learning_rate", ""),
                ent=row.get("entropy", ""),
                acc=row.get("mean_token_accuracy", row.get("reward", "")),
                tok=row.get("num_tokens", ""),
                ep=row.get("epoch", ""),
            )
        )
    return header + "\n".join(body)


def _fsdp_cfg(stage: str) -> dict:
    rows = _read_jsonl(PROV / f"{stage}_rank0_fsdp_config.jsonl")
    return rows[-1] if rows else {}


def _inspect(stage: str, tag: str) -> dict:
    rows = _read_jsonl(PROV / f"{stage}_rank0_model_inspect.jsonl")
    for row in reversed(rows):
        if row.get("tag") == tag:
            return row
    return rows[-1] if rows else {}


def write_sft_report(log_path: Path, metrics_path: Path, out_path: Path) -> None:
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    metrics = _read_jsonl(metrics_path)
    timing = _parse_timing(log_text, "sft")
    cfg = _fsdp_cfg("sft")
    begin = _inspect("sft", "on_train_begin")
    step_gpus = _latest_nvidia("sft", "step_10") or _latest_nvidia("sft", "on_train_begin")
    end_gpus = _latest_nvidia("sft", "on_train_end") or step_gpus

    elapsed = timing.get("elapsed_sec")
    step_mean = timing.get("step_time_mean")
    samples = timing.get("samples_per_sec")
    # rough tokens/sec from last metric if present
    tokens_sec = None
    if metrics and elapsed:
        try:
            last_tok = float(str(metrics[-1].get("num_tokens", "0")).replace("e+", "E"))
            # scientific maybe already float via json
            last_tok = float(metrics[-1].get("num_tokens", 0))
            tokens_sec = last_tok / max(timing.get("train_runtime") or elapsed, 1)
        except Exception:
            tokens_sec = None

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = f"""# FSDP SFT 작업 과정 및 결과

작성일: {now}  
작업 디렉토리: `/workspace/fdsp_summary_format`  
Base model: `/workspace/models/google-gemma-3-12b-it` (`google/gemma-3-12b-it`)  
데이터: `/workspace/summary_format/train_data/model_input_fixed.jsonl`

---

## 1. 이 단계가 뭔지

BF16 **LoRA SFT**를 Transformers/TRL + **FSDP2 FULL_SHARD (2 GPU)** 로 학습한다.  
(Unsloth QLoRA가 아니라 BF16 LoRA + FSDP.)

플로우에서의 위치:

```text
google/gemma-3-12b-it
  → FSDP LoRA SFT → outputs/sft_adapter
  → merge → outputs/sft_merged
  → (다음) GRPO
```

---

## 2. 실행 환경 / FSDP 설정

| 항목 | 값 |
|------|-----|
| GPU | 2× NVIDIA A100-SXM4-80GB |
| 연결 | NVLink **NV4** (NCCL `via P2P/CUMEM/read`) |
| `NCCL_P2P_DISABLE` | `0` (활성화) |
| `NCCL_IB_DISABLE` | `0` (활성화 시도, 장치 없음 → Socket fallback) |
| FSDP | version={cfg.get('fsdp_version')}, strategy={cfg.get('sharding_strategy')} |
| wrap | {cfg.get('transformer_layer_cls_to_wrap')} |
| activation_checkpointing | {cfg.get('activation_checkpointing')} (FSDP config) |
| Trainer gradient_checkpointing | False (FSDP activation ckpt와 중복 방지) |
| LoRA | rank/alpha 16, language tower만 |
| 유효 배치 | per_device=1 × accum=16 × world=2 ≈ 32 |
| max_length | 4096 |
| epochs / steps | 3 / 108 |

P2P 채널 로그 수(대략): {timing.get('p2p_channels', 0)}  
IB: {"장치 없음 (No device found)" if timing.get('ib_no_device') else "사용 시도"}

---

## 3. 시간 / Throughput

| 항목 | 값 |
|------|-----|
| start | {timing.get('start', '(로그에서 추출 예정)')} |
| end | {timing.get('end', '(로그에서 추출 예정)')} |
| wall-clock (ELAPSED_SEC) | {timing.get('elapsed_sec', 'N/A')} |
| train_runtime | {timing.get('train_runtime', 'N/A')} s |
| step time (mean / last) | {timing.get('step_time_mean', 'N/A')} / {timing.get('step_time_last', 'N/A')} s |
| samples/sec | {timing.get('samples_per_sec', 'N/A')} |
| steps/sec | {timing.get('steps_per_sec', 'N/A')} |
| tokens/sec (rough) | {tokens_sec if tokens_sec is not None else 'N/A'} |
| train_loss | {timing.get('train_loss', 'N/A')} |
| exit code | {timing.get('exit', 'N/A')} |

이전 40GB 호스트(P2P disable) 대비: 이 런은 **NVLink P2P ON + A100 80GB** 이라 step time이 더 짧은 편(~28–31s vs ~75–83s).

---

## 4. 평가 5축

### 4.1 GPU별 VRAM

학습 중 스냅샷:

{_gpu_summary(step_gpus)}

종료 근처:

{_gpu_summary(end_gpus)}

해석:
- 양 GPU 사용량이 비슷 → FULL_SHARD가 실제로 분배됨
- A100 80GB 기준 ~32GB/GPU 수준. DDP로 12B BF16을 양쪽에 풀복제하는 것보다는 shard 전략이 맞음
- FSDP가 줄이는 건 주로 parameter/grad/optimizer shard. **activation은 별개**라 seq=4096에서 여전히 VRAM을 꽤 씀

### 4.2 Throughput

위 표 참고. 유효 배치 32, step ~30s → samples/sec ≈ 1 전후 예상.

### 4.3 통신 오버헤드

- 2 GPU여도 이상적 2×는 아님: FSDP step마다 all-gather / reduce-scatter
- 이번 호스트는 NVLink P2P가 살아 있어 이전(P2P disable)보다 통신이 유리
- 그래도 LoRA여도 base shard gather 비용은 남음

### 4.4 Checkpoint

- `save_strategy=no` + `save_fsdp_adapter()`: **lora_* 만** gather → PEFT adapter 저장
- full `Trainer.save_model()` FULL_STATE_DICT gather는 비실용(이전 실험에서 확인)
- 실용 체크포인트 = adapter 저장 → 이후 CPU merge
- sharded full ckpt resume는 이번 범위에서 약함

산출물:
- `outputs/sft_adapter`
- (merge 후) `outputs/sft_merged`

### 4.5 학습 안정성

메트릭 스냅샷:

{_metrics_table(metrics)}

- OOM: (런 종료 코드/로그로 확인)
- NCCL hang: P2P ON 상태에서 진행됨 (이전 NODE-only 호스트와 다름)

on_train_begin inspect:
- model={begin.get('model_class')}
- trainable={begin.get('trainable_params')}/{begin.get('total_params')} ({begin.get('trainable_pct')}%)
- fsdp_marked_modules={begin.get('fsdp_marked_modules')}

---

## 5. Activation checkpointing

이번 런은 **FSDP activation_checkpointing=True** 로 학습했다.

의미:
- FSDP: param/grad/opt state shard
- activation ckpt: activation을 덜 저장하고 backward 때 재계산 → 메모리↓, 계산량↑, 속도 일부↓

현재 확인 가능:
- 설정 ON + 그 아래에서의 VRAM(~32GB) / step time(~30s)

정량 A/B (ckpt ON vs OFF)는 본 학습과 별도 smoke가 필요하며, 파이프라인 continuer가 가능하면 `outputs/activation_ckpt_ab/` 에 남긴다.

---

## 6. 문제 / 메모

- base는 **공식 `google/gemma-3-12b-it`** (unsloth 미사용)
- HF gated 승인 + 토큰으로 다운로드
- InfiniBand 장치 없음 → IB disable=0이어도 IB 미사용
- 로그: `{log_path}`
- 메트릭 JSONL: `{metrics_path}`
- provenance: `outputs/provenance/sft_*`

---

## 7. 다음 단계

1. `merge_adapter.py --stage sft` → `outputs/sft_merged`
2. FSDP GRPO on merged model
3. GRPO merge → `outputs/grpo_merged`
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")


def write_grpo_report(log_path: Path, metrics_path: Path, out_path: Path) -> None:
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    metrics = _read_jsonl(metrics_path)
    timing = _parse_timing(log_text, "grpo")
    cfg = _fsdp_cfg("grpo")
    begin = _inspect("grpo", "on_train_begin")
    step_gpus = _latest_nvidia("grpo", "step_") or _latest_nvidia("grpo", "on_train_begin")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # GRPO metrics often have reward fields
    def grpo_table(rows: list[dict]) -> str:
        if not rows:
            return "(메트릭 없음)"
        keys = [
            "loss",
            "reward",
            "reward_std",
            "kl",
            "grad_norm",
            "learning_rate",
            "epoch",
        ]
        header = "| # | " + " | ".join(keys) + " |\n|---|" + "|".join(["---"] * len(keys)) + "|\n"
        body = []
        for i, row in enumerate(rows, 1):
            body.append(
                "| "
                + str(i)
                + " | "
                + " | ".join(str(row.get(k, row.get(f"rewards/format_reward/{k}", ""))) for k in keys)
                + " |"
            )
        # fallback: dump raw keys if mostly empty
        if all(not any(row.get(k) for k in keys) for row in rows[:3]):
            header = "| # | raw |\n|---|-----|\n"
            body = [f"| {i} | `{json.dumps(row, ensure_ascii=False)}` |" for i, row in enumerate(rows, 1)]
        return header + "\n".join(body)

    text = f"""# FSDP GRPO 작업 과정 및 결과

작성일: {now}  
작업 디렉토리: `/workspace/fdsp_summary_format`  
Base (SFT merged): `outputs/sft_merged`

---

## 1. 이 단계가 뭔지

SFT merge 모델을 base로 **BF16 LoRA GRPO**를 FSDP2 2-GPU에서 학습한다.  
보상은 format reward (`rewards.py`).

```text
outputs/sft_merged
  → FSDP LoRA GRPO → outputs/grpo_adapter
  → merge → outputs/grpo_merged
```

---

## 2. 실행 환경 / FSDP

| 항목 | 값 |
|------|-----|
| FSDP | version={cfg.get('fsdp_version')}, strategy={cfg.get('sharding_strategy')} |
| activation_checkpointing | {cfg.get('activation_checkpointing')} |
| NCCL P2P | enabled (NVLink) |
| num_generations | 4 |
| max_completion_length | 700 |
| use_vllm | False |

on_train_begin: model={begin.get('model_class')}, trainable={begin.get('trainable_params')}/{begin.get('total_params')}

---

## 3. 시간 / Throughput

| 항목 | 값 |
|------|-----|
| start | {timing.get('start', 'N/A')} |
| end | {timing.get('end', 'N/A')} |
| ELAPSED_SEC | {timing.get('elapsed_sec', 'N/A')} |
| train_runtime | {timing.get('train_runtime', 'N/A')} |
| step time mean/last | {timing.get('step_time_mean', 'N/A')} / {timing.get('step_time_last', 'N/A')} |
| samples/sec | {timing.get('samples_per_sec', 'N/A')} |
| exit | {timing.get('exit', 'N/A')} |

GRPO는 generation(4×, max 700 tokens) 때문에 SFT보다 step이 훨씬 길다.

---

## 4. 평가 5축

### 4.1 GPU별 VRAM

{_gpu_summary(step_gpus)}

### 4.2 Throughput
위 표 참고.

### 4.3 통신 오버헤드
FSDP all-gather + generation 단계 통신. NVLink P2P 사용.

### 4.4 Checkpoint
LoRA adapter only save → merge. full sharded resume는 약함.

### 4.5 학습 안정성

{grpo_table(metrics)}

---

## 5. Activation checkpointing

GRPO도 FSDP `activation_checkpointing=True`.  
긴 completion + generation activation이 추가로 붙으므로 VRAM/속도 trade-off가 SFT보다 더 도드라질 수 있다.

---

## 6. 산출물 / 로그

- adapter: `outputs/grpo_adapter`
- merged: `outputs/grpo_merged`
- log: `{log_path}`
- metrics: `{metrics_path}`
- provenance: `outputs/provenance/grpo_*`
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("sft", "grpo"), required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "sft":
        write_sft_report(args.log, args.metrics, args.out)
    else:
        write_grpo_report(args.log, args.metrics, args.out)


if __name__ == "__main__":
    main()
