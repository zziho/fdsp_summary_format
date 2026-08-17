# Single-GPU SFT Baseline — 작업 과정 및 결과

작성일: 2026-08-17 13:32 UTC  
작업 디렉토리: `/workspace/fdsp_summary_format`  
로그: `single_outputs/train.log`  
GPU telemetry: `single_outputs/nvidia_smi.csv`

---

## 1. 목적

FSDP SFT와 동일한 모델·정밀도·LoRA·데이터·sequence length·optimizer·LR·유효 배치를 사용하되,
**FSDP 없이 물리 GPU0 한 장만** 사용해 25 step 단기 benchmark를 수행한다.

- 총 25 optimizer steps
- 최초 5 step은 warm-up으로 제외
- 이후 20 step으로 평균 throughput 계산
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
| exit code | **0** |
| 완료 step | **25/25** |
| wall-clock | **1176s** |
| OOM | **NO** |
| 안정 완료 | **YES** |
| Peak VRAM (`nvidia-smi`) | **33160 MiB / 81920 MiB** |
| Peak allocated (PyTorch) | **31820 MiB** |
| Peak reserved (PyTorch) | **32636 MiB** |
| 평균 sec/step (warm-up 제외) | **45.89 s** |
| min / max sec/step | **40.89 / 51.42 s** |
| samples/sec (effective samples) | **0.697** |
| tokens/sec (Trainer num_tokens 기준) | **997.0** |
| GPU utilization (active samples 평균) | **96.5%** |
| 마지막 loss | **0.9176717400550842** |
| 마지막 token accuracy | **0.7578540463000536** |

모델:
- total params: 12252795504
- trainable params: 65470464 (0.53%)

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

- OOM: 없음
- benchmark 완료: 성공
- 20-step 측정 구간에서 sec/step 변동 폭: 40.89–51.42s

---

## 5. FSDP SFT와 비교

FSDP SFT 실측:
- Peak VRAM: 약 34GB/GPU
- 평균 step: 약 29–30s
- samples/sec: 1.09
- tokens/sec: 약 1555

Single-GPU 실측:
- Peak VRAM: 33160 MiB
- 평균 step: 45.89s
- samples/sec: 0.697
- tokens/sec: 997.0

주의: FSDP 값은 전체 3 epoch, single 값은 warm-up 제외 단기 benchmark다.
동일 유효 배치를 맞췄지만 장기 scheduler 상태와 데이터 순서 구간은 완전히 같지 않다.

### 비교 해석

| 지표 | 2-GPU FSDP | Single GPU | 관찰 |
|------|------------|------------|------|
| 평균 sec/step | ~29–30s | 45.89s | FSDP가 약 **1.5× 빠름** |
| samples/sec | 1.09 | 0.697 | FSDP가 약 **1.56× 높음** |
| tokens/sec | ~1555 | 997 | FSDP가 약 **1.56× 높음** |
| GPU당 peak VRAM | ~34GB | 33.16GB | 이번 LoRA 설정에서는 GPU당 절감이 관찰되지 않음 |
| 총 GPU VRAM 점유 | 약 68GB (2장 합) | 약 33GB | FSDP는 총량보다 분산/확장 경로에 초점 |

GPU당 VRAM이 비슷한 이유를 과대해석하면 안 된다.

1. LoRA는 optimizer/gradient가 전체 모델이 아니라 약 0.5%의 adapter에만 생겨,
   FSDP가 shard해서 줄일 optimizer/gradient state 자체가 작다.
2. activation은 FSDP로 shard되지 않으며 양쪽 GPU에 존재한다.
3. FSDP 경로는 tied embedding/lm_head의 cross-group alias를 피하려고 동일 값의
   output head를 별도 storage로 복제했다. 그래서 FSDP 측 파라미터 표시는 약
   **13.26B**, single 측은 원래 tied 상태인 약 **12.25B**다.
4. FSDP all-gather 버퍼와 wrapper overhead도 있다.

따라서 이번 결과의 정확한 결론은:

> 12B BF16 LoRA SFT는 A100-80GB 한 장에서도 약 33GB로 안정 실행됐다.
> 2-GPU FSDP는 GPU당 VRAM을 더 낮추지는 못했지만, 동일 유효 배치에서
> throughput을 약 1.56× 높였다. 즉 이번 LoRA 조건에서는 FSDP의 주된 실측
> 이득이 메모리 절감보다 처리량이었고, 2× 선형 가속에는 미치지 못했다.
