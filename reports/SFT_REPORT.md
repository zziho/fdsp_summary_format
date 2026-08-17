# FSDP SFT 작업 과정 및 결과

작성일: 2026-08-16 20:10 UTC  
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
| FSDP | version=2, strategy=FULL_SHARD |
| wrap | ['Gemma3DecoderLayer'] |
| activation_checkpointing | True (FSDP config) |
| Trainer gradient_checkpointing | False (FSDP activation ckpt와 중복 방지) |
| LoRA | rank/alpha 16, language tower만 |
| 유효 배치 | per_device=1 × accum=16 × world=2 ≈ 32 |
| max_length | 4096 |
| epochs / steps | 3 / 108 |

P2P 채널 로그 수(대략): 0  
IB: 사용 시도

---

## 3. 시간 / Throughput

| 항목 | 값 |
|------|-----|
| start | (로그에서 추출 예정) |
| end | 2026-08-16T20:10:09Z |
| wall-clock (ELAPSED_SEC) | 3150 |
| train_runtime | 3113.0 s |
| step time (mean / last) | 28.9545 / 28.83 s |
| samples/sec | 1.09 |
| steps/sec | 0.035 |
| tokens/sec (rough) | 1449.4057179569547 |
| train_loss | 0.6712 |
| exit code | 0 |

이전 40GB 호스트(P2P disable) 대비: 이 런은 **NVLink P2P ON + A100 80GB** 이라 step time이 더 짧은 편(~28–31s vs ~75–83s).

---

## 4. 평가 5축

### 4.1 GPU별 VRAM

학습 중 스냅샷:

- GPU0 ( NVIDIA A100-SXM4-80GB):  32126/ 81920 MiB, util= 99%
- GPU1 ( NVIDIA A100-SXM4-80GB):  32266/ 81920 MiB, util= 64%

종료 근처:

- GPU0 ( NVIDIA A100-SXM4-80GB):  34028/ 81920 MiB, util= 49%
- GPU1 ( NVIDIA A100-SXM4-80GB):  34046/ 81920 MiB, util= 99%

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

| # | loss | grad_norm | lr | entropy | mean_token_acc | num_tokens | epoch |
|---|------|-----------|----|---------|----------------|------------|-------|
| 1 | 1.68 | 0.7082 | 0.0001996 | 0.9318 | 0.6822 | 4.554e+05 | 0.2827 |
| 2 | 1.076 | 0.237 | 0.0001921 | 1.081 | 0.7321 | 9.109e+05 | 0.5654 |
| 3 | 0.818 | 0.2477 | 0.0001759 | 0.8193 | 0.7821 | 1.376e+06 | 0.8481 |
| 4 | 0.7156 | 0.2235 | 0.0001526 | 0.7069 | 0.81 | 1.796e+06 | 1.113 |
| 5 | 0.5695 | 0.2204 | 0.0001244 | 0.545 | 0.8372 | 2.248e+06 | 1.396 |
| 6 | 0.5181 | 0.2518 | 9.384e-05 | 0.4983 | 0.8515 | 2.706e+06 | 1.678 |
| 7 | 0.4705 | 0.2731 | 6.388e-05 | 0.465 | 0.8641 | 3.174e+06 | 1.961 |
| 8 | 0.3825 | 0.2422 | 3.731e-05 | 0.4019 | 0.886 | 3.588e+06 | 2.226 |
| 9 | 0.3686 | 0.2515 | 1.664e-05 | 0.3924 | 0.889 | 4.05e+06 | 2.509 |
| 10 | 0.3691 | 0.3188 | 3.817e-06 | 0.3773 | 0.8891 | 4.512e+06 | 2.792 |

- OOM: (런 종료 코드/로그로 확인)
- NCCL hang: P2P ON 상태에서 진행됨 (이전 NODE-only 호스트와 다름)

on_train_begin inspect:
- model=FSDPPeftModelForCausalLM
- trainable=65470464/13259674224 (0.4938%)
- fsdp_marked_modules=51

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
- 로그: `outputs/logs/sft_train.log`
- 메트릭 JSONL: `outputs/sft_metrics.jsonl`
- provenance: `outputs/provenance/sft_*`

---

## 7. 다음 단계

1. `merge_adapter.py --stage sft` → `outputs/sft_merged`
2. FSDP GRPO on merged model
3. GRPO merge → `outputs/grpo_merged`

### Merge 결과
- merged path: `outputs/sft_merged`
- merge wall-clock: 60s
