# FSDP GRPO 작업 과정 및 결과

작성일: 2026-08-17 04:50 UTC (재시작 후 진행 중)  
작업 디렉토리: `/workspace/fdsp_summary_format`  
Base (SFT merged): `outputs/sft_merged`  
원본 base: `/workspace/models/google-gemma-3-12b-it` (`google/gemma-3-12b-it`)

---

## 1. 이 단계가 뭔지

SFT merge 모델을 base로 **BF16 LoRA GRPO**를 Transformers/TRL + **FSDP2 FULL_SHARD (2 GPU)** 로 학습한다.  
보상은 format reward (`rewards.py`).

```text
outputs/sft_merged
  → FSDP LoRA GRPO → outputs/grpo_adapter   ← 현재 (재시작 런)
  → merge → outputs/grpo_merged
```

실행: `run_grpo_only.sh` / tmux `fdsp_grpo`

---

## 2. 이전 런이 왜 죽었는지 (101/240 OOM)

첫 본 학습(`2026-08-16T21:51Z` 시작)은 **~6.7시간 / 101/240**에서 CUDA OOM으로 종료.

| 항목 | 값 |
|------|-----|
| 종료 | `2026-08-17T04:32:10Z` EXIT:1 ELAPSED≈24070s |
| 실패 위치 | `selective_log_softmax` / `log_softmax` (vocab logits) |
| GPU0 | 79.21 GiB / 79.44 GiB, free 226 MiB |

### 진짜 원인 (EOS)

Gemma chat은 턴을 **`<end_of_turn>` (id=106)** 으로 끝내고 `<eos>`(id=1)는 거의 안 씀.  
TRL GRPO는 `tokenizer.eos_token_id`(=1)만 stop으로 써서:

- completion이 거의 전부 **700토큰까지 생성** (`clipped_ratio ≈ 0.99`)
- logprob forward의 vocab×seq logits가 비정상적으로 커짐 → OOM
- `mask_truncated_completions=True`라 truncated 샘플은 학습 신호도 거의 없음

이전 메트릭 예: `completions/mean_length≈700`, `clipped_ratio≈1.0`.

체크포인트: `save_strategy=no` + 중간 adapter 저장 없음 → **가중치 이어받기 불가**. 재시작이 맞음.

실패한 로그/메트릭 보관:
- `outputs/logs/grpo_train_failed_*.log`
- `outputs/grpo_metrics_failed_*.jsonl`

---

## 3. 재시작 시 고친 것

1. **EOS override** → `tokenizer.eos_token = <end_of_turn>` + `generation_kwargs.eos_token_id=[106,1]`
2. **VRAM 여유** → `per_device=4`, `grad_accum=2`, `generation_batch=16` (prompts/step는 그대로 4)
3. **중간 adapter 저장** → 매 20 step `outputs/grpo_adapter_step{N}`

2-step smoke 결과:

| 항목 | 이전 본런 | smoke (수정 후) |
|------|-----------|-----------------|
| clipped_ratio | ~0.99–1.0 | **0 ~ 0.31** |
| mean_length | ~700 | **~286–482** |
| peak VRAM | ~79GB → OOM | **~43GB / ~36GB** |
| step time | ~169s | ~106–143s (짧은 completion일 때 더 빠름) |
| checkpoint | 없음 | step1/2 adapter 저장 확인 |

---

## 4. 현재 런 설정

### 4.1 하드웨어 · NCCL

| 항목 | 값 |
|------|-----|
| GPU | 2× A100-SXM4-80GB |
| 연결 | **NVLink NV4** |
| `NCCL_P2P_DISABLE` | **0** |
| `NCCL_IB_DISABLE` | 0 (IB 장치 없음) |

### 4.2 FSDP

| 항목 | 값 |
|------|-----|
| version / strategy | 2 / FULL_SHARD |
| wrap | Gemma3DecoderLayer |
| activation_checkpointing | True |
| Trainer gradient_checkpointing | False |
| trainable | 65,470,464 (0.4938% LoRA) |

### 4.3 GRPO 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| `GRPO_PER_DEVICE_BATCH` | **4** |
| `GRPO_GRAD_ACCUM` | **2** |
| `GRPO_GENERATION_BATCH` | **16** |
| `GRPO_NUM_GENERATIONS` | 4 |
| prompts/step | **4** `(16÷4)` |
| est_steps | **240** |
| `GRPO_EOS_TOKEN` | `<end_of_turn>` |
| `max_completion_length` | 700 |
| `GRPO_CHECKPOINT_EVERY` | 20 |
| lr / beta / optim | 2e-6 / 0.04 / adamw_torch_fused |
| use_vllm | False |

---

## 5. 시간 / Throughput (진행 중 · 2026-08-17 12:23 UTC 갱신)

| 항목 | 값 |
|------|-----|
| 재시작 start | **2026-08-17T04:49:52Z** |
| 진행 | **~108/240 (~45%)** |
| 정상 step time | **~150–160 s/it** |
| end / ELAPSED | *(완료 후)* |
| exit | *(완료 후)* |

### Eval @ 100 (및 예정 @ 200)

`max_steps=-1` 전체 학습이면 `eval_strategy="steps"`, `eval_steps=100` 이 켜진다.

- step **100**에서 GRPO **eval** 실행 → 내부 tqdm **84/84**
- eval 프롬프트 ≈168, prompts/batch≈2 → 84 batches × ~2분 ≈ **~3시간**
- step **200**에서도 동일 예상
- tqdm이 eval 시간을 step 101에 섞어 **60h+ ETA**가 뜬 적 있음 → **표시 ETA는 무시**, 최근 step 간격으로 추정

### 남은 시간 (실측, tqdm 무시)

| | |
|--|--|
| 남은 학습 | ~6시간 |
| eval @200 | ~3시간 |
| **지금부터 합계** | **약 8.5~9시간** |

로그: `outputs/logs/grpo_train.log` / `grpo_only_pipeline.log`  
모니터: `show_progress.py`

---

## 6. 평가 5축

### 6.1 GPU별 VRAM

- on_train_begin: ~15.7 / 16.6 GB  
- smoke peak: ~43 / 36 GB  
- 본 런 학습 중: ~50–56 GB / util 99% (양 GPU)

### 6.2 Throughput

- 학습 step ~150–160s, prompts/step=4  
- eval 100마다 ~3h 추가

### 6.3 통신

NVLink P2P ON + FSDP all-gather. IB 없음.

### 6.4 Checkpoint

- 확인됨: `grpo_adapter_step20/40/60/80/100`  
- 종료 시 `outputs/grpo_adapter` → merge → `outputs/grpo_merged`

### 6.5 학습 안정성 (중간)

| 지표 | 대략 |
|------|------|
| clipped_ratio | **0.05–0.08** (1차 런 0.99 대비 정상) |
| mean_length | **~280–320** |
| reward | **~4.0–5.3** |
| OOM / NCCL hang | 없음 |

주의: 일부 step에서 `loss`/`kl`이 비정상적으로 큼 (예: step100 근처).  
완료 후 jsonl로 재검토. 현재는 reward·length·clipped가 더 신뢰 가능.

---

## 7. Activation checkpointing

FSDP `activation_checkpointing=True` 유지.  
이전 OOM의 주원인은 act-ckpt 부족이 아니라 **EOS 미정지로 인한 700토큰 풀 길이 logits**였음.

---

## 8. 산출물 / 다음

| 산출물 | 경로 | 상태 |
|--------|------|------|
| GRPO adapter | `outputs/grpo_adapter` | 학습 종료 후 |
| mid ckpt | `outputs/grpo_adapter_step{20…100…}` | ✅ 진행 중 |
| merged | `outputs/grpo_merged` | merge 후 |
| 전체 요약 | `reports/FSDP_전체_작업_기록.md` | ✅ |

완료 후 timing/metrics/exit 반영 + merge 자동 진행.  
다음 런 권장: `eval_strategy="no"` (또는 간격 확대).
