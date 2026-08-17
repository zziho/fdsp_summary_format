# FSDP 학습 전체 기록 (SFT → GRPO)

작성/갱신: 2026-08-17 12:23 UTC (GRPO 진행 중, ~108/240)  
작업 디렉토리: `/workspace/fdsp_summary_format`  
상세 리포트:
- `reports/SFT_작업_과정_및_결과.md` (SFT 완료)
- `reports/GRPO_작업_과정_및_결과.md` (GRPO 진행 중)

---

## 0. 한 줄 요약

**공식 `google/gemma-3-12b-it`** 를 2×A100-80GB에서 **FSDP2 FULL_SHARD + BF16 LoRA** 로  
SFT → merge → GRPO → (예정) merge 파이프라인을 돌리는 실험.

- SFT: **완료** (108 step, ~52분, merge 60초)
- GRPO: **재시작 런 진행 중** (~108/240, EOS 수정 후)
- NVLink **NV4** + `NCCL_P2P_DISABLE=0` (IB 장치 없음)

---

## 1. 왜 FSDP인가 / 다른 방법과 뭐가 다른지

| 방식 | 이 실험에서의 의미 |
|------|-------------------|
| Unsloth QLoRA (단일 GPU) | 빠르지만 FSDP multi-GPU 경로와 다름. **이번엔 사용 안 함** |
| DDP | 12B BF16을 GPU마다 풀복제 → 메모리 부담 큼 |
| **FSDP2 FULL_SHARD** | param/grad/opt를 2 GPU에 shard. 필요 시 all-gather |
| ZeRO | DeepSpeed 경로. 이번엔 HF/TRL FSDP 플러그인 사용 |

핵심 메시지: FSDP는 “2배 속도” 마법이 아니라 **메모리 shard로 BF16 multi-GPU를 가능하게 하는 전략**.  
통신(all-gather/reduce-scatter) 때문에 이상적 2× throughput은 안 나옴.

---

## 2. 공통 환경 (SFT·GRPO)

| 항목 | 값 |
|------|-----|
| GPU | 2× NVIDIA A100-SXM4-**80GB** |
| 토폴로지 | GPU0↔GPU1 = **NVLink NV4** |
| NCCL P2P | **ON** (`NCCL_P2P_DISABLE=0`) |
| NCCL IB | 시도 ON, **장치 없음** → 미사용 |
| FSDP | version **2**, `FULL_SHARD`, `reshard_after_forward=True` |
| wrap | `Gemma3DecoderLayer` |
| activation_checkpointing | **True** (FSDP) |
| Trainer gradient_checkpointing | **False** (중복 방지) |
| LoRA | rank/alpha **16**, language tower |
| trainable | ~65.5M / ~13.3B (**0.49%**) |
| base | `/workspace/models/google-gemma-3-12b-it` (gated, HF 토큰) |
| 데이터 | `/workspace/summary_format/train_data/model_input_fixed.jsonl` |

Provenance: `outputs/provenance/{sft,grpo}_*`

---

## 3. SFT (완료)

### 결과

| 항목 | 값 |
|------|-----|
| start→end | ~19:17–20:10 UTC (2026-08-16) |
| wall / train_runtime | **3150s / 3113s** |
| steps | **108** (3 epoch) |
| step time | **~29–31 s** |
| samples/sec | **1.09** |
| train_loss | **0.6712** |
| token acc (최종 근처) | **~0.89** |
| VRAM (학습 중) | 양 GPU **~32–34 GB / 80GB** (대칭 → shard 동작) |
| exit | **0** |

### Checkpoint / merge

- `save_strategy=no` + `save_fsdp_adapter()` → LoRA만 저장
- full FULL_STATE_DICT gather는 비실용 (이전 실험 교훈)
- `outputs/sft_adapter` → merge **60s** → `outputs/sft_merged` (**23GB**)

### 메트릭 추이 (발췌)

| epoch 근처 | loss | mean_token_acc |
|------------|------|----------------|
| 0.28 | 1.68 | 0.68 |
| 1.11 | 0.72 | 0.81 |
| 2.79 | 0.37 | 0.89 |

loss 하락 / acc 상승 / OOM·NCCL hang 없음.

### Activation checkpointing

ON 상태에서 VRAM ~34GB, step ~30s.  
(ON vs OFF 정량 A/B는 본 파이프라인에서 미실시.)

---

## 4. GRPO

### 4.1 실패한 1차 런 (기록용)

| 항목 | 값 |
|------|-----|
| 기간 | 2026-08-16 21:51 → 08-17 04:32 UTC |
| 진행 | **101/240**에서 사망 |
| 원인 | CUDA OOM @ `selective_log_softmax` |
| 근본 | EOS 불일치: Gemma는 `<end_of_turn>(106)`, TRL은 `<eos>(1)`만 stop |
| 증상 | `mean_length≈700`, `clipped_ratio≈0.99` |
| ckpt | 없음 → **이어받기 불가** |
| 보관 | `outputs/logs/grpo_train_failed_*.log`, `grpo_metrics_failed_*.jsonl` |

부가 이슈: `rewards.py` JSON 하위항목 `int+str` TypeError → **수정** (bool 캐스팅 + try/except).

### 4.2 재시작 런 설정 (현재)

| 항목 | 값 |
|------|-----|
| start | **2026-08-17T04:49:52Z** |
| tmux | `fdsp_grpo` |
| EOS | `<end_of_turn>` + generation `eos_token_id=[106,1]` |
| per_device / grad_accum | **4 / 2** |
| generation_batch / num_generations | **16 / 4** |
| prompts/step / est_steps | **4 / 240** |
| max_completion | 700 |
| checkpoint every | **20** → `outputs/grpo_adapter_step{N}` |
| NVLink / P2P | ON |

Smoke (2 step): clipped 0~0.31, mean_len ~286–482, peak VRAM ~43/36GB, OOM 없음.

### 4.3 진행 스냅샷 (2026-08-17 12:23 UTC)

| 항목 | 값 |
|------|-----|
| progress | **~108/240 (~45%)** |
| 정상 step | **~150–160 s/it** |
| clipped_ratio | **~0.05–0.08** (EOS 수정 효과 유지) |
| mean_length | **~280–320** |
| reward (로그 샘플) | **~4.0–5.3** |
| mid-ckpt | step **20,40,60,80,100** 저장됨 |
| 상태 | 프로세스 alive, GPU util ~99% |

### 4.4 Eval이 시간을 잡아먹음 (중요)

`train_grpo.py` 설정:

```text
eval_strategy = "steps"   # max_steps=-1(전체 학습)일 때
eval_steps = 100
```

- **step 100**에서 eval 실행 → 내부 bar **84/84** (eval ~168 prompts ÷ 2 prompts/batch)
- 한 eval ≈ **~3시간** (생성 기반이라 step과 비슷하게 김)
- **step 200**에서 한 번 더 예정

tqdm ETA가 60h+로 튀는 이유: eval 3시간이 step 101 시간에 섞여 **평균 s/it가 왜곡**됨.  
실제 학습 step은 다시 ~2.5분. **표시 ETA ≠ 실제 남은 시간**.

### 4.5 남은 시간 추정 (실측 기준, tqdm 무시)

| 구간 | 예상 |
|------|------|
| 남은 학습 step (~132) | ~6시간 |
| eval @ 200 | ~3시간 |
| **합계 (지금부터)** | **약 8.5~9시간** |
| 이후 | adapter 저장 + merge (~1분대) + 리포트 갱신 |

### 4.6 GRPO 메트릭 주의

로깅 간격 10 step. 일부 step에서 `loss`/`kl`이 비정상적으로 큼  
(예: step 100 근처 `loss~8e4`, `kl~1.8e6`) — 스케일/마스킹/수치 이슈 가능.  
**완료 후** `outputs/grpo_metrics.jsonl` + reward 추이로 재해석 필요.  
현재로선 **reward·mean_length·clipped_ratio**가 안정성 판단에 더 신뢰 가능.

---

## 5. 평가 5축 (이번 FSDP 실험에서 본 것)

### 5.1 GPU별 VRAM

| 단계 | 대략 |
|------|------|
| SFT 학습 | ~32–34 GB / GPU (대칭) |
| GRPO (EOS 수정 후) | 평소 ~50–56 GB, peak provenance에 더 높게도 기록됨 |
| GRPO (EOS 깨진 1차) | ~79 GB → OOM |

FSDP가 줄이는 건 주로 **param/grad/opt**. activation·generate 버퍼는 별개.

### 5.2 Throughput / tokens/sec (NVLink P2P ON)

| 단계 | step time | samples/sec | **tokens/sec** | 계산 |
|------|-----------|-------------|----------------|------|
| **SFT** | ~30 s | **1.09** | **≈1555** | `num_tokens 4.841e6 / train_runtime 3113s` |
| **GRPO** (재시작 런) | ~150–160 s | (step당 prompt 4) | **≈140–150** | Trainer `num_tokens` 증가분 ÷ (10×step_time) 평균 ≈**144** |
| GRPO completion만 (참고) | 同 | — | **≈30** | `16 comps/step × ~300 tok / ~155s` (생성 토큰만) |
| GRPO+eval | eval당 ~3 h | — | — | 100/200 step |

참고:
- SFT tokens/sec는 **학습에 들어간 토큰 총량 / wall time** (가장 표준적인 지표).
- GRPO `num_tokens`는 TRL이 누적하는 값이라 prompt+completion 등이 포함될 수 있음. completion-only ≈30 tok/s는 참고용.
- NVLink P2P ON 환경에서의 실측. P2P OFF A/B는 이번 런에서 안 함.

### 5.3 통신 오버헤드

- NVLink P2P ON → PCIe-only보다 유리
- 그래도 FSDP gather 때문에 2GPU≠2×
- IB 없음

### 5.4 Checkpoint

| | 방식 | 결과 |
|--|------|------|
| SFT/GRPO | LoRA adapter gather | 실용, 성공 |
| full state dict | FULL gather | 비실용 |
| GRPO mid | 매 20 step | step20…100 확인 |
| resume | sharded full | 이번 범위 약함 |

### 5.5 학습 안정성

- SFT: loss↓ acc↑, 안정 완료
- GRPO 1차: OOM으로 중단
- GRPO 2차: OOM 없음, EOS 정상, clipped↓ / reward 유지 중
- NCCL hang: P2P ON 환경에서 재발 없음

---

## 6. Activation checkpointing

- FSDP `activation_checkpointing=True` 전 구간 ON
- SFT: ~34GB에서 학습 가능
- GRPO OOM의 주원인은 act-ckpt 부족이 아니라 **EOS로 인한 700토큰 풀 길이**
- ON/OFF A/B 정량은 미실시 (필요 시 별도 smoke)

---

## 7. 발생한 문제와 해결 (타임라인)

1. **HF gated** `google/gemma-3-12b-it` → 라이선스 승인 + 토큰
2. **unsloth 경로 금지** → 공식 google 가중치만 사용
3. **NCCL P2P/NVLink** → disable=0, topo NV4 확인
4. **보상함수 TypeError** → `rewards.py` bool/guard 수정
5. **GRPO OOM @101** → EOS `<end_of_turn>` + 배치 축소 + mid-ckpt
6. **에이전트 disconnect** → tmux `fdsp_grpo`로 생존
7. **progress 모니터 공백** → completion 표가 tail을 덮음 → `show_progress.py`
8. **eval 3시간 / 가짜 60h ETA** → `eval_steps=100` + tqdm 왜곡 (문서화). 다음 런에선 eval off 권장

---

## 8. 산출물 경로

| 산출물 | 경로 | 상태 |
|--------|------|------|
| SFT adapter | `outputs/sft_adapter` | ✅ |
| SFT merged | `outputs/sft_merged` | ✅ |
| SFT metrics | `outputs/sft_metrics.jsonl` | ✅ |
| SFT report | `reports/SFT_작업_과정_및_결과.md` | ✅ |
| GRPO mid ckpt | `outputs/grpo_adapter_step{20…100}` | ✅ 진행 중 |
| GRPO final adapter | `outputs/grpo_adapter` | ⏳ 종료 후 |
| GRPO merged | `outputs/grpo_merged` | ⏳ merge 후 |
| GRPO log | `outputs/logs/grpo_train.log` / `grpo_only_pipeline.log` | ✅ |
| GRPO report | `reports/GRPO_작업_과정_및_결과.md` | ⏳ 완료 후 최종 갱신 |
| Provenance | `outputs/provenance/` | ✅ |
| Progress helper | `show_progress.py` | ✅ |

---

## 9. 다음에 하면 좋은 것

1. GRPO 완료 → merge → 최종 metrics/ELAPSED를 GRPO 리포트에 기입
2. `eval_strategy="no"` 또는 eval 간격을 크게 (시간 절약)
3. `show_progress.py` ETA를 “최근 N step 간격”으로 (eval spike 무시)
4. (선택) activation ckpt ON/OFF / P2P ON/OFF 짧은 A/B
5. step 100 근처 비정상 `loss`/`kl` 원인 점검

---

## 10. 현재 한 줄 상태

**SFT 완료 · GRPO 재시작 런 ~45% · NVLink/P2P ON · EOS 수정 유지 · mid-ckpt 있음 · 남은 시간 실측 ~9h (학습~6h + eval@200 ~3h).**
