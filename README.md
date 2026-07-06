# SLICE
# VideoQA를 위한 파라미터-프리 등의미(Iso-Semantic) 키프레임 샘플링

긴 영상에서 VideoQA(비디오 질의응답)에 가장 유용한 프레임을 선택하는 **파라미터-프리(Parameter-Free)**
키프레임 샘플러입니다. 시간 축을 균일하게 나누거나(Uniform) 관련성 상위 K개를 그대로 취하는(Top-K)
대신, 물리적 시간 축을 **의미 축(Semantic Timeline)** 으로 왜곡(Warping)하여 선택된 모든 프레임이
**동일한 양의 쿼리 관련성 에너지**를 담도록 만듭니다 — **에너지 균등 분할(Energy-Based
Equipartition, EBE)**.

사용자가 지정하는 값은 **K**(선택할 프레임 수) 하나뿐이며, 온도·다양성·대역폭 등
(`κ`, `λ`, `α` …) **튜닝해야 할 하이퍼파라미터가 전혀 없습니다.**

---

## 왜 Iso-Semantic 인가?

프레임별 쿼리 관련성 점수 `s(t)`(예: BLIP / CLIP 이미지–텍스트 매칭 점수)가 주어졌을 때:

| 전략 | 실패 양상 |
|------|-----------|
| **Uniform(시간 균등)** | 관련 없는 긴 구간에 프레임을 낭비하고, 짧게 스치는 핵심 근거를 놓침 |
| **Top-K(점수 상위)** | 점수가 높은 한 구간에 몰려 시간적 커버리지와 다양성을 잃음 |
| **Iso-Semantic(제안 기법)** | 관련성 밀도에 **비례하여** 프레임을 배분 — 근거가 밀집된 구간엔 많이, 빈 구간엔 적게, 그러면서도 시간적으로 고르게 퍼짐 |

핵심 아이디어: **시간(x) 축이 아니라 에너지(y) 축을 K등분한다.**

---

## 방법 — 에너지 균등 분할(EBE)

구현 위치: [`frame_selector_parameter_free.py`](frame_selector_parameter_free.py) →
`ParameterFreeSampler.sample_iso_semantic(frame_indices, scores, k)`

**1. 스케일 적응형 평활화 (파라미터-프리 대역폭)**
원본 점수 곡선을 가우시안으로 평활화하되, 그 폭을 영상 길이에 맞춰 **자동으로** 결정합니다
(수동 대역폭 없음):
```
σ = log(n)          # n = 점수가 매겨진 프레임 수
smooth = gaussian_filter(scores, sigma=σ)
```

**2. 의미 밀도(PDF) & 누적 타임라인(CDF)**
평활화된 비음수 곡선을 시간에 대한 확률 밀도(*의미 밀도*)로 간주하고, 그 누적합을
*누적 관련성 에너지*로 삼습니다:
```
pdf = max(smooth, 0) / Σ max(smooth, 0)
cdf = cumsum(pdf)                      # 0 → 1 로 단조 증가
```

**3. 등의미 분할 (역함수 CDF 워핑)**
에너지 축을 K개의 동일 구간으로 나누고, **역함수(Inverse CDF)** 로 각 구간을 시간 축의
윈도우로 되돌립니다. 관련성 밀도가 높은 구간은 에너지 공간에서 "넓어져" 더 많은 윈도우를
할당받습니다:
```
energy_edges = linspace(0, 1, k + 1)
time_edges   = searchsorted(cdf, energy_edges)   # 에너지 → 시간 워핑
```

**4. 워핑 윈도우 내 지역 최대화**
각 에너지 균등 윈도우 안에서 **원본(un-smoothed)** 점수가 가장 높은 프레임을 선택합니다.
평활화는 *어디를 볼지*, 원본 점수는 *정확히 어떤 프레임*을 결정합니다:
```
각 윈도우 [start, end) 에 대해:
    argmax(raw_scores[start:end]) 선택
```

강건한 예외 처리가 내장되어 있습니다: `n ≤ k` 이면 전체 프레임 반환, 전체 에너지가 0이면
균일 샘플링으로 대체, 퇴화된 윈도우는 보정하여 항상 K개 슬롯을 모두 채웁니다.

---

## 저장소 구조

```
IsoSemantic_Keyframe_Sampler/
├── frame_selector_parameter_free.py   # 샘플러 본체 (제안 기법)
├── requirements.txt                   # 전체 연구 환경 (pip)
├── README.md                          # 본 문서
└── lmms-eval/                         # 평가 하네스 (VideoQA 정확도)
    └── lmms_eval/
        ├── tasks/                     # longvideobench, videomme, nextqa, egoschema, mlvu, ...
        └── models/                    # llava_onevision, qwen2_vl, llava_vid, ... (use_topk=True)
```

---

## 설치

```bash
# Python 3.10–3.12. VideoQA 평가 단계에는 CUDA GPU가 필요합니다.
pip install -r requirements.txt

# 평가 하네스 설치 (editable):
cd lmms-eval && pip install -e . && cd ..
```

---

## 데이터 준비 — 프레임별 관련성 점수

샘플러는 **사전 계산된** 프레임별 관련성 점수를 입력으로 사용합니다. 이미지–텍스트 매칭
모델(BLIP/CLIP)로 한 번 생성해 두며, 서로 정렬된 두 개의 JSON 파일을 만듭니다:

| 파일 | 내용 |
|------|------|
| `scores.json` | (QA 항목별) 프레임별 관련성 점수 리스트 |
| `frames.json` | (QA 항목별) 위 점수가 대응되는 프레임 인덱스 리스트 |

> 본 프로젝트에서는 AKS 스코어링 파이프라인
> (`blip_image_text_matching.py` / `feature_extract*.py`)으로 생성하며,
> `AKS/outscores/<dataset>/blip/{scores.json,frames.json}` 에 저장됩니다.

---

## 1단계 — 샘플러 실행

`frame_selector_parameter_free.py` 하단의 경로를 수정하거나
`process_lvb_dataset_parallel(...)` 를 직접 호출한 뒤 실행합니다:

```bash
python frame_selector_parameter_free.py
```

`process_lvb_dataset_parallel` 의 입력 / 출력:

| 인자 | 의미 |
|------|------|
| `input_json_path` | 벤치마크 QA JSON (예: `videomme.json`, `lvb_val.json`) |
| `blip_score_path` | `scores.json` (프레임별 관련성) |
| `frames_json_path`| `frames.json` (해당 점수의 프레임 인덱스) |
| `k` | **선택할 프레임 수 — 유일한 조정값** |
| `output_json_path`| 증강되어 저장되는 JSON (`include_frame_idx_*.json`) |
| `visualize` | `True` 이면 항목별 CDF / 분할 / 선택 시각화 저장 |

각 출력 항목에는 다음이 추가됩니다:
```json
{ "...": "...", "frame_idx": [12, 47, 91, ...], "frame_num": 32, "method": "iso_semantic_ebe" }
```

---

## 2단계 — lmms-eval 로 VideoQA 정확도 평가

1단계에서 생성한 JSON을 태스크의 `data_files` 로 지정합니다. 예를 들어
`lmms-eval/lmms_eval/tasks/longvideobench/longvideobench_val_v.yaml` 에서:

```yaml
dataset_kwargs:
  data_files: {"validation": include_frame_idx_isosemantic_k32.json}
```

그 다음 평가를 실행합니다. 모델은 `use_topk=True` 를 통해 각 항목의 `frame_idx` 리스트를
읽어 **정확히 그 프레임들만** 디코딩합니다:

```bash
cd lmms-eval
accelerate launch --num_processes 4 --main_process_port 12345 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,use_topk=True \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision_isosemantic \
    --output_path ./results/
```

### 프레임 인덱스가 소비되는 방식
`use_topk=True` 이면 VideoQA 모델은 기본 균일 샘플러를 우회하고 다음을 수행합니다:
```python
top_id    = doc['frame_idx']          # Iso-Semantic 샘플러가 생성
frame_idx = sorted(top_id[:max_frames_num])
spare_frames = vr.get_batch(frame_idx)   # 정확히 이 프레임들을 디코딩
```
따라서 실험 간 **유일한 변수는 프레임 선택 연산자**뿐이며, 동일한 VQA 모델 아래에서
샘플링 전략을 공정하게(apples-to-apples) 비교할 수 있습니다.

---

## 지원 벤치마크 / 모델

- **벤치마크:** LongVideoBench, Video-MME, NExT-QA, EgoSchema, MLVU, LVBench (`lmms_eval/tasks/` 참조).
- **모델:** `lmms_eval/models/` 의 `use_topk` 지원 모델 — 예: `llava_onevision`,
  `qwen2_vl`, `llava_vid`, `longva`, `llama_vid`.

---

## 라이선스

`lmms-eval/` 평가 하네스는 원본 라이선스를 따릅니다(`lmms-eval/LICENSE` 참조).
`frame_selector_parameter_free.py` 샘플러는 연구 목적으로 공개됩니다.
