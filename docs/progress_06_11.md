# 작업 기록 — FlashMLA SM100 fp8 Sparse Prefill (2026-06-11)

> 이 문서는 2026-06-10 ~ 06-11에 걸쳐 진행한 FlashMLA fp8 sparse prefill 작업의
> **전체 과정과 그 이유**를 기록한다. 코드 자체보다 "왜 이렇게 했는가"에 무게를 둔다.

---

## 0. 왜 이 일을 시작했는가 (동기)

온라인에 **"B300의 CP(context-parallel) 환경에서 FlashMLA가 trtllm-gen보다 훨씬 느리다"**는
주장이 있었다. 이를 검증하려고 sglang의 MLA attention 커널 마이크로벤치마크를 작성해 두었다
(`galbi-joa/sglang` 브랜치 `claude/trusting-mccarthy-SPNA5`,
`benchmark/mla_cp_kernel_profile/bench_mla_kernels.py`).

> 주의: 여기서 "FlashMLA"는 sglang이 vendor한 `sgl_kernel.flash_mla`이지, 우리가 공부 중인
> deepseek-ai/FlashMLA 원본이 아니다. sgl-kernel은 `sgl-project/FlashMLA @ df022eb`를 핀하며,
> 이는 우리 로컬 베이스(deepseek main `9241ae3`)보다 구버전이다.

벤치의 핵심 결론은 다음과 같았다 (B300, fp8 배포 기준 trtllm 우세):

| 경로 | 승자 | 격차 |
|---|---|---|
| sparse decode (fp8) | trtllm | 2.4–3.5× |
| **sparse prefill (fp8)** | **trtllm** | **1.7–2.0× (vs flashmla bf16)** |
| dense decode (B300) | trtllm only | flashmla엔 SM100 커널 없음 |

그리고 trtllm이 이기는 **이유 #1**이 바로:

> "FlashMLA의 sparse prefill 커널(`flash_mla_sparse_fwd`)은 **bf16 전용**이다.
> SM100 prefill 소스(`phase1_k512.cu`, `phase1_k576.cu`)에 fp8 변형이 없고,
> 시그니처에 `is_fp8_kvcache`가 없다. trtllm은 prefill에서 native fp8을 지원해
> KV read 대역폭을 절반으로 줄인다."

**→ 이번 작업의 목표: 이 갭(FlashMLA sparse prefill의 fp8 부재)을 실제로 메우는 것.**

---

## 1. 사전 탐구 ① — 왜 커널이 "small topk"와 아닌 것으로 나뉘어 있는가

작업 전에 `csrc/sm100/prefill/sparse`의 구조를 이해해야 했다. 두 디렉토리로 갈려 있다:

- `fwd/` (regular): `B_TOPK=128`, 깊은 software pipeline, bf16 전용
- `fwd_for_small_topk/`: `B_TOPK=64`, persistent(CLC)/split-KV, decode+prefill 겸용, fp8 지원

**선택 지점**은 `csrc/api/sparse_fwd.h:225`의 한 줄: `topk <= 1280`이면 small_topk.

**왜 나눴는가 (성능 트레이드오프):**

1. **Pipeline startup/drain 오버헤드 (핵심).** regular 커널은 KV를 128 블록으로 스트리밍하며
   Copy↔MMA↔Scale&Exp를 겹친다. 이건 KV 블록이 많을 때(큰 topk)만 이득. topk가 작으면
   prologue/epilogue만 하다 끝나 throughput이 무너진다.
2. **블록 granularity.** regular는 `topk % 128 == 0`을 강제. small_topk는 64로 잘게 쪼개
   tail 낭비가 적다.
3. **GPU 점유율.** regular는 `grid = 2*s_q`만 띄워 query×KV 일감이 충분해야 SM을 채운다.
   small_topk는 persistent+CLC(prefill) / split-KV(decode)로 부족한 병렬성을 보충한다.
4. **decode 워크로드.** small_topk는 decode(본질적으로 작은 topk·memory-bound)의 본거지이기도 하다.

---

## 2. 사전 탐구 ② — fwd_for_small_topk는 원래 decode인가?

git 히스토리 추적으로 **사실임을 확인**했다:

- decode 커널 `flash_fwd_splitkv_mla_fp8_sparse`는 원래
  `csrc/sm100/decode/sparse_fp8/splitkv_mla.cu`(592줄, 전용 decode 디렉토리)에 있었다.
- 리팩토링 커밋 `082094b "Multiple updates and refactorings (#150)"`에서 그 파일을 **삭제**하고
  `fwd_for_small_topk/head128/phase1.cuh`(1107줄)로 옮기며 `FWD_MODE` 템플릿으로 일반화해
  **prefill까지 처리**하게 만들었다.
- 흔적: decode launcher는 원래 이름 `flash_fwd_splitkv_mla_fp8_sparse_kernel`을 그대로 유지하고,
  prefill launcher만 새 이름을 받았다. 둘 다 같은 `sparse_attn_fwd_kernel_devfunc`를 호출한다.
- config가 decode 중심(K_raw fp8 버퍼, e8m0 scale, split-KV 에필로그)이고 prefill은
  `IS_PREFILL` 분기로 우회한다.

**이 사실이 이번 구현의 결정적 발판이 되었다**: decode가 이미 fp8 KV 데이터 경로
(index 변환 → NoPE/RoPE TMA gather → in-smem dequant)를 갖고 있으니,
이걸 prefill 제어 흐름에 접목하면 fp8 prefill을 만들 수 있다.

---

## 3. 핵심 설계 결정 — "native fp8"이란 무엇인가

자연스러운 발상은 e4m3를 텐서코어에 직접 먹이는 것(`tcgen05.mma.kind::mxf8f6f4`). 캐시의
e8m0 per-64dim scale이 MX scale 포맷과도 유사하다. 그러나 **이 커널 형상에서는 이득이 없음을
정량적으로 확인하고 기각했다** (CUTLASS 고정 SHA의 MMA atom static_assert로 검증):

1. **기존 bf16 "dual gemm"이 이미 100% MAC 효율.** 각 CTA가 head-dim 절반(256)만 보유 →
   per-CTA TMA·smem 절반. 2-SM UMMA(M=128, N=128, K=256)가 두 half-dim partial을 lane 절반씩에
   적재하고 softmax가 합산. 낭비 0%.
2. **MX MMA 형상 제약이 이 트릭을 표현 못 함.** MX는 M=128(1SM)/M=256(2SM)만 허용 →
   어떤 구성이든 50% garbage(2× fp8 속도 × 절반 효율 = bf16 본전), 트래픽 2배, 또는 TMEM 초과.
3. **S·V는 native fp8 자체가 불가능.** V의 e8m0 scale이 reduction 축(토큰)별로 변하는데
   MX scale은 32개 연속 reduction 원소가 공유해야 한다.
4. **정확도.** dequant-to-bf16는 quant.py 레시피를 비트 단위 재현(e4m3→bf16 exact, 2의 거듭제곱
   e8m0 곱 exact). native-MX는 Q까지 e4m3 양자화해 오차 추가.

**→ 결론: SM100 형상에서 "native fp8"이 텐서코어에서 사주는 게 없다. 진짜 이득은 fp8 캐시의
2.8× 작은 풋프린트와 fused in-smem dequant다.** 그래서 그것을 구현했다:
**MMA는 bf16 유지, KV는 fp8로 HBM에 두고 커널 내부에서 TMA gather + smem dequant.**

(참고 자료: gau-nernst tcgen05 블로그로 TMA/mbarrier/tmem/2-SM MMA/CLC 메커니즘을 학습,
CUTLASS `mma_sm100_umma.hpp`로 `SM100_MMA_MXF8F6F4_*`의 M 제약을 직접 확인.)

---

## 4. 구현 ① — MODEL1(=V4) native fp8 prefill 커널

`SparseAttnFwdMode::PrefillFp8` — `fwd_for_small_topk/head128` 템플릿의 세 번째 모드.

**재사용 (decode 데이터 경로):** index 변환 warp(좌표·e8m0 scale·validity mask),
raw NoPE TMA gather, RoPE TMA gather, dequant warpgroup(e4m3×2→bf16×2 with e8m0 scale),
3 bf16 K버퍼 + 2 raw fp8 버퍼.

**재사용 (prefill 제어 흐름):** CLC persistent 스케줄링, Q 경로(UTCCP→TMEM dual-gemm),
MMA 발행 루프, softmax warpgroup, full-bf16 에필로그, max_logits/lse 출력.

**새로 작성:** fp8-prefill 전용 index 변환 분기(~50줄)와 모드 배선. fp8-prefill에선
warp10=CLC 유지, warp11이 NoPE TMA 전담(decode는 10+11 분담).

**가장 검증이 필요한 부분 — barrier 도착 카운트** (docs §2에 표로 정리):
`bar_clc_empty=838`, `bar_valid_coord_scales_empty = 128+(cta==1)+1+128` 등.

**제약:** d_qk=512, h_q=128, topk%64==0, MQA. MODEL1 캐시(584B/토큰).

---

## 5. 구현 ② — V3.2 지원 (별도 커널 없이 fused dequant)

작업 중 "V3.2(d=576, 656B, fp32 scale)는 미지원이면 실사용을 어떻게 하냐"는 질문이 나왔다.
PrefillFp8 커널은 MODEL1 전용(small_topk 템플릿이 D_Q=512 고정, V3.2 scale은 fp32)이라
직접 확장은 별도 프로젝트다. 그러나 **비용 모델로 새 커널이 불필요함을 보였다**
(s_q=10K, s_kv=100K, topk=2048):

| 비용 | 바이트 | 시간(@6TB/s) |
|---|---:|---:|
| 풀 전체 dequant (layer당 1회) | 181 MB | ~30 µs |
| attention KV read (bf16) | 23.6 GB | compute-bound (~4 ms MMA) |

dequant 패스는 attention 자체 KV 트래픽의 **0.77%**. topk=2048에서 bf16 `fwd` 커널은
compute-bound라 bf16-vs-fp8 read 차이는 대부분 MMA에 가려진다.

**→ `flash_mla_sparse_fwd`에 fused 경로 추가:** 656B V3.2 캐시 자동 감지 →
`dequantize_v32_kv_cache`(풀 전체 bf16 dequant, `dequant_workspace`로 레이어 간 재사용,
quant.py 레퍼런스와 **CPU에서 비트 단위 일치 실측**) → 기존 bf16 d=576 커널 실행.

**trtllm과의 잔여 격차**는 대역폭이 아니라 trtllm의 **fp8 텐서코어 레이트**(scale 없는 plain
e4m3 캐시 → plain `kind::f8f6f4` MMA 2× 가능, 정확도 체제가 다름) 때문. 이건 `fwd` 스트리밍
커널의 fp8-MMA 변형이라는 별도 작업.

---

## 6. V4 확인 — MODEL1이 곧 DeepSeek V4였다

"내 레포 모델이 V4일 수도 있다"는 지적을 따라 upstream sglang을 확인한 결과:

- DeepSeek-V4 (Flash 284B / Pro 1.6T) 실재. 전용 `attention_backend="dsv4"`,
  기본 `kv_cache_dtype=fp8_e4m3`, page_size 256.
- dsv4 테스트가 명시하는 캐시: **"packed FP8-nope/BF16-rope"** = FlashMLA의 **MODEL1 포맷**.

**→ 즉 우리가 만든 PrefillFp8은 "벤치용"이 아니라 V4의 fp8 prefill 커널 그 자체다.**
MODEL1 production topk는 512–1024로, small_topk 커널의 스위트스폿 안이다.

---

## 7. 벤치마크 통합 (sgl-kernel 리빌드 없이)

sglang `bench_mla_kernels.py`를 수정해 sglang-flashmla와 단독-flashmla를 나란히 출력하게 했다.
**standalone `flash_mla` 패키지만 pip install**하면 되고(모듈명 `flash_mla` vs
`sgl_kernel.flash_mla`라 공존, 리빌드 불필요; 없으면 새 칼럼만 n/a), 기존 코드는 무변경:

- `--mode sparse_prefill`에 칼럼 추가: `sa_bf16`(standalone bf16 — 코드 버전 효과 분리),
  `sa_fp8v32`(V3.2 fp8, dequant 포함 측정 — 벤치가 이미 만드는 656B 캐시 그대로 입력).
- 새 `--mode sparse_prefill_v4`: V4/MODEL1 형상(d=512). `sgl_bf16`(bf16 baseline, fp8 배포 시
  bf16 캐시 복사본 필요) vs `sa_fp8`(우리 native 커널) vs `trtllm`(kv_lora_rank=448 cubin
  없어 n/a 예상 — trtllm 쪽 공백 기록).
- `--topk` 옵션(V4 스윕용 512/1024).
- 벤치의 `quantize_k_cache_v4`는 quant.py와 **비트 단위 일치 검증**.

여기서 입장이 역전된다: **V4 형상에선 trtllm이 커널이 없고 FlashMLA만 native fp8을 갖는다.**

---

## 8. 산출물 정리

### deepseek-ai/FlashMLA 포크 (`galbi-joa/FlashMLA`, 브랜치 `fp8-sparse-prefill`)

| 커밋 | 내용 |
|---|---|
| `333f3ba` | Add native fp8 sparse prefill mode (PrefillFp8) for SM100 |
| `4df16a7` | Support V3.2 fp8 KV cache via fused pool dequantization |

변경 파일:
- `csrc/params.h` — `SparseAttnFwdMode::PrefillFp8` + 페이지드 fp8 캐시 필드
- `csrc/api/sparse_fwd.h` — kv dtype 디스패치 + `sparse_attn_prefill_fp8_interface`
- `csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/config.h` — `IS_FP8_KV`, 버퍼/worker 카운트, TmaParams 변형
- `csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.cuh` — fp8-prefill 분기
- `.../instantiations/phase1_prefill_fp8_k512.cu` *(신규)*
- `flash_mla/flash_mla_interface.py` — V3.2 fused dequant + `dequantize_v32_kv_cache`
- `setup.py`, `README.md`
- `docs/20260610-sm100-fp8-sparse-prefill.md` *(신규, 설계·비용·barrier 분석)*
- `tests/test_flash_mla_sparse_prefill_fp8.py` *(신규, MODEL1/V4)*
- `tests/test_flash_mla_sparse_prefill_fp8_v32.py` *(신규, V3.2)*

### galbi-joa/sglang (브랜치 `claude/trusting-mccarthy-SPNA5`)

| 커밋 | 내용 |
|---|---|
| `9d540b5` | bench: add standalone-FlashMLA fp8 prefill columns + V4/MODEL1 mode |

---

## 9. 검증 상태 (정직하게)

- **CPU로 실측 검증됨:** `dequantize_v32_kv_cache`와 벤치 `quantize_k_cache_v4`가
  FlashMLA `tests/quant.py` 레퍼런스와 비트 단위 일치.
- **정적으로만 검증됨 (이 Mac에 CUDA 없음):** native PrefillFp8 커널은 **컴파일·실행 미검증**.
  모드 분기·barrier 카운트는 전수 정적 점검했으나 하드웨어 확인 필요.

**B200/B300에서 할 일:**
1. 빌드: `FLASH_MLA_DISABLE_SM90=1 pip install -v --no-build-isolation .` (CUDA ≥ 12.9)
2. 정확성: `tests/test_flash_mla_sparse_prefill_fp8_v32.py`(V3.2),
   `tests/test_flash_mla_sparse_prefill_fp8.py`(V4 native — 행 걸리면 docs §2 barrier 카운트부터)
3. 회귀: 기존 bf16 prefill / decode 테스트 (템플릿 공유)
4. 벤치: `--mode sparse_prefill`, `--mode sparse_prefill_v4 --topk 1024`

**예상 결과:**
- V3.2(`sa_fp8v32`): `fmla_sparse_bf16` + 수십 µs 수준. trtllm 대비 열세는 줄지만 잔존
  (잔여 격차 = trtllm fp8 텐서코어 레이트).
- V4(`sparse_prefill_v4`): `sa_fp8`이 `sgl_bf16` 대비 KV read 절반(584B vs 1024B/토큰) 이득,
  trtllm은 n/a. **V4 시대엔 벤치 결론이 반대 방향이 될 수 있다.**

---

## 10. 한 줄 요약

> "B300 fp8 prefill에서 FlashMLA가 진다"는 관찰의 직접 원인(sparse prefill의 fp8 부재)을,
> SM100 텐서코어 형상 분석에 근거해 **MMA는 bf16 유지 + KV는 fp8로 두고 커널 내부 dequant**
> 방식으로 메웠다. MODEL1(=DeepSeek V4)은 native 커널, V3.2는 저비용 fused dequant 경로로
> 처리했고, 둘 다 포크에 푸시 + 벤치마크에 비교 칼럼으로 통합했다. 커널의 하드웨어 검증만 남았다.
