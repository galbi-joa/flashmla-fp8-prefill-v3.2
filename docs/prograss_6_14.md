# 2026-06-14 FlashMLA fp8 sparse prefill 진행 기록

이 문서는 2026-06-14 기준으로 지금까지 구현한 내용, 현재 코드 상태, 남은 문제, 다음에 수정하기 전에 반드시 다시 봐야 하는 문서를 정리한 기록이다. 파일명은 요청한 그대로 `prograss_6_14.md`로 둔다.

## 0. 현재 한 줄 결론

현재 production으로 실제 연결된 경로는 native Blackwell MX fp8 sparse prefill이 아니다.

지금 동작하는 실용 경로는 V3.2 fp8 KV cache를 bf16 workspace로 dequantize한 뒤, 기존 bf16 sparse prefill 커널을 재사용하는 방식이다. 반면 `fwd_v32_native_fp8_prefill.cu`는 SM100 native MX fp8를 붙여 보기 위한 research shell이며, setup.py/dispatch에 등록된 production 커널은 아니다.

따라서 현재 상태를 다음처럼 구분해야 한다.

- production-wired: V3.2 fp8 cache -> bf16 dequant workspace -> 기존 bf16 sparse prefill
- research shell 1: V3.2 cache를 커널 내부에서 읽어 bf16 경로로 넘기는 실험 파일
- research shell 2: V3.2 cache를 Blackwell MX fp8 tile로 바꿔 tcgen05.mma를 쓰려는 native fp8 실험 파일
- 아직 안 된 것: 실제 production dispatch에서 native MX fp8 tcgen05 sparse prefill을 실행하는 것

## 1. 이번 작업의 배경

처음 문제는 B300에서 sparse prefill/extend 성능이 TensorRT-LLM fp8 경로보다 느리다는 점이었다. 사용자가 준 기준 성능은 대략 다음과 같았다.

| B | NEW | CACHED | KV_TOT | H | trtllm_fp8 | fmla_sparse_bf16 | fmla_kv_fp8 | sa_bf16 | sa_fp8v32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 875 | 69125 | 70000 | 64 | 0.231 | 0.397 | 0.757 | 0.391 | 0.749 |
| 2 | 875 | 69125 | 70000 | 64 | 0.457 | 0.790 | 1.596 | 0.858 | 1.252 |
| 4 | 875 | 69125 | 70000 | 64 | 0.922 | 1.807 | 3.048 | 1.744 | 2.100 |
| 8 | 875 | 69125 | 70000 | 64 | 2.135 | 3.630 | 5.999 | 3.641 | 3.875 |

여기서 `sa_fp8v32`는 native fp8 Tensor Core 경로가 아니라, V3.2 fp8 cache를 bf16으로 dequantize한 뒤 기존 bf16 sparse attention을 탄 결과로 해석해야 한다.

## 2. 구현된 production-wired 경로

실제로 Python API에서 연결된 구현은 `flash_mla/flash_mla_interface.py`에 있다.

핵심 판단은 KV cache의 마지막 차원이 656 byte인지 확인하는 것이다. V3.2 cache는 token 하나가 656 byte로 저장된다. 이 모양을 보면 sparse prefill 호출 전에 fp8 cache를 bf16 workspace로 풀어낸다.

개념적으로는 다음 구조다.

```python
# V3.2 cache는 token당 656 byte인 fp8 압축 cache다.
is_v32_cache = kv_cache.dim() == 4 and kv_cache.shape[-1] == 656

if is_v32_cache:
    # fp8 V3.2 cache를 bf16 workspace로 dequantize한다.
    dequantize_v32_kv_cache(kv_cache, out=dequant_workspace)

    # 기존 sparse prefill 커널은 bf16 K/V cache 모양을 기대한다.
    kv_cache_for_prefill = dequant_workspace.view(num_tokens, 1, 576)

    # 이후는 기존 bf16 sparse prefill 경로를 그대로 탄다.
    sparse_prefill_fwd(q, kv_cache_for_prefill, ...)
```

이 방식의 장점은 기존 bf16 prefill 커널을 거의 그대로 재사용할 수 있다는 것이다. QK, softmax, SV, sink 처리, topk 처리, 기존 테스트 구조를 크게 흔들지 않는다.

단점은 분명하다. fp8 cache를 읽긴 하지만 Tensor Core가 native fp8 MMA를 하는 것은 아니다. 계산 자체는 여전히 bf16 UMMA 경로에 가깝고, 앞단에 dequant 비용과 bf16 workspace traffic이 추가된다.

## 3. 테스트와 benchmark 쪽 변경

`tests/test_flash_mla_sparse_prefill_fp8_v32.py`에서는 V3.2 경로의 시간을 더 분리해서 볼 수 있게 했다.

추가/정리한 측정 항목은 다음이다.

- `V3.2 dequant only`: V3.2 fp8 cache를 bf16 workspace로 푸는 시간
- `V3.2 fused GPU e2e`: dequant와 sparse prefill을 합친 end-to-end GPU 시간
- `V3.2 attn kernel only`: 이미 dequant된 bf16 workspace를 쓰는 attention kernel 시간

이 구분이 중요한 이유는 TensorRT-LLM이나 기존 FlashMLA KV fp8 결과와 비교할 때 `attn kernel only`만 보면 안 되기 때문이다. 실제 사용자 경로의 비용은 dequant를 포함한 `fused GPU e2e`에 더 가깝다.

B300 표에 맞춰 다음 케이스도 추가했다.

```python
# B300 표와 맞춰 보기 위한 대표 케이스
TestParam(
    seqlen_q=875,
    seqlen_k=70000,
    topk=2048,
    h_q=64,
    have_attn_sink=True,
    reuse_workspace=True,
)
```

## 4. 문서화한 내용

`docs/20260610-sm100-fp8-sparse-prefill.md`에는 SM100/B300 sparse fp8 prefill 판단을 업데이트했다.

기록한 핵심 내용은 다음이다.

- V3.2 practical path는 fp8 cache -> bf16 workspace -> bf16 sparse prefill이다.
- 이 경로는 native MX fp8가 아니므로, 성능 비교 시 dequant 시간을 반드시 포함해야 한다.
- Hopper fp8 decode kernel의 pipeline 아이디어는 참고할 수 있지만, SM100 prefill에 그대로 옮기면 안 된다.
- Blackwell/SM100은 tcgen05.mma, TMA, TMEM, 1SM/2SM MMA, PDL, cache hint 같은 도구를 제공하지만, 현재 FlashMLA V3.2 cache format과 native MX fp8 MMA의 scale 구조가 잘 맞지 않는다.
- native MX fp8 prefill은 연구 대상이지, 지금 바로 production에 붙일 수 있는 상태가 아니다.

`writeup.md`에는 한국어로 문제의 흐름과 해결 방향을 짧게 정리했다.

`howto_cn.md`에는 중국어로 구현 메모와 작업 요약을 남겼다.

## 5. research shell 1: fwd_v32_fp8_prefill.cu

`csrc/sm100/prefill/sparse/fwd_v32_fp8_prefill.cu`는 V3.2 cache를 대상으로 한 fp8 sparse prefill 실험 파일이다.

이 파일의 목적은 production Python 경로처럼 전역 workspace에 먼저 dequantize하는 방식 말고, 커널 쪽에서 V3.2 cache를 더 직접 읽어 sparse prefill에 가까운 형태로 가져갈 수 있는지 확인하는 것이다.

다만 현재 이 파일도 setup.py에 등록된 production kernel로 봐서는 안 된다. 실제 사용 경로에 들어간 안정 커널이라기보다, SM100 sparse prefill 전용으로 어떤 producer/consumer 구조를 만들 수 있는지 보는 연구 파일이다.

## 6. research shell 2: fwd_v32_native_fp8_prefill.cu

`csrc/sm100/prefill/sparse/fwd_v32_native_fp8_prefill.cu`는 더 공격적인 연구 파일이다.

목표는 다음과 같다.

1. V3.2 fp8 cache를 읽는다.
2. Blackwell MX fp8 MMA가 먹을 수 있는 tile layout으로 transcode한다.
3. QK와 S@V를 bf16 UMMA가 아니라 tcgen05 native MX fp8 MMA로 실행할 수 있는지 검토한다.

파일 안에서 잡은 주요 상수는 다음 의미를 가진다.

```cpp
// V3.2의 논리 head 구성
static constexpr int D_QK = 576;    // Q/K head dim = NoPE 512 + RoPE 64
static constexpr int D_V = 512;     // V dim
static constexpr int D_NOPE = 512;  // non-positional part
static constexpr int D_ROPE = 64;   // rotary positional part

// V3.2 cache token 하나의 물리 크기
static constexpr int V32_BYTES_PER_TOKEN = 656;

// Blackwell MX fp8 MMA scale group
static constexpr int MX_GROUP_K = 32;

// 576은 MX tile에 바로 예쁘게 맞지 않기 때문에 640으로 padding해서 검토
static constexpr int D_QK_PAD = 640;
```

여기서 중요한 점은 `D_QK=576`과 `D_V=512`가 임의로 만든 값이 아니라는 것이다. V3.2 cache에서 Q/K는 NoPE 512 + RoPE 64로 계산되고, V는 512 dim이다. token당 656 byte는 이 데이터를 V3.2 fp8 포맷으로 저장한 물리 크기다.

이 research 파일에서 추가로 만진 부분은 다음이다.

- V3.2 token이 한 block 안에서 split될 수 있는 상황을 다루기 위해 `maybe_split_token` 계열 helper를 둔다.
- Q transcode는 row/head-major로 잡았다. 한 lane이 한 head row를 맡고 K group을 따라간다.
- K transcode는 token-major로 잡았다. 한 lane이 topk token 하나를 맡고 20개 MX K-group을 따라간다.
- K NoPE는 기본적으로 V3.2의 dim128 scale을 네 개의 dim32 MX scale로 복제한다.
- 이론적인 per32 requant 경로는 `FLASHMLA_ENABLE_THEORETICAL_MXF8_V32_NOPE_REQUANT`로 남겨 두었다. 기본 경로로 보면 너무 비싸다.
- V compact transcode는 한 lane이 value dim 하나를 맡고 32-token group을 register에 들고 처리하는 쪽으로 정리했다.
- V scale load는 warp lane 0이 읽고 `__shfl_sync`로 broadcast하도록 정리했다. 같은 token scale을 512개 value dim마다 반복해서 읽는 낭비를 줄이려는 목적이다.
- scale stage를 lane 0이 UTCCP copy하기 전에 모든 lane의 write가 끝나도록 named barrier를 추가했다.

이 파일은 아직 production에 등록되어 있지 않다. 현재 의미는 "가능성을 따지는 연구 코드"다.

## 7. native MX fp8가 바로 빨라지지 않는 이유

처음에는 Blackwell이 fp8 Tensor Core를 제공하므로, fp8 cache를 native fp8 MMA로 먹이면 바로 빨라질 것처럼 보였다.

하지만 현재 V3.2 cache format과 SM100 MX fp8 MMA의 요구가 잘 맞지 않는다.

QK에서 reduction dimension은 head dim이다. 현재 Q/K는 576 dim인데, MX tile에 맞추려면 640으로 padding하는 구성이 된다. heads=64 기준으로 보면 실제 유효 계산은 `64 * 128 * 576`인데, research shell에서 고려한 MX QK tile은 `128 * 128 * 640` 쪽으로 간다.

대략적인 비율은 다음이다.

```text
QK actual/useful work
= 128 * 128 * 640 / (64 * 128 * 576)
= 약 2.22x

fp8 Tensor Core가 bf16보다 2x 빠르다고 가정해도
2.22 / 2 = 약 1.11x
```

즉 QK는 Tensor Core peak만 보면 좋아 보여도 padding과 tile shape 때문에 이론 하한부터 이미 bf16보다 뚜렷하게 유리하지 않다.

S@V도 비슷하다. S@V에서 reduction dimension은 topk token 축이다. 그런데 V3.2의 scale은 token마다, dim group마다 붙어 있다. 반면 MX fp8 MMA는 GEMM-K 방향의 32개 또는 64개 chunk가 scale을 공유하는 구조를 기대한다.

S@V를 MX MMA에 맞추려면 V를 다시 `token_group32 x value_dim` 형태로 transcode해야 한다. B=1만 봐도 필요한 V transcode 양은 다음 정도다.

```text
875 * 2048 * 512 = 약 917.5M values
```

반면 전체 cache를 한 번 bf16으로 푸는 현재 V3.2 dequant는 다음 정도다.

```text
70000 * 512 = 약 35.84M values
```

B=1에서 native V transcode 후보가 이미 pool dequant보다 약 25.6배 큰 일을 한다. B=8이면 이 비율은 약 204.8배까지 커진다. 이 구조라면 native MX fp8가 Tensor Core에서 빨라져도, 앞단 transcode가 이득을 먹어 버릴 가능성이 높다.

## 8. Hopper fp8 decode kernel에서 가져올 수 있는 것과 없는 것

사용자가 제공한 Hopper fp8 dequant kernel 설명에서 참고할 만한 아이디어는 있었다.

가져올 수 있는 것:

- load/dequant/compute를 서로 겹치려는 pipeline 사고방식
- cache hint를 사용해 fp8 cache read의 cache 오염을 줄이는 방향
- TMA 또는 copy pipeline을 더 잘게 쪼개는 방식
- producer/consumer 역할을 분리하는 사고방식
- PDL이나 persistent scheduling을 통해 tail과 launch overhead를 줄이는 방향

그대로 가져오면 안 되는 것:

- Hopper decode는 decode 중심이고, 현재 문제는 sparse prefill이다.
- decode는 새 token이 작고 KV cache가 이미 누적된 상황을 다루지만, prefill/extend는 NEW=875 같은 큰 query batch가 들어온다.
- Hopper fp8 decode의 dequant 구조를 그대로 SM100 native MX fp8 prefill에 붙이면, scale layout과 tile layout 문제가 해결되지 않는다.
- V3.2 cache format을 유지한 채 native MX S@V를 하려면 topk별 V transcode 비용이 매우 커진다.

따라서 Hopper 아이디어는 "pipeline 설계 감각"으로 참고하는 것이 맞고, 코드 구조를 그대로 복사하는 것은 맞지 않다.

## 9. cache format 변경 문제

현재 연구에서 가장 큰 결론은 cache format이 병목이라는 점이다.

V3.2 cache는 token 중심으로 저장되어 있고, token마다 scale이 붙는다. 이 구조는 KV cache append와 decode에는 자연스럽다. 하지만 SM100 MX fp8 MMA는 GEMM-K chunk 기준 scale을 기대한다.

QK는 그나마 head dim이 reduction 축이라 dim group scale과 맞춰 볼 여지가 있다. 하지만 S@V는 reduction 축이 token/topk 축이므로, V scale을 token_group32 기준으로 다시 맞춰야 한다.

즉 MX-friendly V cache를 만들려면 다음 중 하나가 필요하다.

- V를 `value_dim x token_group32` 또는 이와 비슷한 MMA-friendly layout으로 따로 저장한다.
- token별 V3.2 scale을 유지하지 않고, 더 coarse한 fp8 scale 정책을 쓴다.
- append 시점 또는 prefill 시작 전에 MX tile용 prepack cache를 별도로 만든다.

이건 파일 하나 추가로 끝나는 일이 아니다. production cache format을 바꾸면 최소한 다음 영역을 같이 봐야 한다.

- Python API의 cache shape 판별과 dispatch
- V3.2 quant/dequant test helper
- sparse prefill kernel entry
- sparse decode API
- sm100 decode head64 config/kernel
- sm90 compatibility 경로
- setup.py 커널 등록
- benchmark/test/documentation

따라서 현재는 native MX fp8를 production에 바로 붙이기보다, research file에서 cache format 가정을 바꾼 microbenchmark를 먼저 만드는 쪽이 맞다.

## 10. 앞으로 해야 할 일

우선순위는 다음 순서가 좋다.

1. 현재 production V3.2 bf16-dequant 경로의 e2e 측정을 더 정확히 한다.
   - dequant only, attn only, fused e2e를 같은 stream/event 기준으로 재확인한다.
   - B=1/2/4/8/16/32까지 B300 표와 같은 조건으로 다시 맞춘다.
   - 비교 대상은 `attn only`가 아니라 `fused GPU e2e`로 둔다.

2. `fwd_v32_native_fp8_prefill.cu`를 production에 붙이기 전에 standalone bring-up부터 한다.
   - setup.py 등록 전에 별도 compile gate 또는 isolated target으로 문법/링크를 확인한다.
   - single-buffer launch path가 의도대로 동작하는지 먼저 본다.
   - 현재 grid.y = n_chunk 구조는 QK/S를 D_V chunk마다 다시 계산할 수 있으므로 성능 비교용으로 쓰면 안 된다.

3. native MX path의 microbenchmark를 만든다.
   - V3.2 원본 cache를 읽는 버전
   - 미리 MX-friendly로 prepacked된 V cache를 읽는 버전
   - 두 버전을 분리해서 transcode cost와 MMA cost를 따로 본다.

4. cache format 변경의 범위를 작게 실험한다.
   - production cache format을 바로 바꾸지 않는다.
   - 먼저 research-only prepacked cache tensor를 하나 추가한다.
   - 그 tensor를 쓰는 native MX S@V microkernel을 만든다.
   - 이 결과가 `sa_fp8v32`보다 충분히 좋아야 production 변경을 검토한다.

5. Blackwell-specific optimization은 native path의 의미가 확인된 뒤 적용한다.
   - fine-grained TMA-GEMM pipeline
   - cache hint
   - PDL
   - persistent/tile scheduler
   - 1SM/2SM MMA 선택

현재 단계에서 pipeline을 예쁘게 만드는 것보다 더 중요한 질문은 "이 cache format에서 native MX path가 이론적으로 이길 수 있는가"이다.

## 11. 다음에 수정하기 전에 반드시 다시 볼 문서

수정 전에 최소한 아래 문서는 다시 봐야 한다.

- `docs/progress_06_11.md`
  - 초기에 폴더 목적, sparse prefill 작업 맥락, 기존 진행 상태를 파악한 문서다.

- `docs/20260610-sm100-fp8-sparse-prefill.md`
  - 현재 이슈의 중심 문서다. V3.2 practical path, native MX path의 제약, 측정 기준, B300 gate가 정리되어 있다.

- `writeup.md`
  - 한국어로 문제 해결 흐름을 정리한 문서다. 논리 흐름을 다시 잡을 때 보기 좋다.

- `howto_cn.md`
  - 중국어 작업 보고/설명용 문서다. 구현한 내용을 중국어로 설명해야 할 때 기준으로 쓰면 된다.

- `Blackwell SM100 GEMMs.md`
  - Blackwell SM100 GEMM, tcgen05, TMA/TMEM, MMA tile 제약을 판단할 때 참고한 문서다.

- `tcgen05.md`
  - tcgen05.mma, MX fp8 scale layout, instruction-level 제약을 다시 확인할 때 봐야 한다.

- `csrc/sm100/prefill/sparse/fwd_v32_fp8_prefill.cu`
  - V3.2 fp8 cache를 sparse prefill 쪽에서 직접 다루려는 research shell이다.

- `csrc/sm100/prefill/sparse/fwd_v32_native_fp8_prefill.cu`
  - native MX fp8 research의 중심 파일이다. production에 붙이기 전에 이 파일의 assumptions를 먼저 정리해야 한다.

- `tests/test_flash_mla_sparse_prefill_fp8_v32.py`
  - 성능 숫자를 해석할 때 benchmark label을 반드시 확인해야 한다. `attn only`와 `fused GPU e2e`를 섞으면 결론이 틀어진다.

## 12. 커밋할 때 주의할 점

이번 커밋에는 실제 의미 있는 코드/문서만 포함해야 한다.

포함할 가치가 있는 파일:

- `flash_mla/flash_mla_interface.py`
- `tests/test_flash_mla_sparse_prefill_fp8_v32.py`
- `docs/20260610-sm100-fp8-sparse-prefill.md`
- `docs/mla_explained.py`
- `csrc/sm100/prefill/sparse/fwd_v32_fp8_prefill.cu`
- `csrc/sm100/prefill/sparse/fwd_v32_native_fp8_prefill.cu`
- `Blackwell SM100 GEMMs.md`
- `tcgen05.md`
- `how.md`
- `writeup.md`
- `howto_cn.md`
- `docs/progress_06_11.md`
- `docs/prograss_6_14.md`

주의할 파일:

- `.DS_Store` 계열 파일은 커밋하지 않는다.
- reference 문서인 `Blackwell SM100 GEMMs.md`, `tcgen05.md`는 분석 재현성을 위해 이번 커밋에 포함한다. 단, 나중에 정리할 때는 원문 덤프 대신 핵심 링크와 요약 문서로 줄이는 편이 낫다.
- `how.md`, `docs/mla_explained.py`는 이번 대화에서 나온 구현/MLA 계산 설명 자료라 함께 포함한다.

## 13. 현재 판단

지금까지의 구현은 "V3.2 fp8 cache를 받아서 기존 bf16 prefill로 이어지게 만든 실용 경로"와 "Blackwell native MX fp8 sparse prefill 가능성을 검토하는 research shell"이 같이 존재하는 상태다.

성능을 당장 올리는 가장 현실적인 길은 먼저 현재 V3.2 dequant+bf16 경로의 overhead를 정확히 재고, 그 다음 native MX가 이론적으로 이길 수 있는 cache format을 별도 prepack 실험으로 검증하는 것이다.

현재 V3.2 token-centric cache를 그대로 둔 채 native MX fp8 tcgen05.mma만 붙이는 접근은, QK padding과 S@V transcode 비용 때문에 Tensor Core peak만큼의 이득을 기대하기 어렵다.
