/*
Standalone SM100 V3.2 fp8 sparse-prefill kernel shell.

Purpose
-------
The current Python V3.2 path does:

    [656B fp8 KV cache in HBM] -> dequantize whole pool to bf16 workspace
    -> reuse the existing bf16 sparse prefill.

That is simple, but it writes and rereads a 576-bf16-token workspace from HBM.
For the B300 sparse-prefill case we care about, the experiment should instead be:

    [656B fp8 KV cache in HBM] -> gather selected topk tokens
    -> dequantize inside the prefill kernel into bf16 shared memory
    -> reuse the existing bf16 UMMA QK and S@V pipeline.

This file keeps that idea visible as a separate V3.2-only kernel shell.  It is
not registered in setup.py yet.

B300 shape audit
----------------
The user-provided B300 result table explicitly says heads=64 and topk=2048.
The d_qk=576, d_v=512 assumptions come from the V3.2 cache format and the
existing SM100 regular-prefill d=576 path, not from that table text itself.
This workload dispatches to csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh,
not the head128 regular prefill path and not the MODEL1 small-topk fp8 path.

That matters for the pipeline:

  head128 regular prefill:
    K producer and V producer are separate, with separate K-part and V-part
    readiness barriers.

  head64 regular prefill:
    one NoPE shared-memory tile is reused as K-NoPE for QK and as V for S@V.
    RoPE is K-only.  Therefore a B300 V3.2 kernel should not add a separate
    VProducer that dequantizes the same NoPE bytes again.  It should replace
    the head64 KV-NoPE TMA producer with an fp8->bf16 NoPE producer, keep the
    RoPE copy path separate, and preserve:

      bar_kv_nope_ready[cur_buf][0/1]
      bar_kv_rope_ready
      bar_qk_nope_done[cur_buf]
      bar_qk_rope_done
      bar_sv_done[cur_buf]

The head128 skeleton below is still useful as a contrast case, but it is not
table-equivalent for the B300 heads=64 numbers.

Scope note
----------
This is not a rewrite of the existing MODEL1 small-topk PrefillFp8 kernel in
fwd_for_small_topk/head128.  That kernel uses B_TOPK=64 and a combined
bar_KV_full/bar_KV_empty contract because the same half-tile shared-memory
buffer is interpreted as K for QK and as V for S@V.

The original contrast shell in this file targets the V3.2 d=576 regular
sparse-prefill pipeline in fwd/head128, where K and V have separate logical
layouts and separate part-ready barriers.  The B300-relevant target is the
head64 contract below.  The Hopper/H800 decode kernel is only a scheduling
reference here: keep producer work overlapped with Tensor-Core work, but do not
copy its seesaw math.

For the measured B300 heads=64 shape, the real implementation target is the
head64 producer contract described above.

Blackwell GEMM notes
--------------------
Blackwell SM100 GEMMs.md changes the implementation priority:

  - Existing bf16 UMMA is already a Blackwell tcgen05 path.  The document lists
    bf16 as a legacy type with the same alignment rules as Hopper and legal
    1SM tile shapes such as 64x128 and 64x256, which match the head64 QK and
    S@V shapes used by fwd/head64/phase1.cuh.

  - Native narrow-precision tcgen05.mma is not a drop-in replacement for the
    current prefill.  The narrow-precision tables are for narrow A and narrow B
    operands, not bf16 Q/S times fp8 K/V.  A native fp8 QK path would need Q to
    be quantized too.  A native fp8 S@V path would need dynamic softmax S to be
    quantized too, which is a much larger change.

  - Block-scaled MX instructions attach scale factors along GEMM-K chunks
    of 16/32 elements, or 32/64 elements for sparse.  V3.2 stores fp32 NoPE
    scales per token and per 128 dims.  QK can maybe re-express that by
    replicating one V3.2 scale over several MX K chunks, but S@V cannot map
    cleanly because the reduction K is the topk/token axis while V scales vary
    per token and dim group.

So the first B300 experiment should keep tcgen05 bf16 UMMA and only replace the
global bf16 workspace with in-kernel fp8 cache consumption plus bf16 smem
materialization.  Native MX fp8 should be treated as a separate research path.
*/

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cstdint>

#include "defines.h"
#include "params.h"
#include "utils.h"

namespace sm100::fwd_v32_fp8_prefill {

using uint8 = std::uint8_t;

static constexpr int D_QK = 576;
static constexpr int D_V = 512;
static constexpr int D_NOPE = 512;
static constexpr int D_ROPE = 64;
static constexpr int QUANT_GROUP = 128;
static constexpr int NUM_SCALES = 4;
static constexpr int BYTES_NOPE = D_NOPE;
static constexpr int BYTES_SCALE = NUM_SCALES * sizeof(float);
static constexpr int BYTES_ROPE = D_ROPE * sizeof(bf16);
static constexpr int BYTES_PER_TOKEN = BYTES_NOPE + BYTES_SCALE + BYTES_ROPE;  // 656

static constexpr int B_H = 128;
static constexpr int B_TOPK = 128;
static constexpr int K_ROWS_PER_CTA = B_TOPK / 2;
static constexpr int NUM_THREADS = 512;
static constexpr int NUM_BUFS = 2;

// B300 table-equivalent V3.2 target.  These constants mirror
// sm100::fwd::head64::config.h and are used by the head64 producer sketches
// below.  They intentionally do not replace B_H/B_TOPK above because the
// existing standalone theory kernel body is still the head128 contrast shell.
static constexpr int B300_B_H = 64;
static constexpr int B300_B_TOPK = 64;
static constexpr int B300_NUM_BUFS = 3;

// These match sm100::fwd::head128::KernelTemplate<576>.
// QK is computed in two pieces:
//   1. local sQ part, D_sQ = 192, via SS UMMA
//   2. tmem tQ part, D_tQ = 384, via TS UMMA
static constexpr int D_sQ = 192;
static constexpr int D_tQ = 384;

static_assert(BYTES_PER_TOKEN == 656);
static_assert(D_QK == D_NOPE + D_ROPE);
static_assert(D_V == D_NOPE);
static_assert(D_NOPE % QUANT_GROUP == 0);
static_assert(D_sQ + D_tQ == D_QK);

struct V32TokenRef {
    const uint8* base;
    bool valid;
};

struct V32TokenParts {
    const uint8* nope;      // 512 x e4m3 bytes
    const float* scales;    // 4 x fp32, one scale per 128 NoPE dims
    const bf16* rope;       // 64 x bf16
};

struct V32PrefillCta {
    int cta_idx;
    int s_q_idx;
    int effective_topk_length;
    int num_k_blocks;
    const int* indices;
};

__device__ __forceinline__ V32TokenParts split_token(const V32TokenRef token) {
    return {
        token.base,
        reinterpret_cast<const float*>(token.base + BYTES_NOPE),
        reinterpret_cast<const bf16*>(token.base + BYTES_NOPE + BYTES_SCALE),
    };
}

__device__ __forceinline__ V32TokenRef resolve_token(
    const SparseAttnFwdParams& params,
    int token_idx,
    int topk_pos,
    int effective_topk_length) {
    const bool valid =
        token_idx >= 0 &&
        token_idx < params.s_kv &&
        topk_pos < effective_topk_length;

    if (!valid) {
        return {nullptr, false};
    }

    const int block_idx = static_cast<unsigned int>(token_idx) / params.page_block_size;
    const int idx_in_block = static_cast<unsigned int>(token_idx) % params.page_block_size;

    const auto* kv_bytes = reinterpret_cast<const uint8*>(params.kv);
    const uint8* base =
        kv_bytes +
        static_cast<int64_t>(block_idx) * params.stride_kv_block +
        static_cast<int64_t>(idx_in_block) * BYTES_PER_TOKEN;

    return {base, true};
}

__device__ __forceinline__ bf16 fp8_nope_to_bf16(
    const V32TokenParts parts,
    int col) {
    const auto raw = reinterpret_cast<const __nv_fp8_e4m3*>(parts.nope);
    const float scale = __ldg(parts.scales + col / QUANT_GROUP);
    const float value = static_cast<float>(raw[col]) * scale;
    return bf16(value);
}

__device__ __forceinline__ nv_bfloat162 fp8_nope_x2_to_bf16x2(
    const V32TokenParts parts,
    int even_col) {
    const auto raw = reinterpret_cast<const __nv_fp8x2_e4m3*>(parts.nope + even_col);
    const float scale = __ldg(parts.scales + even_col / QUANT_GROUP);
    const float2 value = static_cast<float2>(*raw);
    return __float22bfloat162_rn(float2{value.x * scale, value.y * scale});
}

__device__ __forceinline__ bf16 load_k_element_bf16(
    const V32TokenRef token,
    int col) {
    if (!token.valid) {
        return bf16(0.0f);
    }

    const V32TokenParts parts = split_token(token);
    if (col < D_NOPE) {
        return fp8_nope_to_bf16(parts, col);
    }
    return __ldg(parts.rope + (col - D_NOPE));
}

__device__ __forceinline__ bf16 load_v_element_bf16(
    const V32TokenRef token,
    int col) {
    if (!token.valid) {
        return bf16(0.0f);
    }

    const V32TokenParts parts = split_token(token);
    return fp8_nope_to_bf16(parts, col);
}

__device__ __forceinline__ float load_q_bf16(
    const SparseAttnFwdParams& params,
    int s_q_idx,
    int h_idx,
    int col) {
    const bf16* q =
        params.q +
        static_cast<int64_t>(s_q_idx) * params.stride_q_s_q +
        static_cast<int64_t>(h_idx) * params.stride_q_h_q;
    return float(__ldg(q + col));
}

/*
Slow reference kernel
---------------------
This is not the performance target.  It is a complete standalone V3.2 fp8
prefill kernel for proving the data path:

  - no bf16 KV workspace in HBM
  - K is reconstructed from 512 fp8 NoPE + 64 bf16 RoPE
  - V is reconstructed from the same 512 fp8 NoPE values
  - sparse topk attention is computed directly

One CUDA thread computes one (s_q, h_q) output row.  That is intentionally slow,
but it gives a readable correctness baseline for the producer/UMMA kernel shell
below.
*/
__global__ void sparse_attn_v32_fp8_prefill_reference_kernel(
    __grid_constant__ const SparseAttnFwdParams params) {
    const int s_q_idx = blockIdx.x;
    const int h_idx = blockIdx.y;

    if (s_q_idx >= params.s_q || h_idx >= params.h_q) {
        return;
    }

    const int requested_topk_length =
        params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int effective_topk_length =
        requested_topk_length < params.topk ? requested_topk_length : params.topk;
    const int* indices =
        params.indices +
        static_cast<int64_t>(s_q_idx) * params.stride_indices_s_q;

    float acc[D_V];
    for (int col = 0; col < D_V; ++col) {
        acc[col] = 0.0f;
    }

    float m = -CUDART_INF_F;
    float l = 0.0f;
    bool has_valid_token = false;

    for (int pos = 0; pos < effective_topk_length; ++pos) {
        const int token_idx = __ldg(indices + pos);
        const V32TokenRef token = resolve_token(params, token_idx, pos, effective_topk_length);
        if (!token.valid) {
            continue;
        }
        has_valid_token = true;

        float score = 0.0f;
        for (int col = 0; col < D_QK; ++col) {
            score += load_q_bf16(params, s_q_idx, h_idx, col) *
                     float(load_k_element_bf16(token, col));
        }
        score *= params.sm_scale;

        const float new_m = fmaxf(m, score);
        const float old_scale = expf(m - new_m);
        const float cur_scale = expf(score - new_m);

        for (int col = 0; col < D_V; ++col) {
            const float v = float(load_v_element_bf16(token, col));
            acc[col] = acc[col] * old_scale + v * cur_scale;
        }

        l = l * old_scale + cur_scale;
        m = new_m;
    }

    bf16* out =
        params.out +
        static_cast<int64_t>(s_q_idx) * params.h_q * params.d_v +
        static_cast<int64_t>(h_idx) * params.d_v;
    float* max_logits = params.max_logits + static_cast<int64_t>(s_q_idx) * params.h_q + h_idx;
    float* lse = params.lse + static_cast<int64_t>(s_q_idx) * params.h_q + h_idx;

    if (!has_valid_token || l == 0.0f) {
        for (int col = 0; col < D_V; ++col) {
            out[col] = bf16(0.0f);
        }
        *max_logits = -CUDART_INF_F;
        *lse = -CUDART_INF_F;
        return;
    }

    const float real_lse = m + logf(l);
    float sink_scale = 1.0f;
    if (params.attn_sink != nullptr) {
        const float sink = __ldg(params.attn_sink + h_idx);
        sink_scale = l / (l + expf(sink - m));
    }

    const float inv_l = sink_scale / l;
    for (int col = 0; col < D_V; ++col) {
        out[col] = bf16(acc[col] * inv_l);
    }
    *max_logits = m;
    *lse = real_lse;
}

void run_v32_fp8_prefill_reference_kernel_experimental(const SparseAttnFwdParams& params) {
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.page_block_size > 0);
    KU_ASSERT(params.stride_kv_block >= params.page_block_size * BYTES_PER_TOKEN);
    KU_ASSERT(params.s_kv == params.num_blocks * params.page_block_size);

    const dim3 grid(params.s_q, params.h_q, 1);
    const dim3 block(1, 1, 1);
    sparse_attn_v32_fp8_prefill_reference_kernel<<<grid, block, 0, params.stream>>>(params);
}

template<class Store>
__device__ __forceinline__ void materialize_k_row(
    const SparseAttnFwdParams& params,
    const int* indices,
    Store&& store,
    int topk_block,
    int local_row,
    int global_topk_row,
    int col_begin,
    int col_end,
    int effective_topk_length) {
    const int token_idx = __ldg(indices + topk_block * B_TOPK + global_topk_row);
    const V32TokenRef token = resolve_token(
        params,
        token_idx,
        topk_block * B_TOPK + global_topk_row,
        effective_topk_length);

    for (int col = col_begin; col < col_end; ++col) {
        store(local_row, col, load_k_element_bf16(token, col));
    }
}

template<class Store>
__device__ __forceinline__ void materialize_k_row_x2(
    const SparseAttnFwdParams& params,
    const int* indices,
    Store&& store_x2,
    int topk_block,
    int local_row,
    int global_topk_row,
    int col_begin,
    int col_end,
    int effective_topk_length) {
    /*
    Vectorized correctness path for the eventual performance producer.
    The scalar materialize_k_row() is easier to inspect, so
    produce_k_tile_v32_fp8_part() calls that today.  This x2 version documents
    the next mechanical step:
    fp8x2 -> bf16x2 conversion followed by a 32-bit or 128-bit shared-memory
    store, depending on how the final swizzled store functor is implemented.
    */

    const int token_idx = __ldg(indices + topk_block * B_TOPK + global_topk_row);
    const V32TokenRef token = resolve_token(
        params,
        token_idx,
        topk_block * B_TOPK + global_topk_row,
        effective_topk_length);

    if (!token.valid) {
        for (int col = col_begin; col < col_end; col += 2) {
            store_x2(local_row, col, __float22bfloat162_rn(float2{0.0f, 0.0f}));
        }
        return;
    }

    const V32TokenParts parts = split_token(token);
    int col = col_begin;
    for (; col + 1 < col_end && col < D_NOPE; col += 2) {
        store_x2(local_row, col, fp8_nope_x2_to_bf16x2(parts, col));
    }
    for (; col < col_end; col += 2) {
        const int rope_col = col - D_NOPE;
        const bf16 lo = __ldg(parts.rope + rope_col);
        const bf16 hi = __ldg(parts.rope + rope_col + 1);
        store_x2(local_row, col, __float22bfloat162_rn(float2{float(lo), float(hi)}));
    }
}

template<class Store>
__device__ __forceinline__ void materialize_v_row(
    const SparseAttnFwdParams& params,
    const int* indices,
    Store&& store,
    int topk_block,
    int local_row,
    int global_topk_row,
    int col_begin,
    int col_end,
    int effective_topk_length) {
    const int token_idx = __ldg(indices + topk_block * B_TOPK + global_topk_row);
    const V32TokenRef token = resolve_token(
        params,
        token_idx,
        topk_block * B_TOPK + global_topk_row,
        effective_topk_length);

    for (int col = col_begin; col < col_end; ++col) {
        store(local_row, col, load_v_element_bf16(token, col));
    }
}

/*
B300/head64 producer sketches
-----------------------------
These are the producer-side pieces that actually match the measured B300
heads=64 case.

The important difference from the head128 sketch is that there is no separate
V tile.  head64 stores NoPE once in plan.u.k.k_nope[cur_buf].  QK consumes the
same tile as K-NoPE, and S@V later reinterprets that same tile as V.  Therefore
the right fp8 path is:

  - dequant NoPE once per selected token into the existing KNoPE smem layout
  - signal bar_kv_nope_ready[cur_buf][part_idx] as each half is ready
  - copy RoPE bf16 separately into KRoPE smem and signal bar_kv_rope_ready
  - let the unchanged bf16 UMMA path consume those smem tiles

This transfers Hopper's fine-grained producer/GMMA overlap idea, but not its
seesaw schedule and not a separate VProducer.
*/

template<class Store>
__device__ __forceinline__ void materialize_head64_nope_row(
    const SparseAttnFwdParams& params,
    const int* indices,
    Store&& store,
    int topk_block,
    int local_row,
    int global_topk_row,
    int col_begin,
    int col_end,
    int effective_topk_length) {
    const int topk_pos = topk_block * B300_B_TOPK + global_topk_row;
    const int token_idx = __ldg(indices + topk_pos);
    const V32TokenRef token = resolve_token(params, token_idx, topk_pos, effective_topk_length);

    for (int col = col_begin; col < col_end; ++col) {
        store(local_row, col, load_v_element_bf16(token, col));
    }
}

template<class Store>
__device__ __forceinline__ void materialize_head64_rope_row(
    const SparseAttnFwdParams& params,
    const int* indices,
    Store&& store,
    int topk_block,
    int local_row,
    int global_topk_row,
    int effective_topk_length) {
    const int topk_pos = topk_block * B300_B_TOPK + global_topk_row;
    const int token_idx = __ldg(indices + topk_pos);
    const V32TokenRef token = resolve_token(params, token_idx, topk_pos, effective_topk_length);

    for (int col = 0; col < D_ROPE; ++col) {
        store(local_row, col, load_k_element_bf16(token, D_NOPE + col));
    }
}

template<class Plan, class NoPEStore>
__device__ void produce_head64_nope_tile_v32_fp8_part(
    const SparseAttnFwdParams& params,
    const int* indices,
    Plan& plan,
    NoPEStore&& store_nope,
    int topk_block,
    int part_idx,
    int effective_topk_length) {
    /*
    This replaces the head64 bf16 TMA gather of plan.u.k.k_nope[cur_buf].

    Existing bf16 order in fwd/head64/phase1.cuh:

      plan.bar_sv_done[cur_buf].wait(...)
      load NoPE part 0 with TMA gather
      arrive bar_kv_nope_ready[cur_buf][0]
      load NoPE part 1 with TMA gather
      arrive bar_kv_nope_ready[cur_buf][1]

    V3.2 should keep that exact barrier shape.  The body changes from TMA
    gather of bf16 to gather of a 656B token record plus fp8->bf16 conversion.
    There is no second V materialization pass.
    */
    const int col_begin = part_idx * (D_NOPE / 2);
    const int col_end = col_begin + (D_NOPE / 2);

    for (int row = 0; row < B300_B_TOPK; ++row) {
        materialize_head64_nope_row(
            params,
            indices,
            store_nope,
            topk_block,
            row,
            row,
            col_begin,
            col_end,
            effective_topk_length);
    }

    (void)plan;
}

template<class Plan, class RoPEStore>
__device__ void produce_head64_rope_tile_v32_bf16(
    const SparseAttnFwdParams& params,
    const int* indices,
    Plan& plan,
    RoPEStore&& store_rope,
    int topk_block,
    int effective_topk_length) {
    /*
    This replaces the current head64 cp.async RoPE copy source address.
    It should still run independently from NoPE and should still be gated by
    bar_qk_rope_done/bar_kv_rope_ready, because RoPE is only needed for QK.
    */
    for (int row = 0; row < B300_B_TOPK; ++row) {
        materialize_head64_rope_row(
            params,
            indices,
            store_rope,
            topk_block,
            row,
            row,
            effective_topk_length);
    }

    (void)plan;
}

__device__ __forceinline__ V32PrefillCta make_cta_context(
    const SparseAttnFwdParams& params) {
    const int cta_idx = blockIdx.x & 1;
    const int s_q_idx = blockIdx.x >> 1;
    const int requested_topk_length =
        params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int effective_topk_length =
        requested_topk_length < params.topk ? requested_topk_length : params.topk;
    const int raw_num_k_blocks = (effective_topk_length + B_TOPK - 1) / B_TOPK;
    const int num_k_blocks = raw_num_k_blocks > 0 ? raw_num_k_blocks : 1;
    const int* indices = params.indices + s_q_idx * params.stride_indices_s_q;

    return {cta_idx, s_q_idx, effective_topk_length, num_k_blocks, indices};
}

/*
How this maps onto the existing SM100 head128 d=576 prefill kernel
-----------------------------------------------------------------
This section is a contrast/reference path, not the B300 heads=64 target.
It is still useful for understanding why a separate KProducer/VProducer split
can be correct for head128 and wrong for head64.

Existing bf16 producer in csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:

  K producer:
    - each CTA owns 64 of the 128 topk rows
    - TMA gathers bf16 K[D_QK=576] into plan.u.s.k
    - split readiness:
        bar_k_part0_ready: first D_sQ = 192 dims
        bar_k_part1_ready: remaining D_tQ = 384 dims

  V producer:
    - both CTAs see all 128 topk rows
    - CTA0 produces V columns 0..255
    - CTA1 produces V columns 256..511
    - split readiness:
        bar_v_part0_ready: first 64 rows
        bar_v_part1_ready: second 64 rows

For V3.2 fp8 prefill the MMA side should not change.  Only the two producer
warpgroups change:

  K:
    - gather a 656B V3.2 token record by address
    - dequant NoPE cols 0..511 with fp32 scale[col/128]
    - copy RoPE cols 512..575 as bf16
    - write the same bf16 smem K layout consumed by the current QK UMMA

  V:
    - use only NoPE cols 0..511
    - dequant with the same 4 fp32 scales
    - write the same bf16 smem V layout consumed by the current S@V UMMA

The simple element loops above are correctness-first.  A performance version
should replace them with the vectorized pattern already used by
csrc/sm100/decode/head64/kernel.cuh:

  - one warp produces token coordinates and four scales
  - raw NoPE is gathered as 128-bit/256-bit chunks
  - dequant warpgroup converts fp8x2 -> bf16x2 and stores shared::cta.b128
  - RoPE is copied as bf16 without arithmetic

Target overlap, expressed at topk-block granularity:

  block b:
    K part0 ready -> QK part0 starts
    K part1 ready -> QK part1 finishes P_b
    softmax builds S_b
    V part0/part1 ready -> S_b @ V_b finishes O update

  while block b is in QK/softmax/SV:
    KProducer prepares K_{b+1} as soon as qk_part/qk_done frees the buffer
    VProducer prepares V_b only before S@V needs it

The important rule is that V work must not sit in front of QK.  QK only needs
K, so a new V3.2 implementation should preserve the regular prefill
K-part/V-part barrier separation instead of introducing a single kv_ready gate.
*/

/*
B300/head64 transfer matrix for Hopper decode ideas
---------------------------------------------------
Apply:
  - fine-grained producer/UMMA overlap, but express it as NoPE part0/part1 and
    RoPE readiness, matching head64 barriers.
  - fp8 cache consumption in the producer, followed by bf16 smem materialization
    for the existing SM100 bf16 UMMA path.
  - keep the current head64 legacy bf16 tcgen05 tile choices.  QK uses a
    64x128-style shape after the dual-gemm view, and S@V uses 64x256; both are
    legal 1SM legacy tile shapes in Blackwell SM100 GEMMs.md.

Measure before adopting:
  - cache hint changes.  head64 currently uses EVICT_LAST for sparse KV-NoPE
    TMA gather and cp.async for RoPE.  Hopper's EVICT_FIRST result is not a
    portable rule for this B300 prefill access pattern.

Do not copy:
  - Hopper seesaw math.  SM100 prefill already has an online softmax/S@V
    pipeline with TMEM and tcgen05 synchronization.
  - a separate VProducer for head64.  NoPE is both K-NoPE and V here.
  - PDL as a core single-kernel optimization.  It only matters if we keep a
    separate dequantize_v32_kv_cache kernel before prefill, which is precisely
    the path this file is trying to eliminate.
  - native MX fp8 tensor core as the first step.  Blackwell's block-scaled
    narrow-precision path wants narrow A and narrow B operands plus K-axis scale
    metadata.  Current prefill has bf16 Q, dynamic bf16/float softmax S, and a
    V3.2 cache whose scales are token/dim-group metadata.  That is a separate
    kernel design, not a producer-only replacement.
*/

template<class Plan, class KStore>
__device__ void produce_k_tile_v32_fp8_part(
    const SparseAttnFwdParams& params,
    const int* indices,
    Plan& plan,
    KStore&& store_k,
    int cta_idx,
    int topk_block,
    int col_begin,
    int col_end,
    int effective_topk_length) {
    const int row_base = cta_idx * K_ROWS_PER_CTA;

    /*
    K must stay split exactly like the bf16 producer:

      if (topk_block > 0)
          wait bar_qk_part_done[(topk_block - 1) % NUM_BUFS]
      fill cols [0, D_sQ)
      arrive bar_k_part0_ready[cur_buf]

      if (topk_block > 0)
          wait bar_qk_done[(topk_block - 1) % NUM_BUFS]
      fill cols [D_sQ, D_QK)
      arrive bar_k_part1_ready[cur_buf]

    Keeping part0/part1 as separate calls avoids overwriting a double buffer
    before the MMA warpgroup has consumed that part of the previous block.
    The store_k functor must write to plan.u.s.k using the same swizzled
    SmemLayoutKTiles<576/64> coordinates that the existing bf16 TMA path uses.
    */

    for (int row = 0; row < K_ROWS_PER_CTA; ++row) {
        materialize_k_row(
            params,
            indices,
            store_k,
            topk_block,
            row,
            row_base + row,
            col_begin,
            col_end,
            effective_topk_length);
    }

    (void)plan;
}

template<class Plan, class VStore>
__device__ void produce_v_tile_v32_fp8_part(
    const SparseAttnFwdParams& params,
    const int* indices,
    Plan& plan,
    VStore&& store_v,
    int cta_idx,
    int topk_block,
    int row_begin,
    int row_end,
    int effective_topk_length) {
    const int col_begin = cta_idx ? D_V / 2 : 0;
    const int col_end = col_begin + D_V / 2;

    /*
    V must also stay split like the bf16 producer:

      if (topk_block > 0)
          wait bar_sv_part_done[(topk_block - 1) % NUM_BUFS]
      fill rows [0, B_TOPK/2)
      arrive bar_v_part0_ready[cur_buf]

      if (topk_block > 0)
          wait bar_sv_done[(topk_block - 1) % NUM_BUFS]
      fill rows [B_TOPK/2, B_TOPK)
      arrive bar_v_part1_ready[cur_buf]

    The existing MMA layout expects CTA0 to write
    V[:, 0:256] and CTA1 to write V[:, 256:512], for all 128 topk rows.
    V3.2 stores V in the 512 NoPE values; the 64 RoPE bf16 values are K-only.
    */

    for (int row = row_begin; row < row_end; ++row) {
        materialize_v_row(
            params,
            indices,
            store_v,
            topk_block,
            row,
            row,
            col_begin,
            col_end,
            effective_topk_length);
    }

    (void)plan;
}

template<class Plan, class MakeKStore, class MakeVStore>
__device__ void v32_fp8_prefill_phase1_kernel_body(
    const SparseAttnFwdParams& params,
    Plan& plan,
    MakeKStore&& make_k_store,
    MakeVStore&& make_v_store) {
    const V32PrefillCta ctx = make_cta_context(params);

    /*
    This is the intended standalone control-flow skeleton.

    warpgroup 0:
      same as bf16 head128 phase1:
        - initialize barriers
        - TMA load Q
        - allocate TMEM
        - run online softmax / exp / S materialization

    warpgroup 1:
      V3.2 fp8 K producer, implemented below.

    warpgroup 2:
      V3.2 fp8 V producer, implemented below.

    warpgroup 3:
      same as bf16 head128 phase1:
        - wait for bf16 K tiles
        - QK UMMA
        - wait for bf16 V tiles
        - S@V UMMA
        - store O/LSE/max_logits

    The reason this still counts as a separate fp8 prefill kernel is that HBM
    consumption, token addressing, and K/V materialization are all fp8-native.
    The tensor core math intentionally remains bf16 after smem materialization.
    */

    const int warpgroup_idx = threadIdx.x / 128;

    if (warpgroup_idx == 1) {
        for (int topk_block = 0; topk_block < ctx.num_k_blocks; ++topk_block) {
            const int cur_buf = topk_block & 1;
            auto store_k = make_k_store(plan, ctx.cta_idx, topk_block);

            if (topk_block > 0) {
                // plan.bar_qk_part_done[(topk_block - 1) % NUM_BUFS].wait(((topk_block - 1) / NUM_BUFS) & 1);
            }

            produce_k_tile_v32_fp8_part(
                params,
                ctx.indices,
                plan,
                store_k,
                ctx.cta_idx,
                topk_block,
                0,
                D_sQ,
                ctx.effective_topk_length);
            // plan.bar_k_part0_ready[cur_buf].arrive_and_expect_tx(B_TOPK * D_sQ * sizeof(bf16));

            if (topk_block > 0) {
                // plan.bar_qk_done[(topk_block - 1) % NUM_BUFS].wait(((topk_block - 1) / NUM_BUFS) & 1);
            }

            produce_k_tile_v32_fp8_part(
                params,
                ctx.indices,
                plan,
                store_k,
                ctx.cta_idx,
                topk_block,
                D_sQ,
                D_sQ + D_tQ,
                ctx.effective_topk_length);
            // plan.bar_k_part1_ready[cur_buf].arrive_and_expect_tx(B_TOPK * D_tQ * sizeof(bf16));

            (void)cur_buf;
        }
    } else if (warpgroup_idx == 2) {
        for (int topk_block = 0; topk_block < ctx.num_k_blocks; ++topk_block) {
            const int cur_buf = topk_block & 1;
            auto store_v = make_v_store(plan, ctx.cta_idx, topk_block);

            if (topk_block > 0) {
                // plan.bar_sv_part_done[(topk_block - 1) % NUM_BUFS].wait(((topk_block - 1) / NUM_BUFS) & 1);
            }

            produce_v_tile_v32_fp8_part(
                params,
                ctx.indices,
                plan,
                store_v,
                ctx.cta_idx,
                topk_block,
                0,
                B_TOPK / 2,
                ctx.effective_topk_length);
            // plan.bar_v_part0_ready[cur_buf].arrive_and_expect_tx((B_TOPK / 2) * D_V * sizeof(bf16));

            if (topk_block > 0) {
                // plan.bar_sv_done[(topk_block - 1) % NUM_BUFS].wait(((topk_block - 1) / NUM_BUFS) & 1);
            }

            produce_v_tile_v32_fp8_part(
                params,
                ctx.indices,
                plan,
                store_v,
                ctx.cta_idx,
                topk_block,
                B_TOPK / 2,
                B_TOPK,
                ctx.effective_topk_length);
            // plan.bar_v_part1_ready[cur_buf].arrive_and_expect_tx((B_TOPK / 2) * D_V * sizeof(bf16));

            (void)cur_buf;
        }
    } else {
        /*
        Keep the bf16 d=576 phase1 logic here:
          - Q TMA and UTCCP
          - QK UMMA over K smem
          - online softmax
          - S@V UMMA over V smem

        The only required invariant is that make_k_store/make_v_store write the
        exact same swizzled smem layouts as the bf16 TMA producers.  Once that is
        true, the MMA side cannot tell whether K/V came from bf16 HBM or from a
        V3.2 fp8 cache dequantized inside this kernel.
        */
    }
}

template<class Plan, class MakeKStore, class MakeVStore>
__global__ void __launch_bounds__(NUM_THREADS, 1, 2)
sparse_attn_v32_fp8_prefill_phase1_kernel_theory(
    __grid_constant__ const SparseAttnFwdParams params,
    __grid_constant__ const MakeKStore make_k_store,
    __grid_constant__ const MakeVStore make_v_store) {
    extern __shared__ char wksp_buf[];
    Plan& plan = *reinterpret_cast<Plan*>(wksp_buf);
    v32_fp8_prefill_phase1_kernel_body(params, plan, make_k_store, make_v_store);
}

void run_v32_fp8_prefill_phase1_kernel_experimental(const SparseAttnFwdParams& params) {
    /*
    This is intentionally not launched yet.  It is the head128 contrast shell,
    not the B300 heads=64 target.

    Required host-side contract:

      q:       [s_q, 128, 576], bf16
      kv:      [num_blocks, page_block_size, 1, 656], byte-addressable V3.2 cache
      indices: [s_q, 1, topk], int32 token ids encoded as block*page_block_size + offset
      d_v:     512
      topk:    multiple of 128

    Required implementation step:

      Use sparse_attn_v32_fp8_prefill_phase1_kernel_theory() as the single-file
      kernel shell.  Its grid/block contract should be the same as the bf16
      head128 d=576 prefill:

          grid    = dim3(2 * params.s_q, 1, 1)
          block   = dim3(512, 1, 1)
          cluster = dim3(2, 1, 1)

      Keep the existing Q/O TMA setup, TMEM allocation, scale/softmax, QK UMMA,
      and S@V UMMA.  Replace only the K and V producer warpgroup bodies with
      produce_k_tile_v32_fp8_part() and produce_v_tile_v32_fp8_part().  Pass

          indices = params.indices + s_q_idx * params.stride_indices_s_q

      just like the existing bf16 head128 phase1 kernel does.

    This keeps the head128 contrast experiment scoped:

      1. no global bf16 KV workspace
      2. no native fp8 tensor-core dependency
      3. same numerical route as bf16 prefill after shared-memory dequant
    */

    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.topk % B_TOPK == 0);
    KU_ASSERT(params.page_block_size > 0);
    KU_ASSERT(params.stride_kv_block >= params.page_block_size * BYTES_PER_TOKEN);
    KU_ASSERT(params.s_kv == params.num_blocks * params.page_block_size);
}

void run_v32_fp8_head64_prefill_phase1_kernel_experimental(const SparseAttnFwdParams& params) {
    /*
    This is the B300-relevant experimental contract.

    Source of the shape:
      - heads=64 and topk=2048 came from the user-provided B300 timing table.
      - d_qk=576 and d_v=512 come from the V3.2 NoPE+RoPE format and the
        existing SM100 d=576 regular prefill path.

    Required host-side contract:

      q:       [s_q, 64, 576], bf16
      kv:      [num_blocks, page_block_size, 1, 656], byte-addressable V3.2 cache
      indices: [s_q, 1, topk], int32 token ids encoded as block*page_block_size + offset
      d_v:     512
      topk:    multiple of 64

    Launch shape should match fwd/head64/phase1.cuh:

      grid  = dim3(params.s_q, 1, 1)
      block = dim3(384, 1, 1)
      no 2-CTA cluster

    Producer replacement plan:

      1. Keep Q TMA, TMEM allocation, QK UMMA, online softmax, S@V UMMA, and
         O/LSE store from fwd/head64/phase1.cuh unchanged.

      2. Replace the current KV-NoPE bf16 TMA gather with
         produce_head64_nope_tile_v32_fp8_part().

         Preserve this order for each cur_buf:

           wait bar_sv_done[cur_buf]
           produce NoPE cols [0, 256)
           signal bar_kv_nope_ready[cur_buf][0]
           produce NoPE cols [256, 512)
           signal bar_kv_nope_ready[cur_buf][1]

         The same smem tile is then consumed first as K-NoPE by QK and later
         as V by S@V.  Do not add a separate VProducer.

      3. Keep RoPE as a separate K-only path:

           wait bar_qk_rope_done
           copy token RoPE bf16 from offset 512 + 4*sizeof(float)
           signal bar_kv_rope_ready

      4. Keep invalid-index handling equivalent to the bf16 path.  If a whole
         NoPE part is skipped, complete the corresponding ready transaction
         with the same byte count that QK expects.

    This uses Blackwell tcgen05 bf16 UMMA, so it is still an SM100-native
    tensor-core path.  It only avoids the global bf16 KV workspace.
    */

    KU_ASSERT(params.h_q == B300_B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.topk % B300_B_TOPK == 0);
    KU_ASSERT(params.page_block_size > 0);
    KU_ASSERT(params.stride_kv_block >= params.page_block_size * BYTES_PER_TOKEN);
    KU_ASSERT(params.s_kv == params.num_blocks * params.page_block_size);
}

}  // namespace sm100::fwd_v32_fp8_prefill
