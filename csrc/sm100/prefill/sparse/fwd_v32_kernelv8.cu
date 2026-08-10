// fwd_v32_kernelv8.cu -- standalone V3.2 FP8 native head64 sparse prefill
// kernel for SM100.
//
// This translation unit owns the production v8 implementation and entrypoints.
// It has no source, symbol, or link dependency on earlier experimental
// versions. Internal compile-time switches preserve the audited code
// generation; this file instantiates the 512-thread dual-DQ pair-pipeline
// configuration exclusively.
//
// The producer roles are metadata, raw TMA gather4, RoPE fetch, and the single
// elected Q/QK/S@V issuer.  Both dequant workgroups cooperate on the same tile;
// there is no data partition or extra buffering between them.
//
// The selected NoPE tile is dequantized exactly once.  Q remains BF16, and the
// same immutable BF16 tile is consumed first by QK and later by S@V.  There is
// no global BF16 workspace, Q requantization, CTA cluster, or DSM traffic.

#include <climits>
#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cuda_host_adapter.hpp>
#include <cutlass/float8.h>

#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"
#include "sm100/prefill/sparse/common_subroutine.h"
#include "sm100/prefill/sparse/v8_trace_schema.h"

namespace sm100::fwd::head64::v8 {

using namespace cute;
using fp8_e4m3 = cutlass::float_e4m3_t;

// -----------------------------------------------------------------------------
// Fixed V3.2/head64 contract
// -----------------------------------------------------------------------------

static constexpr int D_Q = 576;
static constexpr int D_NOPE = 512;
static constexpr int D_ROPE = 64;
static constexpr int D_V = 512;

static constexpr int B_H = 64;
static constexpr int B_TOPK = 64;
static constexpr int NUM_BUFS = 2;
static constexpr int SINGLE_DQ_NUM_THREADS = 384;
static constexpr int V8_NUM_THREADS = 512;
static constexpr int V8_WG0_REGS = 208;
static constexpr int V8_DQ_REGS = 112;

// The traced variant records one representative CTA.  Production launches use
// v8_kernel<false>, so all trace branches and global stores are compiled out.
// The Python exporter interprets the fixed [track, tile-or-summary, point]
// layout and converts %globaltimer nanoseconds to Perfetto slices.
struct V8TraceParams {
    std::uint64_t* timestamps;
    int s_q_idx;
    int num_tiles;
    int num_tracks;
};

template <bool ENABLED>
CUTE_DEVICE void trace_stamp(
    V8TraceParams const& trace,
    int track,
    int tile,
    int point) {
    if constexpr (ENABLED) {
        if (trace.timestamps != nullptr &&
            static_cast<int>(blockIdx.x) == trace.s_q_idx &&
            track >= 0 && track < trace.num_tracks &&
            tile >= 0 && tile <= trace.num_tiles &&
            point >= 0 && point < TRACE_POINTS) {
            std::uint64_t timestamp_ns;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp_ns));
            volatile std::uint64_t* dst = trace.timestamps +
                (static_cast<std::int64_t>(track) * (trace.num_tiles + 1) + tile) *
                    TRACE_POINTS +
                point;
            *dst = timestamp_ns;
        }
    }
}

static constexpr int V32_TOKEN_BYTES = 656;
static constexpr int V32_SCALE_OFFSET = 512;
static constexpr int V32_ROPE_OFFSET = 528;
static constexpr int V32_NUM_SCALES = 4;
static constexpr int V32_SCALE_GROUP = 128;

static constexpr int RAW_ROW_BYTES = D_NOPE;
static constexpr int RAW_TILE_BYTES = B_TOPK * RAW_ROW_BYTES;
static constexpr int BF16_TILE_BYTES = B_TOPK * D_NOPE * sizeof(bf16);
static constexpr int ROPE_TILE_BYTES = B_TOPK * D_ROPE * sizeof(bf16);

static_assert(D_NOPE + D_ROPE == D_Q);
static_assert(D_NOPE == D_V);
static_assert(D_NOPE / V32_SCALE_GROUP == V32_NUM_SCALES);
static_assert(V32_SCALE_OFFSET + V32_NUM_SCALES * sizeof(float) == V32_ROPE_OFFSET);
static_assert(V32_ROPE_OFFSET + D_ROPE * sizeof(bf16) == V32_TOKEN_BYTES);

// -----------------------------------------------------------------------------
// TMEM and SMEM layouts: identical to the production BF16 head64 mainloop
// -----------------------------------------------------------------------------

namespace tmem_cols {
static constexpr int O = 0;            // 0..255
static constexpr int Q = 256;          // 256..383, Q NoPE
static constexpr int Q_ROPE = 384;     // 384..399, Q RoPE
static constexpr int P = 400;          // 400..463
}  // namespace tmem_cols

using SmemLayoutQNoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_NOPE>>{},
    Step<_1, _2>{}), Shape<_1, _1>{}));

using SmemLayoutQRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_ROPE>>{},
    Step<_1, _2>{}), Shape<_1, _1>{}));

template <int NUM_TILES>
using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64 * NUM_TILES>>{},
    Step<_1, _2>{}), Shape<_1, _1>{}));

using SmemLayoutO = SmemLayoutOTiles<8>;

template <int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<64 * NUM_TILES>>{},
    Step<_1, _2>{}), Shape<_1, _1>{}));

using SmemLayoutKNoPE = SmemLayoutKTiles<8>;

using SmemLayoutV = decltype(coalesce(
    composition(
        SmemLayoutKNoPE{},
        Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}),
    Shape<_1, _1>{}));

using SmemLayoutKRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<D_ROPE>>{},
    Step<_1, _2>{}), Shape<_1, _1>{}));

using SmemLayoutKNoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK * 2>, Int<D_NOPE / 2>>{},
    Step<_1, _2>{}), Shape<_1, _1>{}));

using SmemLayoutKRoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK * 2>, Int<D_ROPE / 2>>{},
    Step<_1, _2>{}), Shape<_1, _1>{}));

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H>, Int<B_TOPK>>{},
    Step<_1, _2>{}), Shape<_1, _1>{}));

static_assert(cosize_v<SmemLayoutKNoPE> * sizeof(bf16) == BF16_TILE_BYTES);
static_assert(cosize_v<SmemLayoutKRoPE> * sizeof(bf16) == ROPE_TILE_BYTES);

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<
        bf16, bf16, float, B_H, B_TOPK * 2,
        UMMA::Major::K, UMMA::Major::K>{}));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<
        bf16, bf16, float, B_H, 256,
        UMMA::Major::K, UMMA::Major::MN>{}));

// The raw buffers deliberately use a plain row-major layout.  They are only a
// TMA destination and a dequant source; the BF16 destination carries the UMMA
// SW128 layout.
template <bool PAIR_PIPELINE>
struct KStorage {
    array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> k_rope;
    array_aligned<std::uint8_t, B_TOPK * RAW_ROW_BYTES>
        raw_nope[NUM_BUFS];
    array_aligned<bf16, cosize_v<SmemLayoutKNoPE>> k_nope[NUM_BUFS];
};

// q_nope begins after K's RoPE and both raw buffers.  Consequently Q TMA may
// overlap raw0/raw1 TMA, while BF16 slot 0 waits for Q UTCCP before overwriting
// this region.
template <bool PAIR_PIPELINE>
struct QStorage {
    array_aligned<
        std::uint8_t,
        ROPE_TILE_BYTES + NUM_BUFS * RAW_TILE_BYTES> pad;
    array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope;
};

static_assert(offsetof(KStorage<false>, raw_nope) == ROPE_TILE_BYTES);
static_assert(offsetof(KStorage<false>, k_nope) ==
              ROPE_TILE_BYTES + NUM_BUFS * RAW_TILE_BYTES);
static_assert(offsetof(QStorage<false>, q_nope) ==
              ROPE_TILE_BYTES + NUM_BUFS * RAW_TILE_BYTES);
static_assert(offsetof(KStorage<true>, k_nope) ==
              ROPE_TILE_BYTES + NUM_BUFS * RAW_TILE_BYTES);
static_assert(offsetof(QStorage<true>, q_nope) ==
              ROPE_TILE_BYTES + NUM_BUFS * RAW_TILE_BYTES);
static_assert(offsetof(KStorage<false>, raw_nope) % 128 == 0);
static_assert(offsetof(KStorage<false>, k_nope) % 128 == 0);
static_assert(offsetof(QStorage<false>, q_nope) % 128 == 0);
static_assert(offsetof(KStorage<true>, k_nope) % 128 == 0);
static_assert(offsetof(QStorage<true>, q_nope) % 128 == 0);

template <bool PAIR_PIPELINE>
struct SharedMemoryPlan;

template <>
struct SharedMemoryPlan<false> {
    union {
        KStorage<false> k;
        QStorage<false> q;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } u;

    // Kept separate from S so P(k) exchange can overlap S@V(k-1).
    float p_exchange_buf[4][32 * (B_TOPK / 2)];

    union {
        array_aligned<bf16, cosize_v<SmemLayoutS>> s;
        array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;
    } sq;

    // One exact FP32 scale stage.  Warp 9 keeps the next tile in registers.
    CUTE_ALIGNAS(16) float scales[B_TOPK][V32_NUM_SCALES];
    CUTE_ALIGNAS(8) std::uint8_t is_k_valid[NUM_BUFS][B_TOPK / 8];
    // Physical TMA row coordinates are produced once and shared by the raw
    // and RoPE producer warps.  Keeping this ring independent from the exact
    // FP32 scale stage lets TMA issue run ahead while WG1 consumes scales.
    CUTE_ALIGNAS(16) int tma_coord[NUM_BUFS][B_TOPK];

    array_aligned<std::uint32_t, 1> tmem_start_addr;
    float rowwise_scratch[128];

    transac_bar_t bar_q_nope_tma;
    transac_bar_t bar_q_rope_tma;
    transac_bar_t bar_q_nope_utccp;
    transac_bar_t bar_q_rope_utccp;

    transac_bar_t bar_raw_ready[NUM_BUFS];
    transac_bar_t bar_raw_free[NUM_BUFS];
    transac_bar_t bar_scale_ready;
    transac_bar_t bar_scale_free;
    transac_bar_t bar_valid_ready[NUM_BUFS];
    transac_bar_t bar_valid_free[NUM_BUFS];
    transac_bar_t bar_coord_ready[NUM_BUFS];
    // Retain the audited 231,680-byte layout.  Coordinate-slot reuse itself
    // uses named barriers below so racecheck can model the release/acquire edge
    // between the two reader warps and the metadata writer warp.
    std::uint64_t coord_reuse_padding[NUM_BUFS];
    transac_bar_t bar_bf16_ready[NUM_BUFS];

    transac_bar_t bar_qk_done[NUM_BUFS];
    transac_bar_t bar_sv_done[NUM_BUFS];
    transac_bar_t bar_k_rope_ready;
    transac_bar_t bar_qk_rope_done;
    transac_bar_t bar_p_free;
    transac_bar_t bar_so_ready;
};

// v8 reuses an already-consumed p_exchange_buf row for the rowwise max/LSE
// exchange.  This removes the dedicated 512-byte scratch array and lets each
// xor-64 row pair rendezvous independently instead of serializing all of WG0.
template <>
struct SharedMemoryPlan<true> {
    union {
        KStorage<true> k;
        QStorage<true> q;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } u;

    float p_exchange_buf[4][32 * (B_TOPK / 2)];

    union {
        array_aligned<bf16, cosize_v<SmemLayoutS>> s;
        array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;
    } sq;

    CUTE_ALIGNAS(16) float scales[B_TOPK][V32_NUM_SCALES];
    CUTE_ALIGNAS(8) std::uint8_t is_k_valid[NUM_BUFS][B_TOPK / 8];
    CUTE_ALIGNAS(16) int tma_coord[NUM_BUFS][B_TOPK];

    array_aligned<std::uint32_t, 1> tmem_start_addr;

    transac_bar_t bar_q_nope_tma;
    transac_bar_t bar_q_rope_tma;
    transac_bar_t bar_q_nope_utccp;
    transac_bar_t bar_q_rope_utccp;

    transac_bar_t bar_raw_ready[NUM_BUFS];
    transac_bar_t bar_raw_free[NUM_BUFS];
    transac_bar_t bar_scale_ready;
    transac_bar_t bar_scale_free;
    transac_bar_t bar_valid_ready[NUM_BUFS];
    transac_bar_t bar_valid_free[NUM_BUFS];
    transac_bar_t bar_coord_ready[NUM_BUFS];
    std::uint64_t coord_reuse_padding[NUM_BUFS];
    transac_bar_t bar_bf16_ready[NUM_BUFS];

    transac_bar_t bar_qk_done[NUM_BUFS];
    transac_bar_t bar_sv_done[NUM_BUFS];
    transac_bar_t bar_k_rope_ready;
    transac_bar_t bar_qk_rope_done;
    transac_bar_t bar_p_free;
    transac_bar_t bar_so_ready;
};

// SM100 B200/B300 exposes 227 KiB of dynamic SMEM to this launch shape.  Pin the audited
// 231,680-byte layout as well as the architectural upper bound so any member,
// alignment, or barrier-type drift fails at compile time.
static constexpr std::size_t EXPECTED_SMEM_BYTES = 231680;
static_assert(sizeof(SharedMemoryPlan<false>) == EXPECTED_SMEM_BYTES);
static_assert(sizeof(SharedMemoryPlan<true>) ==
              EXPECTED_SMEM_BYTES - 128 * sizeof(float));
static_assert(sizeof(SharedMemoryPlan<false>) < 227 * 1024);
static_assert(sizeof(SharedMemoryPlan<true>) < 227 * 1024);

enum NamedBarriers : int {
    WG0_SYNC = 0,
    WG0_WARP02_SYNC = 1,
    WG0_WARP13_SYNC = 2,
    COORD_FREE_SLOT0 = 3,
    COORD_FREE_SLOT1 = 4,
};

template <
    typename ShapeQNoPE, typename TMAQNoPE,
    typename ShapeQRoPE, typename TMAQRoPE,
    typename ShapeO, typename TMAO>
struct TmaParams {
    ShapeQNoPE shape_q_nope;
    TMAQNoPE tma_q_nope;
    ShapeQRoPE shape_q_rope;
    TMAQRoPE tma_q_rope;
    ShapeO shape_o;
    TMAO tma_o;
    CUtensorMap tensor_map_raw_nope;
    CUtensorMap tensor_map_rope;
};

// -----------------------------------------------------------------------------
// V3.2 addressing and exact arbitrary-FP32 dequant helpers
// -----------------------------------------------------------------------------

CUTE_DEVICE bool token_is_valid(
    int absolute_topk_pos,
    int token_idx,
    int topk_length,
    int s_kv) {
    return absolute_topk_pos < topk_length && token_idx >= 0 && token_idx < s_kv;
}

CUTE_DEVICE int token_tma_coord(
    SparseAttnFwdParams const& params,
    int token_idx) {
    const int physical_rows_per_block =
        params.stride_kv_block / V32_TOKEN_BYTES;
    // The benchmark and normal paged-cache layout have no padding between
    // pages.  Avoid an expensive runtime div/mod in that overwhelmingly common
    // case; the general padded-page mapping remains below.
    if (physical_rows_per_block == params.page_block_size) {
        return token_idx;
    }
    const unsigned block = static_cast<unsigned>(token_idx) /
                           static_cast<unsigned>(params.page_block_size);
    const unsigned row = static_cast<unsigned>(token_idx) %
                         static_cast<unsigned>(params.page_block_size);
    return static_cast<int>(
        static_cast<std::int64_t>(block) *
            physical_rows_per_block + row);
}

CUTE_DEVICE const std::uint8_t* physical_row_ptr(
    SparseAttnFwdParams const& params,
    int tma_coord) {
    return reinterpret_cast<const std::uint8_t*>(params.kv) +
           static_cast<std::int64_t>(tma_coord) * V32_TOKEN_BYTES;
}

CUTE_DEVICE nv_bfloat162 fp8x2_to_bf16x2_f32_scale(
    __nv_fp8x2_e4m3 value,
    float scale) {
    float2 x = static_cast<float2>(value);
    x = ku::float2_mul(x, float2{scale, scale});
    return __float22bfloat162_rn(x);
}

CUTE_DEVICE float select_scale(float4 const& scales, int group) {
    if (group == 0) return scales.x;
    if (group == 1) return scales.y;
    if (group == 2) return scales.z;
    return scales.w;
}

struct MetaRegs {
    float4 scale0;
    float4 scale1;
    unsigned valid_pair;
    int coord0;
    int coord1;
};

template <bool PAIR_PIPELINE>
CUTE_DEVICE MetaRegs load_meta_regs(
    SparseAttnFwdParams const& params,
    int const* g_indices,
    int tile,
    int lane,
    int topk_length) {
    const int pos0 = tile * B_TOPK + lane * 2;
    const int pos1 = pos0 + 1;
    const int2 indices = __ldg(reinterpret_cast<int2 const*>(g_indices + pos0));
    const int idx0 = indices.x;
    const int idx1 = indices.y;
    const bool valid0 = token_is_valid(pos0, idx0, topk_length, params.s_kv);
    const bool valid1 = token_is_valid(pos1, idx1, topk_length, params.s_kv);

    MetaRegs regs{};
    regs.coord0 = -1;
    regs.coord1 = -1;
    if (valid0) {
        regs.coord0 = token_tma_coord(params, idx0);
        regs.scale0 = __ldg(reinterpret_cast<const float4*>(
            physical_row_ptr(params, regs.coord0) + V32_SCALE_OFFSET));
    }
    if (valid1) {
        regs.coord1 = token_tma_coord(params, idx1);
        regs.scale1 = __ldg(reinterpret_cast<const float4*>(
            physical_row_ptr(params, regs.coord1) + V32_SCALE_OFFSET));
    }
    regs.valid_pair = static_cast<unsigned>(valid0) |
                      (static_cast<unsigned>(valid1) << 1);
    return regs;
}

template <bool TRACE_ENABLED, bool DUAL_DQ, typename Plan>
CUTE_DEVICE void publish_tma_coords(
    Plan& plan,
    V8TraceParams const& trace,
    int tile,
    int lane,
    int coord0,
    int coord1) {
    const int slot = tile & 1;
    constexpr int meta_track = DUAL_DQ
        ? TRACE_PRODUCER_META
        : TRACE_WG2_META;

    if (lane == 0) {
        trace_stamp<TRACE_ENABLED>(trace, meta_track, tile, 0);
    }
    if (tile >= NUM_BUFS) {
        // Raw and RoPE each contribute one full warp with non-blocking
        // arrivals after their coordinate reads.  The metadata warp is the
        // third participant and waits before overwriting this ring slot.
        NamedBarrier::arrive_and_wait(
            96, NamedBarriers::COORD_FREE_SLOT0 + slot);
    }
    if (lane == 0) {
        trace_stamp<TRACE_ENABLED>(trace, meta_track, tile, 1);
    }

    *reinterpret_cast<int2*>(plan.tma_coord[slot] + lane * 2) =
        int2{coord0, coord1};
    fence_view_async_shared();
    plan.bar_coord_ready[slot].arrive();
    if (lane == 0) {
        trace_stamp<TRACE_ENABLED>(trace, meta_track, tile, 2);
    }
}

template <
    bool TRACE_ENABLED, bool DUAL_DQ,
    typename Plan, typename TiledMMA, typename TensorO>
CUTE_DEVICE void issue_sv_tile(
    Plan& plan,
    V8TraceParams const& trace,
    TiledMMA& tiled_mma_o,
    TensorO tO,
    int prev) {
    const int slot = prev & 1;
    constexpr int mma_track = DUAL_DQ
        ? TRACE_PRODUCER_MMA
        : TRACE_WG2_MMA;
    Tensor sS = make_tensor(
        make_smem_ptr(plan.sq.s.data()), SmemLayoutS{});
    Tensor sV = make_tensor(
        make_smem_ptr(plan.u.k.k_nope[slot].data()), SmemLayoutV{});

    trace_stamp<TRACE_ENABLED>(trace, mma_track, prev, 6);
    plan.bar_so_ready.wait(prev & 1);
    ku::tcgen05_after_thread_sync();
    trace_stamp<TRACE_ENABLED>(trace, mma_track, prev, 7);
    ku::utcmma_ss(tiled_mma_o, sS, sV, tO, prev == 0);
    ku::umma_arrive_noelect(plan.bar_sv_done[slot]);
    trace_stamp<TRACE_ENABLED>(trace, mma_track, prev, 8);
}

template <
    bool TRACE_ENABLED, bool DUAL_DQ, bool PAIR_PIPELINE,
    typename Plan, typename TMAParams>
CUTE_DEVICE void issue_raw_tile(
    Plan& plan,
    V8TraceParams const& trace,
    TMAParams const& tma_params,
    int tile,
    int num_k_blocks) {
    const int slot = tile & 1;
    const int phase = (tile >> 1) & 1;
    constexpr int raw_track = DUAL_DQ
        ? TRACE_PRODUCER_RAW
        : TRACE_WG2_RAW;
    constexpr auto raw_cache_hint = TMA::CacheHintSm90::EVICT_LAST;
    if (elect_one_sync()) {
        trace_stamp<TRACE_ENABLED>(trace, raw_track, tile, 0);
        plan.bar_raw_free[slot].wait(phase ^ 1);
        trace_stamp<TRACE_ENABLED>(trace, raw_track, tile, 1);
        plan.bar_coord_ready[slot].wait(phase);
        ku::tcgen05_after_thread_sync();
        trace_stamp<TRACE_ENABLED>(trace, raw_track, tile, 2);

        plan.bar_raw_ready[slot].arrive_and_expect_tx(RAW_TILE_BYTES);
        fence_view_async_shared();
        int4 coords = *reinterpret_cast<int4 const*>(plan.tma_coord[slot]);
        CUTE_UNROLL
        for (int row = 0; row < B_TOPK; row += 4) {
            if (row + 4 < B_TOPK) {
                const int4 next = *reinterpret_cast<int4 const*>(
                    plan.tma_coord[slot] + row + 4);
                ku::tma_gather4(
                    &tma_params.tensor_map_raw_nope,
                    plan.bar_raw_ready[slot],
                    plan.u.k.raw_nope[slot].data() +
                        row * RAW_ROW_BYTES,
                    0,
                    coords,
                    static_cast<std::int64_t>(raw_cache_hint));
                coords = next;
            } else {
                ku::tma_gather4(
                    &tma_params.tensor_map_raw_nope,
                    plan.bar_raw_ready[slot],
                    plan.u.k.raw_nope[slot].data() +
                        row * RAW_ROW_BYTES,
                    0,
                    coords,
                    static_cast<std::int64_t>(raw_cache_hint));
            }
        }
        trace_stamp<TRACE_ENABLED>(trace, raw_track, tile, 3);
    }
    __syncwarp();
    if (tile + NUM_BUFS < num_k_blocks) {
        NamedBarrier::arrive(
            96, NamedBarriers::COORD_FREE_SLOT0 + slot);
    }
}

// -----------------------------------------------------------------------------
// Kernel
// -----------------------------------------------------------------------------

template <bool TRACE_ENABLED, bool DUAL_DQ, bool PAIR_PIPELINE, typename TMAParams>
__global__ void __launch_bounds__(
    DUAL_DQ ? V8_NUM_THREADS : SINGLE_DQ_NUM_THREADS, 1, 1)
v8_kernel(
    __grid_constant__ const SparseAttnFwdParams params,
    __grid_constant__ const TMAParams tma_params,
    const int output_h_q,
    __grid_constant__ const V8TraceParams trace) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || \
    defined(__CLION_IDE__) || defined(__VSCODE_IDE__)
    const int s_q_idx = blockIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x & 31;
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x & 127;
    constexpr int producer_warp_base = DUAL_DQ ? 12 : 8;
    constexpr int producer_mma_track = DUAL_DQ
        ? TRACE_PRODUCER_MMA
        : TRACE_WG2_MMA;
    constexpr int producer_meta_track = DUAL_DQ
        ? TRACE_PRODUCER_META
        : TRACE_WG2_META;
    constexpr int producer_rope_track = DUAL_DQ
        ? TRACE_PRODUCER_ROPE
        : TRACE_WG2_ROPE;

    const int requested_topk = params.topk_length != nullptr
        ? __ldg(params.topk_length + s_q_idx)
        : params.topk;
    const int topk_length = max(0, min(requested_topk, params.topk));
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);

    extern __shared__ char dynamic_smem[];
    using Plan = SharedMemoryPlan<PAIR_PIPELINE>;
    Plan& plan = *reinterpret_cast<Plan*>(dynamic_smem);
    int const* g_indices = params.indices +
                           s_q_idx * params.stride_indices_s_q;

    TiledMMA_P tiled_mma_p{};
    TiledMMA_O tiled_mma_o{};

    Tensor tP = partition_fragment_C(
        tiled_mma_p, Shape<Int<B_H>, Int<B_TOPK * 2>>{});
    Tensor tQ_nope_part0 = tiled_mma_p.get_slice(_0{}).make_fragment_A(
        partition_shape_A(
            tiled_mma_p, Shape<Int<B_H>, Int<D_NOPE / 4>>{}));
    Tensor tQ_nope_part1 = tiled_mma_p.get_slice(_0{}).make_fragment_A(
        partition_shape_A(
            tiled_mma_p, Shape<Int<B_H>, Int<D_NOPE / 4>>{}));
    Tensor tQ_rope = tiled_mma_p.get_slice(_0{}).make_fragment_A(
        partition_shape_A(
            tiled_mma_p, Shape<Int<B_H>, Int<D_ROPE / 2>>{}));
    Tensor tO = partition_fragment_C(tiled_mma_o, Shape<Int<B_H>, Int<D_V>>{});

    tP.data().get() = tmem_cols::P;
    tQ_nope_part0.data().get() = tmem_cols::Q;
    tQ_nope_part1.data().get() = tmem_cols::Q + 64;
    tQ_rope.data().get() = tmem_cols::Q_ROPE;
    tO.data().get() = tmem_cols::O;

    // One warp initializes all barriers, launches the Q TMAs, and allocates TMEM.
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            cute::prefetch_tma_descriptor(
                tma_params.tma_q_nope.get_tma_descriptor());
            cute::prefetch_tma_descriptor(
                tma_params.tma_q_rope.get_tma_descriptor());
            cute::prefetch_tma_descriptor(
                tma_params.tma_o.get_tma_descriptor());
            cute::prefetch_tma_descriptor(
                &tma_params.tensor_map_raw_nope);
            if constexpr (PAIR_PIPELINE) {
                cute::prefetch_tma_descriptor(
                    &tma_params.tensor_map_rope);
            }

            plan.bar_q_nope_tma.init(1);
            plan.bar_q_rope_tma.init(1);
            fence_barrier_init();

            Tensor gQ_nope = tma_params.tma_q_nope
                .get_tma_tensor(tma_params.shape_q_nope)(_, _, s_q_idx);
            Tensor sQ_nope = make_tensor(
                make_smem_ptr(plan.u.q.q_nope.data()), SmemLayoutQNoPE{});
            ku::launch_tma_copy(
                tma_params.tma_q_nope, gQ_nope, sQ_nope,
                plan.bar_q_nope_tma, TMA::CacheHintSm90::EVICT_FIRST);

            Tensor gQ_rope = tma_params.tma_q_rope
                .get_tma_tensor(tma_params.shape_q_rope)(_, _, s_q_idx);
            Tensor sQ_rope = make_tensor(
                make_smem_ptr(plan.sq.q_rope.data()), SmemLayoutQRoPE{});
            ku::launch_tma_copy(
                tma_params.tma_q_rope, gQ_rope, sQ_rope,
                plan.bar_q_rope_tma, TMA::CacheHintSm90::EVICT_FIRST);

            plan.bar_q_nope_utccp.init(1);
            plan.bar_q_rope_utccp.init(1);
            plan.bar_scale_ready.init(32);
            plan.bar_scale_free.init(DUAL_DQ ? 256 : 128);
            CUTE_UNROLL
            for (int i = 0; i < NUM_BUFS; ++i) {
                plan.bar_raw_ready[i].init(1);
                plan.bar_raw_free[i].init(DUAL_DQ ? 256 : 128);
                plan.bar_valid_ready[i].init(32);
                plan.bar_valid_free[i].init(DUAL_DQ ? 384 : 256);  // WG0 + DQ WGs
                plan.bar_coord_ready[i].init(32);
                plan.bar_bf16_ready[i].init(DUAL_DQ ? 256 : 128);
                plan.bar_qk_done[i].init(1);
                plan.bar_sv_done[i].init(1);
            }
            plan.bar_k_rope_ready.init(PAIR_PIPELINE ? 1 : 32);
            plan.bar_qk_rope_done.init(1);
            plan.bar_p_free.init(128);
            plan.bar_so_ready.init(128);
            fence_barrier_init();
        }

        cute::TMEM::Allocator1Sm().allocate(
            512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // WG0: baseline P reduction, online softmax, S materialization, epilogue
    // -------------------------------------------------------------------------
    if (warpgroup_idx == 0) {
        if constexpr (DUAL_DQ) {
            cutlass::arch::warpgroup_reg_alloc<V8_WG0_REGS>();
        } else {
            cutlass::arch::warpgroup_reg_alloc<224>();
        }

        if (lane_idx == 0) {
            trace_stamp<TRACE_ENABLED>(
                trace, TRACE_WG0_WARP0 + warp_idx, trace.num_tiles, 0);
        }

        float mi = -1e30f;
        float li = 0.0f;
        float real_mi = -CUDART_INF_F;

        bf16* sS_base = plan.sq.s.data() + lane_idx * 8 +
                        (warp_idx & 1) * (B_H / 2) * 8 +
                        (warp_idx / 2) * B_H * (B_TOPK / 2);
        static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            const int slot = k & 1;
            const int phase = (k >> 1) & 1;

            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(
                    trace, TRACE_WG0_WARP0 + warp_idx, k, 0);
            }

            NamedBarrier::arrive_and_wait(
                64, NamedBarriers::WG0_WARP02_SYNC + (warp_idx & 1));
            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(
                    trace, TRACE_WG0_WARP0 + warp_idx, k, 5);
            }
            plan.bar_qk_done[slot].wait(phase);
            plan.bar_valid_ready[slot].wait(phase);
            ku::tcgen05_after_thread_sync();
            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(
                    trace, TRACE_WG0_WARP0 + warp_idx, k, 1);
            }

            float p[NUM_ELEMS_PER_THREAD];
            retrieve_mask_and_reduce_p<
                NUM_ELEMS_PER_THREAD,
                tmem_cols::P,
                NamedBarriers::WG0_WARP02_SYNC,
                NamedBarriers::WG0_WARP13_SYNC,
                false,
                PAIR_PIPELINE>(
                    reinterpret_cast<char*>(plan.is_k_valid[slot]),
                    warp_idx,
                    lane_idx,
                    [&]() { plan.bar_p_free.arrive(); },
                    plan.p_exchange_buf,
                    p);
            plan.bar_valid_free[slot].arrive();

            float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
            cur_pi_max *= params.sm_scale_div_log2;

            if constexpr (PAIR_PIPELINE) {
                // Reuse one word from this lane's already-consumed first
                // float4.  The lane/8 skew makes the 32 words bank-unique;
                // unlike a contiguous [lane] scratch, no lane can overwrite
                // a word that another lane may still be loading.
                const int pair_scratch_col = lane_idx * 4 + lane_idx / 8;
                plan.p_exchange_buf[warp_idx][pair_scratch_col] = cur_pi_max;
                NamedBarrier::arrive_and_wait(
                    64,
                    NamedBarriers::WG0_WARP02_SYNC + (warp_idx & 1));
                cur_pi_max = max(
                    cur_pi_max,
                    plan.p_exchange_buf[warp_idx ^ 2][pair_scratch_col]);
            } else {
                plan.rowwise_scratch[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(128, NamedBarriers::WG0_SYNC);
                cur_pi_max = max(
                    cur_pi_max,
                    plan.rowwise_scratch[idx_in_warpgroup ^ 64]);
            }
            real_mi = max(real_mi, cur_pi_max);

            const bool should_scale_o =
                __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);
            float new_max;
            float scale_for_old;
            if (!should_scale_o) {
                new_max = mi;
                scale_for_old = 1.0f;
            } else {
                new_max = max(cur_pi_max, mi);
                scale_for_old = exp2f(mi - new_max);
            }
            mi = new_max;

            nv_bfloat162 s[NUM_ELEMS_PER_THREAD / 2];
            const float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>(
                s, p, params.sm_scale_div_log2, new_max);
            li = fma(li, scale_for_old, cur_sum);
            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(
                    trace, TRACE_WG0_WARP0 + warp_idx, k, 2);
            }

            if (k > 0) {
                const int prev = k - 1;
                plan.bar_sv_done[prev & 1].wait((prev >> 1) & 1);
            }
            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(
                    trace, TRACE_WG0_WARP0 + warp_idx, k, 3);
            }
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD / 8; ++i) {
                *reinterpret_cast<uint128_t*>(sS_base + B_H * 8 * i) =
                    *reinterpret_cast<uint128_t*>(s + i * 4);
            }

            if (k > 0 && should_scale_o) {
                ku::tcgen05_after_thread_sync();
                rescale_O<D_V, 32, tmem_cols::O>(scale_for_old);
                ku::tcgen05_before_thread_sync();
            }

            fence_view_async_shared();
            plan.bar_so_ready.arrive();
            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(
                    trace, TRACE_WG0_WARP0 + warp_idx, k, 4);
            }
        }

        if (lane_idx == 0) {
            trace_stamp<TRACE_ENABLED>(
                trace, TRACE_WG0_WARP0 + warp_idx, trace.num_tiles, 1);
        }

        if (real_mi == -CUDART_INF_F) {
            li = 0.0f;
            mi = -CUDART_INF_F;
        }

        // The last max-exchange barrier precedes the peer load.  Rendezvous
        // once more so no fast thread can overwrite its scratch entry with li
        // while its peer is still reading the final tile's maximum.
        if constexpr (PAIR_PIPELINE) {
            NamedBarrier::arrive_and_wait(
                64,
                NamedBarriers::WG0_WARP02_SYNC + (warp_idx & 1));
            plan.p_exchange_buf[warp_idx][lane_idx] = li;
            NamedBarrier::arrive_and_wait(
                64,
                NamedBarriers::WG0_WARP02_SYNC + (warp_idx & 1));
            li += plan.p_exchange_buf[warp_idx ^ 2][lane_idx];
        } else {
            NamedBarrier::arrive_and_wait(128, NamedBarriers::WG0_SYNC);
            plan.rowwise_scratch[idx_in_warpgroup] = li;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::WG0_SYNC);
            li += plan.rowwise_scratch[idx_in_warpgroup ^ 64];
        }

        if (idx_in_warpgroup < B_H) {
            const int global_idx = s_q_idx * output_h_q + idx_in_warpgroup;
            float current_lse = fmaf(mi, CUDART_LN2_F, logf(li));
            current_lse = current_lse == -CUDART_INF_F
                ? CUDART_INF_F
                : current_lse;
            params.max_logits[global_idx] = real_mi * CUDART_LN2_F;
            params.lse[global_idx] = current_lse;
        }

        const int last = num_k_blocks - 1;
        if (lane_idx == 0) {
            trace_stamp<TRACE_ENABLED>(
                trace, TRACE_WG0_WARP0 + warp_idx, trace.num_tiles, 4);
        }
        plan.bar_sv_done[last & 1].wait((last >> 1) & 1);
        ku::tcgen05_after_thread_sync();
        if (lane_idx == 0) {
            trace_stamp<TRACE_ENABLED>(
                trace, TRACE_WG0_WARP0 + warp_idx, trace.num_tiles, 2);
        }

        const float attn_sink = params.attn_sink == nullptr
            ? -CUDART_INF_F
            : __ldg(params.attn_sink + (idx_in_warpgroup % B_H)) * CUDART_L2E_F;
        float output_scale = __fdividef(1.0f, li + exp2f(attn_sink - mi));

        Tensor sO = make_tensor(make_smem_ptr(plan.u.o.data()), SmemLayoutO{});
        static constexpr int B_EPI = 64;
        Tensor tma_gO = flat_divide(
            tma_params.tma_o.get_tma_tensor(tma_params.shape_o)(_, _, s_q_idx),
            Shape<Int<B_H>, Int<B_EPI>>{})(_, _, _0{}, _);
        Tensor sO_divided = flat_divide(
            sO, Shape<Int<B_H>, Int<B_EPI>>{})(_, _, _0{}, _);
        auto thr_tma = tma_params.tma_o.get_slice(_0{});

        float2 o[B_EPI / 2];
        const bool have_valid_indices = __any_sync(0xffffffff, li != 0.0f);
        if (!have_valid_indices) {
            CUTE_UNROLL
            for (int i = 0; i < B_EPI / 2; ++i) {
                o[i] = float2{0.0f, 0.0f};
            }
            output_scale = 1.0f;
        }
        const float2 output_scale2{output_scale, output_scale};

        bf16* sO_addrs[B_EPI / 8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI / 8; ++i) {
            sO_addrs[i] = &sO(idx_in_warpgroup % B_H, i * 8);
        }

        CUTE_UNROLL
        for (int c = 0; c < 2; ++c) {
            CUTE_UNROLL
            for (int chunk = 0; chunk < (D_V / 4) / B_EPI; ++chunk) {
                if (have_valid_indices) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(
                        tmem_cols::O + c * 128 + chunk * B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                }

                CUTE_UNROLL
                for (int i = 0; i < B_EPI / 8; ++i) {
                    nv_bfloat162 o_bf16[4];
                    CUTE_UNROLL
                    for (int j = 0; j < 4; ++j) {
                        o[i * 4 + j] = ku::float2_mul(o[i * 4 + j], output_scale2);
                        o_bf16[j] = __float22bfloat162_rn(o[i * 4 + j]);
                    }
                    *reinterpret_cast<uint128_t*>(
                        sO_addrs[i] +
                        (c * (D_V / 2) +
                         (idx_in_warpgroup / 64) * (D_V / 4) +
                         chunk * B_EPI) * 64) =
                        *reinterpret_cast<uint128_t*>(o_bf16);
                }

                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::WG0_SYNC);

                if (warp_idx == 0 && elect_one_sync()) {
                    const int epi_chunk = c * (D_V / 2 / B_EPI) + chunk;
                    cute::copy(
                        tma_params.tma_o,
                        thr_tma.partition_S(sO_divided(_, _, epi_chunk)),
                        thr_tma.partition_D(tma_gO(_, _, epi_chunk)));
                }
                if (warp_idx == 1 && elect_one_sync()) {
                    const int epi_chunk = c * (D_V / 2 / B_EPI) +
                                          (D_V / B_EPI / 4) + chunk;
                    cute::copy(
                        tma_params.tma_o,
                        thr_tma.partition_S(sO_divided(_, _, epi_chunk)),
                        thr_tma.partition_D(tma_gO(_, _, epi_chunk)));
                }
            }
        }

        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
        if (lane_idx == 0) {
            trace_stamp<TRACE_ENABLED>(
                trace, TRACE_WG0_WARP0 + warp_idx, trace.num_tiles, 3);
        }

    // -------------------------------------------------------------------------
    // WG1/WG2 cooperate on the same tile: each 8-thread subgroup owns two
    // rows, with disjoint 0..31 and 32..63 destinations.  Raw/scale/valid
    // storage remains shared.
    // -------------------------------------------------------------------------
    } else if (warpgroup_idx == 1 || (DUAL_DQ && warpgroup_idx == 2)) {
        if constexpr (DUAL_DQ) {
            if constexpr (V8_DQ_REGS < 128) {
                cutlass::arch::warpgroup_reg_dealloc<V8_DQ_REGS>();
            } else if constexpr (V8_DQ_REGS > 128) {
                cutlass::arch::warpgroup_reg_alloc<V8_DQ_REGS>();
            }
        } else {
            cutlass::arch::warpgroup_reg_alloc<208>();
        }

        if (lane_idx == 0) {
            trace_stamp<TRACE_ENABLED>(trace, warp_idx, trace.num_tiles, 0);
        }

        static constexpr int GROUP_SIZE = 8;
        static constexpr int NUM_GROUPS = 128 / GROUP_SIZE;
        static constexpr int ROWS_PER_GROUP =
            DUAL_DQ ? B_TOPK / (2 * NUM_GROUPS) : B_TOPK / NUM_GROUPS;
        static constexpr int COL_CHUNKS = D_NOPE / (GROUP_SIZE * 8);
        static_assert(!DUAL_DQ || ROWS_PER_GROUP * NUM_GROUPS == B_TOPK / 2);

        const int group_idx = idx_in_warpgroup / GROUP_SIZE;
        const int idx_in_group = idx_in_warpgroup % GROUP_SIZE;
        const int row_base = DUAL_DQ ? (warpgroup_idx - 1) * (B_TOPK / 2) : 0;

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            const int slot = k & 1;
            const int phase = (k >> 1) & 1;

            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(trace, warp_idx, k, 0);
            }
            plan.bar_raw_ready[slot].wait(phase);
            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(trace, warp_idx, k, 1);
            }
            plan.bar_scale_ready.wait(k & 1);
            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(trace, warp_idx, k, 2);
            }
            plan.bar_valid_ready[slot].wait(phase);
            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(trace, warp_idx, k, 3);
            }
            plan.bar_sv_done[slot].wait(phase ^ 1);
            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(trace, warp_idx, k, 4);
            }
            if (k == 0) {
                plan.bar_q_nope_utccp.wait(0);
            }
            ku::tcgen05_after_thread_sync();
            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(trace, warp_idx, k, 5);
            }

            Tensor sK = make_tensor(
                make_smem_ptr(plan.u.k.k_nope[slot].data()),
                SmemLayoutKNoPE{});
            bf16* dst_base = &sK(row_base + group_idx, idx_in_group * 8);
            const std::uint32_t dst_base_addr =
                cute::cast_smem_ptr_to_uint(dst_base);

            std::uint8_t const* raw_base =
                plan.u.k.raw_nope[slot].data() +
                (row_base + group_idx) * RAW_ROW_BYTES +
                idx_in_group * 8;

            CUTE_UNROLL
            for (int local_row = 0; local_row < ROWS_PER_GROUP; ++local_row) {
                const int row =
                    row_base + local_row * NUM_GROUPS + group_idx;

                float4 row_scales{};
                int row_valid = 0;
                if (idx_in_group == 0) {
                    row_scales = *reinterpret_cast<float4 const*>(
                        plan.scales[row]);
                    row_valid =
                        (plan.is_k_valid[slot][row >> 3] >> (row & 7)) & 1;
                }
                const int subgroup_leader = (lane_idx / GROUP_SIZE) * GROUP_SIZE;
                row_scales.x = __shfl_sync(0xffffffff, row_scales.x, subgroup_leader);
                row_scales.y = __shfl_sync(0xffffffff, row_scales.y, subgroup_leader);
                row_scales.z = __shfl_sync(0xffffffff, row_scales.z, subgroup_leader);
                row_scales.w = __shfl_sync(0xffffffff, row_scales.w, subgroup_leader);
                row_valid = __shfl_sync(0xffffffff, row_valid, subgroup_leader);

                std::uint64_t next_raw8 = 0;
                if (row_valid) {
                    next_raw8 = *reinterpret_cast<std::uint64_t const*>(
                        raw_base +
                        local_row * NUM_GROUPS * RAW_ROW_BYTES);
                }
                CUTE_UNROLL
                for (int local_col = 0; local_col < COL_CHUNKS; ++local_col) {
                    __int128_t payload = 0;
                    if (row_valid) {
                        const std::uint64_t raw8 = next_raw8;
                        if (local_col + 1 < COL_CHUNKS) {
                            next_raw8 = *reinterpret_cast<std::uint64_t const*>(
                                raw_base +
                                local_row * NUM_GROUPS * RAW_ROW_BYTES +
                                (local_col + 1) * GROUP_SIZE * 8);
                        }
                        ku::nve4m3x2 pairs[4];
                        *reinterpret_cast<std::uint64_t*>(pairs) = raw8;
                        const float scale = select_scale(row_scales, local_col / 2);
                        nv_bfloat162 converted[4];
                        CUTE_UNROLL
                        for (int pair = 0; pair < 4; ++pair) {
                            converted[pair] = fp8x2_to_bf16x2_f32_scale(
                                pairs[pair], scale);
                        }
                        payload = *reinterpret_cast<__int128_t*>(converted);
                    }

                    const std::uint32_t dst_addr = dst_base_addr +
                        sizeof(bf16) *
                            (local_row * NUM_GROUPS * 64 +
                             local_col * B_TOPK * 64);
                    asm volatile(
                        "st.weak.shared::cta.b128 [%0], %1;\n"
                        :
                        : "r"(dst_addr), "q"(payload)
                        : "memory");
                }
            }

            fence_view_async_shared();
            plan.bar_bf16_ready[slot].arrive();
            plan.bar_raw_free[slot].arrive();
            plan.bar_scale_free.arrive();
            plan.bar_valid_free[slot].arrive();
            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(trace, warp_idx, k, 6);
            }
        }
        if (lane_idx == 0) {
            trace_stamp<TRACE_ENABLED>(trace, warp_idx, trace.num_tiles, 1);
        }

    // -------------------------------------------------------------------------
    // Producer WG warps 12..15 retain four roles: Q/UMMA,
    // metadata/coordinates, raw TMA, and RoPE respectively.
    // -------------------------------------------------------------------------
    } else {
        cutlass::arch::warpgroup_reg_dealloc<72>();

        if (warp_idx == producer_warp_base && elect_one_sync()) {
            trace_stamp<TRACE_ENABLED>(trace, producer_mma_track, trace.num_tiles, 0);
            plan.bar_q_rope_tma.arrive_and_expect_tx(B_H * D_ROPE * sizeof(bf16));
            plan.bar_q_rope_tma.wait(0);
            ku::tcgen05_after_thread_sync();
            trace_stamp<TRACE_ENABLED>(trace, producer_mma_track, trace.num_tiles, 1);
            {
                UMMA::SmemDescriptor sQ_rope_desc =
                    UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
                        make_smem_ptr(plan.sq.q_rope.data()),
                        tile_to_shape(
                            UMMA::Layout_K_SW64_Atom<bf16>{},
                            Shape<Int<B_H * 2>, Int<32>>{})));
                CUTE_UNROLL
                for (int subtile = 0; subtile < 2; ++subtile) {
                    SM100_UTCCP_128dp256bit_1cta::copy(
                        sQ_rope_desc + (subtile * 32) / 16,
                        tmem_cols::Q_ROPE + subtile * 8);
                }
            }
            ku::umma_arrive_noelect(plan.bar_q_rope_utccp);
            trace_stamp<TRACE_ENABLED>(trace, producer_mma_track, trace.num_tiles, 2);

            plan.bar_q_nope_tma.arrive_and_expect_tx(
                B_H * D_NOPE * sizeof(bf16));
            plan.bar_q_nope_tma.wait(0);
            ku::tcgen05_after_thread_sync();
            trace_stamp<TRACE_ENABLED>(
                trace, producer_mma_track, trace.num_tiles, 3);
            {
                UMMA::SmemDescriptor sQ_nope_desc =
                    UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
                        make_smem_ptr(plan.u.q.q_nope.data()),
                        tile_to_shape(
                            UMMA::Layout_K_SW128_Atom<bf16>{},
                            Shape<Int<B_H * 2>, Int<64>>{})));
                CUTE_UNROLL
                for (int tile = 0; tile < D_NOPE / 64 / 2; ++tile) {
                    CUTE_UNROLL
                    for (int subtile = 0; subtile < 4; ++subtile) {
                        SM100_UTCCP_128dp256bit_1cta::copy(
                            sQ_nope_desc +
                                (tile * (B_H * 128 * 2) + subtile * 32) / 16,
                            tmem_cols::Q + tile * 32 + subtile * 8);
                    }
                }
            }
            ku::umma_arrive_noelect(plan.bar_q_nope_utccp);
            trace_stamp<TRACE_ENABLED>(trace, producer_mma_track, trace.num_tiles, 4);

            // q_rope and S share storage.  Do not let WG0 publish S until the
            // Q tail has reached TMEM.
            plan.bar_q_rope_utccp.wait(0);
            trace_stamp<TRACE_ENABLED>(trace, producer_mma_track, trace.num_tiles, 5);

            const int num_mma_steps = num_k_blocks + 1;
            CUTE_NO_UNROLL
            for (int step = 0; step < num_mma_steps; ++step) {
                const bool issue_early_sv =
                    step >= 2 && step < num_k_blocks;

                if (step < num_k_blocks) {
                    const int slot = step & 1;
                    const int phase = (step >> 1) & 1;

                    trace_stamp<TRACE_ENABLED>(trace, producer_mma_track, step, 0);
                    plan.bar_p_free.wait((step & 1) ^ 1);
                    ku::tcgen05_after_thread_sync();
                    trace_stamp<TRACE_ENABLED>(trace, producer_mma_track, step, 1);

                    plan.bar_k_rope_ready.wait(step & 1);
                    ku::tcgen05_after_thread_sync();
                    trace_stamp<TRACE_ENABLED>(trace, producer_mma_track, step, 2);
                    {
                        Tensor sK_rope = make_tensor(
                            make_smem_ptr(plan.u.k.k_rope.data()),
                            SmemLayoutKRoPE_TiledMMA{});
                        ku::utcmma_ts(
                            tiled_mma_p, tQ_rope, sK_rope, tP, true);
                        ku::umma_arrive_noelect(plan.bar_qk_rope_done);
                    }
                    trace_stamp<TRACE_ENABLED>(trace, producer_mma_track, step, 3);

                    // Keep startup tile 0 on the proven late path.  Only
                    // middle tiles move early, after S has been published.
                    if (issue_early_sv) {
                        issue_sv_tile<TRACE_ENABLED, DUAL_DQ>(
                            plan, trace, tiled_mma_o, tO, step - 1);
                    }

                    // Schema-3 point 9 isolates the actual BF16/Q dependency
                    // wait from the early S@V work interleaved above.
                    trace_stamp<TRACE_ENABLED>(
                        trace, producer_mma_track, step, 9);
                    if (step == 0) {
                        plan.bar_q_nope_utccp.wait(0);
                    }
                    plan.bar_bf16_ready[slot].wait(phase);
                    ku::tcgen05_after_thread_sync();
                    trace_stamp<TRACE_ENABLED>(trace, producer_mma_track, step, 4);

                    {
                        Tensor sK_nope = make_tensor(
                            make_smem_ptr(plan.u.k.k_nope[slot].data()),
                            SmemLayoutKNoPE_TiledMMA{});
                        Tensor sK_parts = flat_divide(
                            sK_nope,
                            Tile<Int<B_TOPK * 2>, Int<D_NOPE / 4>>{})(
                                _, _, _0{}, _);
                        CUTE_UNROLL
                        for (int part = 0; part < 2; ++part) {
                            ku::utcmma_ts(
                                tiled_mma_p,
                                part ? tQ_nope_part1 : tQ_nope_part0,
                                sK_parts(_, _, part),
                                tP,
                                false);
                        }
                        ku::umma_arrive_noelect(plan.bar_qk_done[slot]);
                    }
                    trace_stamp<TRACE_ENABLED>(trace, producer_mma_track, step, 5);

                }

                // The first S@V stays after QK(step=1), and the final S@V is
                // drained after the last QK.  These are the only late cases;
                // every middle tile is issued exactly once on the early path.
                const bool issue_late_sv =
                    step > 0 && (step == 1 || step == num_k_blocks);
                if (issue_late_sv) {
                    issue_sv_tile<TRACE_ENABLED, DUAL_DQ>(
                        plan, trace, tiled_mma_o, tO, step - 1);
                }
            }

        } else if (warp_idx == producer_warp_base + 1) {
            // Each lane owns two tokens.  The next tile's 2 x float4 scales stay
            // in registers while WG1 consumes the current single SMEM stage.
            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(trace, producer_meta_track, 0, 3);
            }
            MetaRegs regs = load_meta_regs<PAIR_PIPELINE>(
                params, g_indices, 0, lane_idx, topk_length);
            if (lane_idx == 0) {
                trace_stamp<TRACE_ENABLED>(trace, producer_meta_track, 0, 4);
            }
            publish_tma_coords<TRACE_ENABLED, DUAL_DQ>(
                plan, trace, 0, lane_idx, regs.coord0, regs.coord1);

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                const int slot = k & 1;
                const int phase = (k >> 1) & 1;

                // Load k+1 before the single scale-stage wait.  Two MetaRegs
                // objects stay live here, turning that otherwise idle wait
                // into coordinate/scale prefetch latency hiding.
                MetaRegs next_regs{};
                if (k + 1 < num_k_blocks) {
                    if (lane_idx == 0) {
                        trace_stamp<TRACE_ENABLED>(
                            trace, producer_meta_track, k + 1, 3);
                    }
                    next_regs = load_meta_regs<PAIR_PIPELINE>(
                        params, g_indices, k + 1, lane_idx, topk_length);
                    if (lane_idx == 0) {
                        trace_stamp<TRACE_ENABLED>(
                            trace, producer_meta_track, k + 1, 4);
                    }

                    // Coordinates have their own double-buffered lifetime.
                    // Their raw/RoPE consumers release the next slot several
                    // microseconds before the single scale stage is reusable,
                    // so publish k+1 now instead of delaying it until the next
                    // loop iteration.  This preserves the exact phase/count
                    // protocol while extending raw-TMA lookahead.
                    publish_tma_coords<TRACE_ENABLED, DUAL_DQ>(
                        plan,
                        trace,
                        k + 1,
                        lane_idx,
                        next_regs.coord0,
                        next_regs.coord1);
                }

                if (lane_idx == 0) {
                    // Point 7 separates next-tile register prefetch from the
                    // current tile's actual scale/valid stage reuse waits.
                    trace_stamp<TRACE_ENABLED>(
                        trace, producer_meta_track, k, 7);
                }

                // The next coordinates are already visible.  The remaining
                // wait is only for the current exact-FP32 scale/valid stages.
                plan.bar_scale_free.wait((k & 1) ^ 1);
                plan.bar_valid_free[slot].wait(phase ^ 1);
                if (lane_idx == 0) {
                    trace_stamp<TRACE_ENABLED>(trace, producer_meta_track, k, 5);
                }

                *reinterpret_cast<float4*>(plan.scales[lane_idx * 2]) =
                    regs.scale0;
                *reinterpret_cast<float4*>(plan.scales[lane_idx * 2 + 1]) =
                    regs.scale1;

                unsigned mask = regs.valid_pair << ((lane_idx & 3) * 2);
                mask |= __shfl_xor_sync(0xffffffff, mask, 1);
                mask |= __shfl_xor_sync(0xffffffff, mask, 2);
                if ((lane_idx & 3) == 0) {
                    plan.is_k_valid[slot][lane_idx / 4] =
                        static_cast<std::uint8_t>(mask);
                }

                fence_view_async_shared();
                plan.bar_scale_ready.arrive();
                plan.bar_valid_ready[slot].arrive();
                if (lane_idx == 0) {
                    trace_stamp<TRACE_ENABLED>(trace, producer_meta_track, k, 6);
                }

                if (k + 1 < num_k_blocks) {
                    regs = next_regs;
                }
            }

        } else if (warp_idx == producer_warp_base + 2) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                issue_raw_tile<TRACE_ENABLED, DUAL_DQ, PAIR_PIPELINE>(
                    plan, trace, tma_params, k, num_k_blocks);
            }

        } else if (warp_idx == producer_warp_base + 3) {
            if constexpr (PAIR_PIPELINE) {
                // Decode-style RoPE gather: 32 elected-lane TMA instructions
                // replace 16 warp-wide cp.async instructions (512 memory
                // operations) and write the native SW64 layout directly.
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    const int slot = k & 1;
                    const int phase = (k >> 1) & 1;

                    if (lane_idx == 0) {
                        trace_stamp<TRACE_ENABLED>(
                            trace, producer_rope_track, k, 0);
                    }
                    plan.bar_coord_ready[slot].wait(phase);
                    plan.bar_qk_rope_done.wait((k & 1) ^ 1);
                    ku::tcgen05_after_thread_sync();
                    if (lane_idx == 0) {
                        trace_stamp<TRACE_ENABLED>(
                            trace, producer_rope_track, k, 1);
                    }

                    if (elect_one_sync()) {
                        plan.bar_k_rope_ready.arrive_and_expect_tx(
                            ROPE_TILE_BYTES);
                        fence_view_async_shared();
                        if (lane_idx == 0) {
                            trace_stamp<TRACE_ENABLED>(
                                trace, producer_rope_track, k, 2);
                        }
                        int4 coords = *reinterpret_cast<int4 const*>(
                            plan.tma_coord[slot]);
                        CUTE_UNROLL
                        for (int row = 0; row < B_TOPK; row += 4) {
                            CUTE_UNROLL
                            for (int part = 0; part < D_ROPE / 32; ++part) {
                                ku::tma_gather4(
                                    &tma_params.tensor_map_rope,
                                    plan.bar_k_rope_ready,
                                    plan.u.k.k_rope.data() +
                                        row * 32 + part * B_TOPK * 32,
                                    part * 32,
                                    coords,
                                    static_cast<std::int64_t>(
                                        TMA::CacheHintSm90::EVICT_LAST));
                            }
                            if (row + 4 < B_TOPK) {
                                coords = *reinterpret_cast<int4 const*>(
                                    plan.tma_coord[slot] + row + 4);
                            }
                        }
                        if (lane_idx == 0) {
                            trace_stamp<TRACE_ENABLED>(
                                trace, producer_rope_track, k, 3);
                        }
                    }
                    __syncwarp();
                    if (k + NUM_BUFS < num_k_blocks) {
                        NamedBarrier::arrive(
                            96, NamedBarriers::COORD_FREE_SLOT0 + slot);
                    }
                    if (lane_idx == 0) {
                        trace_stamp<TRACE_ENABLED>(
                            trace, producer_rope_track, k, 4);
                    }
                }
            } else {
                // One warp copies all 64 RoPE rows.  Coordinate staging
                // replays the old two-warp SW64 destination mapping.
                static constexpr int GROUP_SIZE = 8;
                static constexpr int GROUPS_PER_WARP = 32 / GROUP_SIZE;
                static constexpr int VIRTUAL_GROUPS = 64 / GROUP_SIZE;
                static constexpr int GROUP_PASSES =
                    VIRTUAL_GROUPS / GROUPS_PER_WARP;
                static constexpr int ROWS_PER_GROUP =
                    B_TOPK / VIRTUAL_GROUPS;
                const int group_in_warp = lane_idx / GROUP_SIZE;
                const int idx_in_group = lane_idx % GROUP_SIZE;
                const int subgroup_leader = group_in_warp * GROUP_SIZE;

                Tensor sK_rope = make_tensor(
                    make_smem_ptr(plan.u.k.k_rope.data()),
                    SmemLayoutKRoPE{});

                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    const int slot = k & 1;
                    const int phase = (k >> 1) & 1;

                    if (lane_idx == 0) {
                        trace_stamp<TRACE_ENABLED>(
                            trace, producer_rope_track, k, 0);
                    }
                    plan.bar_coord_ready[slot].wait(phase);
                    ku::tcgen05_after_thread_sync();
                    if (lane_idx == 0) {
                        trace_stamp<TRACE_ENABLED>(
                            trace, producer_rope_track, k, 1);
                    }

                    int coords[GROUP_PASSES][ROWS_PER_GROUP];
                    CUTE_UNROLL
                    for (int pass = 0; pass < GROUP_PASSES; ++pass) {
                        const int group_idx =
                            group_in_warp + pass * GROUPS_PER_WARP;
                        CUTE_UNROLL
                        for (int local_row = 0;
                             local_row < ROWS_PER_GROUP;
                             ++local_row) {
                            const int row =
                                group_idx + local_row * VIRTUAL_GROUPS;
                            int coord = idx_in_group == 0
                                ? plan.tma_coord[slot][row]
                                : -1;
                            coords[pass][local_row] = __shfl_sync(
                                0xffffffff, coord, subgroup_leader);
                        }
                    }
                    __syncwarp();
                    if (k + NUM_BUFS < num_k_blocks) {
                        NamedBarrier::arrive(
                            96, NamedBarriers::COORD_FREE_SLOT0 + slot);
                    }
                    if (lane_idx == 0) {
                        trace_stamp<TRACE_ENABLED>(
                            trace, producer_rope_track, k, 2);
                    }

                    plan.bar_qk_rope_done.wait((k & 1) ^ 1);
                    if (lane_idx == 0) {
                        trace_stamp<TRACE_ENABLED>(
                            trace, producer_rope_track, k, 3);
                    }
                    CUTE_UNROLL
                    for (int pass = 0; pass < GROUP_PASSES; ++pass) {
                        const int group_idx =
                            group_in_warp + pass * GROUPS_PER_WARP;
                        bf16* dst_base =
                            &sK_rope(group_idx, idx_in_group * 8);
                        CUTE_UNROLL
                        for (int local_row = 0;
                             local_row < ROWS_PER_GROUP;
                             ++local_row) {
                            const int coord = coords[pass][local_row];
                            const std::uint8_t* row_ptr = physical_row_ptr(
                                params, coord >= 0 ? coord : 0);
                            const void* src = row_ptr + V32_ROPE_OFFSET +
                                              idx_in_group * 8 * sizeof(bf16);
                            ku::cp_async_cacheglobal<ku::PrefetchSize::B128>(
                                src,
                                dst_base + local_row * VIRTUAL_GROUPS * 32,
                                coord >= 0);
                        }
                    }
                    cutlass::arch::cpasync_barrier_arrive_noinc(
                        reinterpret_cast<std::uint64_t*>(
                            &plan.bar_k_rope_ready));
                    if (lane_idx == 0) {
                        trace_stamp<TRACE_ENABLED>(
                            trace, producer_rope_track, k, 4);
                    }
                }
            }
        }
    }

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("v8 requires SM100-SM119");
    }
#endif
}

// -----------------------------------------------------------------------------
// Host launcher
// -----------------------------------------------------------------------------

template <bool TRACE_ENABLED, bool DUAL_DQ, bool PAIR_PIPELINE = false>
void run_v8_fp8_prefill_impl(
    SparseAttnFwdParams const& params,
    int output_h_q,
    V8TraceParams const& trace) {
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(output_h_q == B_H || output_h_q == 2 * B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_Q);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.topk > 0 && params.topk % B_TOPK == 0);
    KU_ASSERT(params.num_blocks > 0);
    KU_ASSERT(params.page_block_size > 0);
    KU_ASSERT(params.s_kv == params.num_blocks * params.page_block_size);
    KU_ASSERT(params.stride_kv_block >= params.page_block_size * V32_TOKEN_BYTES);
    KU_ASSERT(params.stride_kv_block % V32_TOKEN_BYTES == 0);
    KU_ASSERT(reinterpret_cast<std::int64_t>(params.kv) % 16 == 0,
              "V3.2 KV base must be 16-byte aligned");
    if constexpr (TRACE_ENABLED) {
        KU_ASSERT(trace.timestamps != nullptr);
        KU_ASSERT(trace.s_q_idx >= 0 && trace.s_q_idx < params.s_q);
        KU_ASSERT(trace.num_tiles >= cute::ceil_div(params.topk, (int)B_TOPK));
    }

    const std::uint64_t physical_rows =
        static_cast<std::uint64_t>(params.num_blocks) *
        static_cast<std::uint64_t>(params.stride_kv_block / V32_TOKEN_BYTES);
    KU_ASSERT(physical_rows <= static_cast<std::uint64_t>(INT_MAX),
              "V3.2 TMA row coordinate exceeds int32");

    auto shape_q_nope = make_shape(params.h_q, D_NOPE, params.s_q);
    auto tma_q_nope = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr(reinterpret_cast<bf16*>(params.q)),
            make_layout(
                shape_q_nope,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q))),
        SmemLayoutQNoPE{});

    auto shape_q_rope = make_shape(params.h_q, D_ROPE, params.s_q);
    auto tma_q_rope = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr(reinterpret_cast<bf16*>(params.q) + D_NOPE),
            make_layout(
                shape_q_rope,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q))),
        SmemLayoutQRoPE{});

    auto shape_o = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_o = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr(reinterpret_cast<bf16*>(params.out)),
            make_layout(
                shape_o,
                make_stride(params.d_v, _1{}, output_h_q * params.d_v))),
        SmemLayoutOTiles<1>{});

    // INT64 elements keep the TMA box dimension <= 256 while moving all 512
    // FP8 bytes of one token.
    CUtensorMap tensor_map_raw_nope = ku::make_tensor_map(
        {static_cast<std::uint64_t>(RAW_ROW_BYTES / 8), physical_rows},
        {static_cast<std::uint64_t>(V32_TOKEN_BYTES)},
        {static_cast<std::uint32_t>(RAW_ROW_BYTES / 8), 1},
        params.kv,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT64,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
    CUtensorMap tensor_map_rope = ku::make_tensor_map(
        {static_cast<std::uint64_t>(D_ROPE), physical_rows},
        {static_cast<std::uint64_t>(V32_TOKEN_BYTES)},
        {static_cast<std::uint32_t>(D_ROPE / 2), 1},
        reinterpret_cast<std::uint8_t*>(params.kv) + V32_ROPE_OFFSET,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
    TmaParams<
        decltype(shape_q_nope), decltype(tma_q_nope),
        decltype(shape_q_rope), decltype(tma_q_rope),
        decltype(shape_o), decltype(tma_o)> tma_params{
            shape_q_nope, tma_q_nope,
            shape_q_rope, tma_q_rope,
            shape_o, tma_o,
            tensor_map_raw_nope,
            tensor_map_rope};

    auto kernel = &v8_kernel<
        TRACE_ENABLED, DUAL_DQ, PAIR_PIPELINE, decltype(tma_params)>;
    constexpr std::size_t smem_size =
        sizeof(SharedMemoryPlan<PAIR_PIPELINE>);
    KU_CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    constexpr int num_threads = DUAL_DQ ? V8_NUM_THREADS : SINGLE_DQ_NUM_THREADS;
    kernel<<<params.s_q, num_threads, smem_size, params.stream>>>(
        params, tma_params, output_h_q, trace);
    KU_CHECK_KERNEL_LAUNCH();
}

void run_v8_fp8_prefill(
    SparseAttnFwdParams const& params,
    int output_h_q) {
    run_v8_fp8_prefill_impl<false, true, true>(
        params, output_h_q,
        V8TraceParams{nullptr, -1, 0, TRACE_TRACKS});
}

void run_v8_fp8_prefill_trace(
    SparseAttnFwdParams const& params,
    int output_h_q,
    std::uint64_t* timestamps,
    int trace_s_q_idx,
    int trace_num_tiles) {
    run_v8_fp8_prefill_impl<true, true, true>(
        params,
        output_h_q,
        V8TraceParams{
            timestamps, trace_s_q_idx, trace_num_tiles, TRACE_TRACKS});
}

}  // namespace sm100::fwd::head64::v8
