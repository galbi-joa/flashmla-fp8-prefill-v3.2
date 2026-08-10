/*
Standalone SM100 V3.2 native-MX-fp8 sparse-prefill research kernel.

This file is intentionally separate from fwd_v32_fp8_prefill.cu.

The previous V3.2 experiment keeps Tensor Cores in bf16:

    V3.2 fp8 cache -> dequant to bf16 smem -> bf16 UMMA

This file explores the harder path:

    V3.2 fp8 cache -> transcode to Blackwell MX fp8 tiles -> MX fp8 tcgen05.mma

The hard part is not the raw e4m3 data.  The hard part is scale ownership:

    V3.2 cache:
      scale[token][dim128], 4 fp32 carrier slots per token
      current tests/quant.py values are rounded to power-of-two scales

    Blackwell MX fp8 MMA:
      scale[m_or_n_index][k_group32], packed into 512B scale-factor basic blocks

So the V3.2 cache cannot be handed directly to tcgen05.mma.kind::mxf8f6f4.
The kernel must build MX-compatible tiles:

    QK:
      Q bf16                 -> MX fp8 A tile
      K V3.2 fp8+fp32-scale  -> MX fp8 B tile
      NoPE scale mapping      : V3.2 dim128 scale -> four MX dim32 groups
      RoPE/pad scale mapping  : compute new MX scales

    S@V:
      softmax P/S fp32       -> MX fp8 A tile
      V V3.2 fp8+fp32-scale  -> MX fp8 B tile
      V scale mapping         : compute new MX scales over 32-token K groups

This is not registered in setup.py.  It is a CUDA-free-environment research
implementation: the data transformations and tcgen05 call sites are explicit,
while final swizzled shared-memory descriptors must be verified on B300.

Scheduling note
---------------
Hopper's dense-decode seesaw schedule is not copied here.  The transferable
idea is narrower: producer work must not serialize in front of Tensor-Core work.
For SM100 sparse prefill, QK only needs the K tile, while V is needed later by
S@V.  The native MX pipeline therefore keeps KProducer and VProducer separate:

    KProducer -> k_ready -> QK
    VProducer -> v_ready -> S@V

A single kv_ready barrier would make QK wait for V transcode even though QK
cannot consume V.  That is exactly the scheduling mistake this research file is
meant to avoid.
*/

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cstdint>

#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
#include <cute/numeric/numeric_types.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#endif

#include "defines.h"
#include "params.h"
#include "utils.h"

#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX) || \
    defined(FLASHMLA_ENABLE_THEORETICAL_TMEM_LOGITS_BRIDGE) || \
    defined(FLASHMLA_ENABLE_THEORETICAL_DIRECT_TMEM_SOFTMAX)
#include <kerutils/kerutils.cuh>
#endif

namespace sm100::fwd_v32_native_fp8_prefill {

using uint8 = std::uint8_t;
using fp8_e8m0 = __nv_fp8_e8m0;

static constexpr int D_QK = 576;
static constexpr int D_V = 512;
static constexpr int D_NOPE = 512;
static constexpr int D_ROPE = 64;
static constexpr int V32_SCALE_GROUP = 128;
static constexpr int V32_NUM_SCALES = 4;
static constexpr int V32_BYTES_NOPE = D_NOPE;
static constexpr int V32_BYTES_SCALE = V32_NUM_SCALES * sizeof(float);
static constexpr int V32_BYTES_ROPE = D_ROPE * sizeof(bf16);
static constexpr int V32_BYTES_PER_TOKEN = V32_BYTES_NOPE + V32_BYTES_SCALE + V32_BYTES_ROPE;

// MX fp8 scale-factor ownership on SM100 block-scaled GEMM.
// A scale factor belongs to one M/N position and one 32-wide GEMM-K group.
// A 512B basic block stores 128 M/N positions x 4 K-groups x 1 byte.
static constexpr int MX_GROUP_K = 32;
static constexpr int MX_SCALE_MN_PER_BASIC_BLOCK = 128;
static constexpr int MX_SCALE_K_GROUPS_PER_BASIC_BLOCK = 4;
static constexpr int MX_SCALE_BASIC_BLOCK_BYTES = 512;

// Native MX fp8 QK stages K in 128-wide scale-copy chunks.  D_QK=576 therefore
// needs five 128-wide chunks, with the last 64 dimensions zero-padded.
//
// Important: CUTLASS' SM100_MMA_MXF8F6F4_SS atom has logical K=32, not 128.
// CUTLASS handles a K128 mainloop tile by copying one 512B SFA/SFB scale block
// and then issuing four K32 atom MMAs with tCtSFA/SFB(_,_,k_block).  This file's
// scale staging is K128-sized, but the final descriptor layer must split the
// actual tcgen05.mma issue stream into K32 atom descriptors before enabling PTX.
//
// The B300 table discussed in this thread has 64 active heads, but the
// block-scaled mx_float8 tile shapes in Blackwell SM100 GEMMs.md start at
// 128 rows for the dense mxf8f6f4 path.  This shell therefore pads the head
// dimension to 128 and treats rows h >= params.h_q as zero.  That is explicit
// 50% garbage work for heads=64, and must be measured against the bf16 path.
static constexpr int MX_MMA_K = 128;
static constexpr int D_QK_PAD = 640;
static constexpr int ACTIVE_H_B300 = 64;
static constexpr int QK_M = 128;        // MX tile heads, padded from active heads
static constexpr int QK_N = 128;        // topk tokens per sparse tile
static constexpr int SV_M = QK_M;       // MX tile heads, padded from active heads
static constexpr int SV_N = D_V;        // value dimension
static constexpr int SV_K = 128;        // topk tokens per sparse tile
static constexpr int PIPE_STAGES = 2;
static constexpr int NUM_THREADS = 512;
static constexpr int WARP_GROUP_THREADS = 128;
static constexpr int MX_QK_K_CHUNKS = D_QK_PAD / MX_MMA_K;
static constexpr int MX_SV_N_TILE = 128;
static constexpr int MX_SV_N_CHUNKS = SV_N / MX_SV_N_TILE;
static constexpr int MX_DENSE_SF_VEC_SIZE = 32;
static constexpr int MX_SF_PER_MMA_K = MX_MMA_K / MX_DENSE_SF_VEC_SIZE;
static constexpr int MX_SCALE_STAGE_BYTES = MX_SCALE_BASIC_BLOCK_BYTES;
static constexpr uint32_t NATIVE_MX_TMEM_CAPACITY_COLS = 512;
static constexpr uint32_t NATIVE_MX_TMEM_ADDR_COLUMN_MASK = 0x3fffffffU;
static constexpr uint32_t NATIVE_MX_TMEM_ADDR_SFID_MASK = 0xc0000000U;
static constexpr int NATIVE_MX_TMEM_ADDR_SFID_SHIFT = 30;
static constexpr uint32_t NATIVE_MX_SCALE_TMEM_K32_STRIDE_COLS = 128;
static constexpr uint32_t NATIVE_MX_SCALE_TMEM_K128_SPAN_COLS =
    NATIVE_MX_SCALE_TMEM_K32_STRIDE_COLS * MX_SF_PER_MMA_K;
static_assert(NATIVE_MX_SCALE_TMEM_K128_SPAN_COLS == NATIVE_MX_TMEM_CAPACITY_COLS);
static constexpr uint32_t NATIVE_MX_SINGLE_BUFFER_SV_TMEM_COL = 0;
static constexpr uint32_t NATIVE_MX_SINGLE_BUFFER_QK_TMEM_COL = 256;
static constexpr uint32_t NATIVE_MX_SINGLE_BUFFER_SCALE_SFA_SFID = 1;
static constexpr uint32_t NATIVE_MX_SINGLE_BUFFER_SCALE_SFB_SFID = 2;

static constexpr float E4M3_MAX_FINITE = 448.0f;
// Keep the zero/near-zero group clamp aligned with tests/quant.py's
// _cast_scale_inv_to_ue8m0() path.  Native MX scale reconstruction should not
// introduce a second, looser underflow policy while comparing against V3.2.
static constexpr float MIN_MX_SCALE = 1.0e-4f;
static constexpr int MX_QK_K_GROUPS = D_QK_PAD / MX_GROUP_K;
static constexpr int MX_SV_K_GROUPS = SV_K / MX_GROUP_K;
static constexpr int V32_MX_GROUPS_PER_SCALE = V32_SCALE_GROUP / MX_GROUP_K;
static constexpr int V32_NOPE_MX_K_GROUPS = D_NOPE / MX_GROUP_K;
static constexpr int V32_ROPE_MX_K_GROUPS = D_ROPE / MX_GROUP_K;

constexpr int ceil_div_constexpr(int x, int y) {
    return (x + y - 1) / y;
}

constexpr int align_up_constexpr(int x, int alignment) {
    return ((x + alignment - 1) / alignment) * alignment;
}

constexpr int mx_scale_layout_bytes_constexpr(int total_mn, int num_k_groups) {
    return ceil_div_constexpr(total_mn, MX_SCALE_MN_PER_BASIC_BLOCK) *
           ceil_div_constexpr(num_k_groups, MX_SCALE_K_GROUPS_PER_BASIC_BLOCK) *
           MX_SCALE_BASIC_BLOCK_BYTES;
}

static constexpr int MX_Q_BYTES = QK_M * D_QK_PAD * sizeof(__nv_fp8_e4m3);
static constexpr int MX_K_BYTES = QK_N * D_QK_PAD * sizeof(__nv_fp8_e4m3);
static constexpr int MX_S_BYTES = SV_M * SV_K * sizeof(__nv_fp8_e4m3);
static constexpr int MX_V_BYTES = SV_K * SV_N * sizeof(__nv_fp8_e4m3);
static constexpr int MX_V_NCHUNK_BYTES = SV_K * MX_SV_N_TILE * sizeof(__nv_fp8_e4m3);

#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
using NativeMxQPayloadLayout = decltype(
    ku::make_umma_canonical_k_major_layout<QK_M, D_QK_PAD, 128, uint8>());
using NativeMxKPayloadLayout = decltype(
    ku::make_umma_canonical_k_major_layout<QK_N, D_QK_PAD, 128, uint8>());
using NativeMxSPayloadLayout = decltype(
    ku::make_umma_canonical_k_major_layout<SV_M, SV_K, 128, uint8>());
using NativeMxVFullPayloadLayout = decltype(
    ku::make_umma_canonical_mn_major_layout<SV_N, SV_K, 128, uint8>());
using NativeMxVCompactPayloadLayout = decltype(
    ku::make_umma_canonical_mn_major_layout<MX_SV_N_TILE, SV_K, 128, uint8>());
using NativeMxScaleStageLayout =
    typename cutlass::detail::Sm1xxBlockScaledBasicChunk<MX_GROUP_K>::SfKMajorAtom;
using NativeMxScaleUtccpOp =
    cute::SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta;
using NativeMxScaleSfaTmemFragment =
    cute::UMMA::tmem_sf_frg<cute::float_ue8m0_t, MX_GROUP_K, 1, true>;
using NativeMxScaleSfbTmemFragment =
    cute::UMMA::tmem_sf_frg<cute::float_ue8m0_t, MX_GROUP_K, 1, false>;
using NativeMxScaleTmemShape = cute::Shape<
    cute::Shape<
        cute::Shape<cute::Int<1>, cute::Int<32>, cute::Int<4>>,
        cute::Shape<cute::Int<32>, cute::Int<1>>>,
    cute::Int<1>,
    cute::Shape<cute::Int<4>, cute::Int<1>>>;

static_assert(cute::cosize_v<NativeMxQPayloadLayout> * sizeof(uint8) <= MX_Q_BYTES);
static_assert(cute::cosize_v<NativeMxKPayloadLayout> * sizeof(uint8) <= MX_K_BYTES);
static_assert(cute::cosize_v<NativeMxSPayloadLayout> * sizeof(uint8) <= MX_S_BYTES);
static_assert(cute::cosize_v<NativeMxVFullPayloadLayout> * sizeof(uint8) <= MX_V_BYTES);
static_assert(
    cute::cosize_v<NativeMxVCompactPayloadLayout> * sizeof(uint8) <=
    MX_V_NCHUNK_BYTES);
static_assert(
    cute::cosize_v<NativeMxScaleStageLayout> * sizeof(uint8) ==
    MX_SCALE_BASIC_BLOCK_BYTES);
static_assert(cute::is_same_v<
    typename cute::Copy_Traits<NativeMxScaleUtccpOp>::SRegisters,
    uint64_t[1]>);
static_assert(cute::is_same_v<
    typename cute::Copy_Traits<NativeMxScaleUtccpOp>::DRegisters,
    uint32_t[1]>);
static_assert(!cute::is_same_v<
    NativeMxScaleSfaTmemFragment,
    NativeMxScaleSfbTmemFragment>);
#endif

static constexpr int MX_Q_SCALE_LAYOUT_BYTES = mx_scale_layout_bytes_constexpr(QK_M, MX_QK_K_GROUPS);
static constexpr int MX_K_SCALE_LAYOUT_BYTES = mx_scale_layout_bytes_constexpr(QK_N, MX_QK_K_GROUPS);
static constexpr int MX_S_SCALE_LAYOUT_BYTES = mx_scale_layout_bytes_constexpr(SV_M, MX_SV_K_GROUPS);
static constexpr int MX_V_SCALE_LAYOUT_BYTES = mx_scale_layout_bytes_constexpr(SV_N, MX_SV_K_GROUPS);
static constexpr int MX_V_NCHUNK_SCALE_LAYOUT_BYTES =
    mx_scale_layout_bytes_constexpr(MX_SV_N_TILE, MX_SV_K_GROUPS);

static constexpr int MX_PERSISTENT_OPERAND_BYTES =
    MX_Q_BYTES + MX_Q_SCALE_LAYOUT_BYTES;
static constexpr int MX_STAGE_OPERAND_BYTES =
    MX_K_BYTES + MX_K_SCALE_LAYOUT_BYTES +
    MX_S_BYTES + MX_S_SCALE_LAYOUT_BYTES +
    MX_V_BYTES + MX_V_SCALE_LAYOUT_BYTES;
static constexpr int MX_TWO_STAGE_OPERAND_BYTES =
    MX_PERSISTENT_OPERAND_BYTES + PIPE_STAGES * MX_STAGE_OPERAND_BYTES;

static constexpr int MX_ROW_STATE_BYTES = 3 * SV_M * sizeof(float);
static constexpr int MX_QK_LOGITS_DEBUG_BYTES = QK_M * QK_N * sizeof(float);
static constexpr int MX_OUT_DEBUG_BYTES = SV_M * SV_N * sizeof(float);
static constexpr int MX_PERSISTENT_SCALE_STAGE_BYTES = MX_SCALE_STAGE_BYTES;
static constexpr int MX_STAGE_SCALE_STAGE_BYTES = 3 * MX_SCALE_STAGE_BYTES;
static constexpr int MX_QK_STAGE_BYTES =
    MX_K_BYTES + MX_K_SCALE_LAYOUT_BYTES + MX_SCALE_STAGE_BYTES;
static constexpr int MX_SV_NCHUNK_STAGE_BYTES =
    MX_S_BYTES + MX_S_SCALE_LAYOUT_BYTES + MX_SCALE_STAGE_BYTES +
    MX_V_NCHUNK_BYTES + MX_V_NCHUNK_SCALE_LAYOUT_BYTES + MX_SCALE_STAGE_BYTES;

static_assert(V32_BYTES_PER_TOKEN == 656);
static_assert(D_QK_PAD % MX_MMA_K == 0);
static_assert(MX_QK_K_CHUNKS == 5);
static_assert(SV_N % MX_SV_N_TILE == 0);
static_assert(MX_DENSE_SF_VEC_SIZE == MX_GROUP_K);
static_assert(MX_SF_PER_MMA_K == MX_SCALE_K_GROUPS_PER_BASIC_BLOCK);
static_assert(MX_SCALE_BASIC_BLOCK_BYTES == MX_SCALE_MN_PER_BASIC_BLOCK * MX_SCALE_K_GROUPS_PER_BASIC_BLOCK);
static_assert(QK_M % MX_SCALE_MN_PER_BASIC_BLOCK == 0);
static_assert(QK_N % MX_SCALE_MN_PER_BASIC_BLOCK == 0);
static_assert(SV_M % MX_SCALE_MN_PER_BASIC_BLOCK == 0);
static_assert(SV_N % MX_SCALE_MN_PER_BASIC_BLOCK == 0);
static_assert(SV_K % MX_GROUP_K == 0);
static_assert(V32_SCALE_GROUP % MX_GROUP_K == 0);
static_assert(V32_NOPE_MX_K_GROUPS == V32_NUM_SCALES * V32_MX_GROUPS_PER_SCALE);
static_assert(D_QK_PAD == D_NOPE + D_ROPE + 64);

struct NativeMxCutlassContract {
    /*
    Facts mirrored from the CUTLASS/CuTe SM100 block-scaled path:

      cute/arch/mma_sm100_umma.hpp:
        SM100_MMA_MXF8F6F4_SS currently asserts M == 128 and N in [8, 256].

      cute/atom/mma_traits_sm100.hpp:
        MMA_Traits<SM100_MMA_MXF8F6F4_SS>::K == 32,
        SFVecSize == 32, and exposes UMMA::tmem_sf_frg<sf_type, 32, 1,
        true/false> for SFA/SFB.

      cutlass/detail/sm100_blockscaled_layout.hpp:
        Sm1xxBlockScaledConfig<32> owns the 512B basic-block scale layout.

      cute/atom/copy_traits_sm100.hpp:
        SM100_UTCCP_4x32dp128bit_1cta is the scale-factor SMEM->TMEM copy
        path for this one-CTA research shell.
    */
    static constexpr int dense_1sm_m = 128;
    static constexpr int dense_1sm_n_max = 256;
    static constexpr int atom_k = 32;
    static constexpr int sf_vec_size = 32;
    static constexpr int sf_blocks_per_mma_k = 4;
};

static_assert(NativeMxCutlassContract::dense_1sm_m == QK_M);
static_assert(NativeMxCutlassContract::atom_k == MX_GROUP_K);
static_assert(NativeMxCutlassContract::sf_vec_size == MX_DENSE_SF_VEC_SIZE);
static_assert(NativeMxCutlassContract::sf_blocks_per_mma_k == MX_SF_PER_MMA_K);

/*
Native MX storage budget, operands only:

  persistent Q_mx + Q_scale_layout:  80KB + 2.5KB
  one stage K/S/V + scales:          80KB + 16KB + 64KB + 5KB
  two stages plus Q:                 roughly 412KB

NativeMxProductionStorageLayout below makes the extra production staging bytes
explicit:

  persistent Q + Q scale staging + row state:  86,528 bytes
  one K/S/V stage + scale staging:            168,960 bytes
  two stages plus persistent:                 424,448 bytes

The CUDA-free debug emulator additionally uses float QK logits (65,536 bytes)
and float O output (262,144 bytes).  Those are not production SMEM operands:
production must bridge QK TMEM to the online softmax path and keep O in TMEM or
epilogue storage.

A more plausible native-MX staging direction is:

  QK stage, K only:                         84,992 bytes
  S@V stage, S plus one V value chunk:      34,816 bytes

So the double-buffered pipeline below is the desired dependency graph, not a
literal shared-memory allocation plan.  A production kernel must either:

  - keep only one full native stage resident and overlap producers at finer
    granularity,
  - split QK and S@V into separate kernels/stages, or
  - rely on a smaller legal tile shape if CUTLASS/PTX exposes one for this
    exact mx_float8 combination.
*/

struct V32TokenRef {
    const uint8* base;
    bool valid;
};

struct V32TokenParts {
    const uint8* nope;
    const float* scales;
    const bf16* rope;
};

struct MxScale {
    // `value` is the real dequant scale used by the debug emulator:
    //
    //   real_value ~= e4m3_payload * value
    //
    // `encoded_ue8m0` is the one-byte scale-factor payload written into the
    // SM100 512B basic-block layout.  The byte is produced with CUDA's e8m0
    // conversion intrinsic, matching the scale type used by mx_float8_t.
    float value;
    uint8 encoded_ue8m0;
};

struct NativeFp8TilePointers {
    // Logical shared-memory or scratch pointers.  The final implementation
    // should bind these to UMMA/TMA canonical swizzled layouts.
    __nv_fp8_e4m3* q_mx;       // [QK_M, D_QK_PAD]
    __nv_fp8_e4m3* k_mx;       // [QK_N, D_QK_PAD]
    __nv_fp8_e4m3* s_mx;       // [SV_M, SV_K]
    __nv_fp8_e4m3* v_mx;       // [SV_K, SV_N]

    // Float scales for debug emulation.
    MxScale* q_scale;          // [QK_M, D_QK_PAD/32]
    MxScale* k_scale;          // [QK_N, D_QK_PAD/32]
    MxScale* s_scale;          // [SV_M, SV_K/32]
    MxScale* v_scale;          // [SV_N, SV_K/32]

    // Physical 512B basic-block scale-factor layouts for tcgen05/CUTLASS.
    uint8* q_scale_layout;
    uint8* k_scale_layout;
    uint8* s_scale_layout;
    uint8* v_scale_layout;

    // One 512B CUTLASS-compatible staging block per operand before UTCCP copies
    // scale factors into UMMA::tmem_sf_frg.  These are not tcgen05 operands by
    // themselves; they are the source SMEM blocks for tcgen05.cp.
    uint8* q_scale_smem_stage;
    uint8* k_scale_smem_stage;
    uint8* s_scale_smem_stage;
    uint8* v_scale_smem_stage;

    float* qk_tmem_or_debug;   // [QK_M, QK_N], debug stand-in for TMEM
    float* out_tmem_or_debug;  // [SV_M, SV_N], debug stand-in for TMEM
    float* row_m;              // [SV_M], online-softmax running max
    float* row_l;              // [SV_M], online-softmax running exp sum
    float* row_o_scale;        // [SV_M], scale old O before current S@V
};

__device__ __forceinline__ NativeFp8TilePointers native_mx_empty_tile_pointers() {
    NativeFp8TilePointers tile{};
    return tile;
}

struct NativeMxValueChunkPointers {
    /*
    Compact production V chunk for S@V.

    Unlike NativeFp8TilePointers::v_mx, this stores only one value-dim chunk:

      v_mx           [SV_K, MX_SV_N_TILE]
      v_scale        [MX_SV_N_TILE, SV_K/32]       optional debug scales
      v_scale_layout 128 x 4 scale-factor block    exactly 512 bytes

    `n_chunk` is the logical value-dim chunk in the full D_V=512 output.
    */
    __nv_fp8_e4m3* v_mx;
    MxScale* v_scale;
    uint8* v_scale_layout;
    uint8* v_scale_smem_stage;
    int n_chunk;
};

struct NativeMxProductionStorageLayout {
    /*
    Operand-only shared-storage contract for the native PTX path.

    This is intentionally separate from the debug emulator storage:
    - MX payloads and packed scale layouts are needed by both paths.
    - MxScale debug arrays are not needed by tcgen05.mma and are left null.
    - qk_tmem_or_debug/out_tmem_or_debug are TMEM in production, not SMEM.

    The byte counts make the current tile choice explicit.  Even without debug
    arrays, two full K/S/V stages plus persistent Q are too large for one CTA's
    shared memory.  This validates the earlier conclusion that a production
    kernel needs finer-grained staging, split kernels, or a smaller legal tile.
    */
    static constexpr int ALIGN = 128;

    static constexpr int PERSISTENT_Q_MX = 0;
    static constexpr int PERSISTENT_Q_SCALE_LAYOUT =
        align_up_constexpr(PERSISTENT_Q_MX + MX_Q_BYTES, ALIGN);
    static constexpr int PERSISTENT_Q_SCALE_STAGE =
        align_up_constexpr(PERSISTENT_Q_SCALE_LAYOUT + MX_Q_SCALE_LAYOUT_BYTES, ALIGN);
    static constexpr int PERSISTENT_ROW_M =
        align_up_constexpr(PERSISTENT_Q_SCALE_STAGE + MX_SCALE_STAGE_BYTES, ALIGN);
    static constexpr int PERSISTENT_ROW_L =
        align_up_constexpr(PERSISTENT_ROW_M + SV_M * sizeof(float), ALIGN);
    static constexpr int PERSISTENT_ROW_O_SCALE =
        align_up_constexpr(PERSISTENT_ROW_L + SV_M * sizeof(float), ALIGN);
    static constexpr int PERSISTENT_BYTES =
        align_up_constexpr(PERSISTENT_ROW_O_SCALE + SV_M * sizeof(float), ALIGN);

    static constexpr int STAGE_K_MX = 0;
    static constexpr int STAGE_K_SCALE_LAYOUT =
        align_up_constexpr(STAGE_K_MX + MX_K_BYTES, ALIGN);
    static constexpr int STAGE_K_SCALE_STAGE =
        align_up_constexpr(STAGE_K_SCALE_LAYOUT + MX_K_SCALE_LAYOUT_BYTES, ALIGN);
    static constexpr int STAGE_S_MX =
        align_up_constexpr(STAGE_K_SCALE_STAGE + MX_SCALE_STAGE_BYTES, ALIGN);
    static constexpr int STAGE_S_SCALE_LAYOUT =
        align_up_constexpr(STAGE_S_MX + MX_S_BYTES, ALIGN);
    static constexpr int STAGE_S_SCALE_STAGE =
        align_up_constexpr(STAGE_S_SCALE_LAYOUT + MX_S_SCALE_LAYOUT_BYTES, ALIGN);
    static constexpr int STAGE_V_MX =
        align_up_constexpr(STAGE_S_SCALE_STAGE + MX_SCALE_STAGE_BYTES, ALIGN);
    static constexpr int STAGE_V_SCALE_LAYOUT =
        align_up_constexpr(STAGE_V_MX + MX_V_BYTES, ALIGN);
    static constexpr int STAGE_V_SCALE_STAGE =
        align_up_constexpr(STAGE_V_SCALE_LAYOUT + MX_V_SCALE_LAYOUT_BYTES, ALIGN);
    static constexpr int STAGE_BYTES =
        align_up_constexpr(STAGE_V_SCALE_STAGE + MX_SCALE_STAGE_BYTES, ALIGN);

    static constexpr int TWO_STAGE_BYTES =
        PERSISTENT_BYTES + PIPE_STAGES * STAGE_BYTES;
};

struct NativeMxSplitStageStorageLayout {
    /*
    Smaller production direction:

      QK stage owns only K_mx/K_scale for one topk block.
      SV stage owns S_mx/S_scale plus one 128-wide V value chunk.

    This matches the dependency graph better than the full K/S/V stage:
    K can be released immediately after QK, and V can be streamed by value
    chunk while S is already available.
    */
    static constexpr int ALIGN = 128;

    static constexpr int QK_K_MX = 0;
    static constexpr int QK_K_SCALE_LAYOUT =
        align_up_constexpr(QK_K_MX + MX_K_BYTES, ALIGN);
    static constexpr int QK_K_SCALE_STAGE =
        align_up_constexpr(QK_K_SCALE_LAYOUT + MX_K_SCALE_LAYOUT_BYTES, ALIGN);
    static constexpr int QK_STAGE_BYTES =
        align_up_constexpr(QK_K_SCALE_STAGE + MX_SCALE_STAGE_BYTES, ALIGN);

    static constexpr int SV_S_MX = 0;
    static constexpr int SV_S_SCALE_LAYOUT =
        align_up_constexpr(SV_S_MX + MX_S_BYTES, ALIGN);
    static constexpr int SV_S_SCALE_STAGE =
        align_up_constexpr(SV_S_SCALE_LAYOUT + MX_S_SCALE_LAYOUT_BYTES, ALIGN);
    static constexpr int SV_V_NCHUNK_MX =
        align_up_constexpr(SV_S_SCALE_STAGE + MX_SCALE_STAGE_BYTES, ALIGN);
    static constexpr int SV_V_NCHUNK_SCALE_LAYOUT =
        align_up_constexpr(SV_V_NCHUNK_MX + MX_V_NCHUNK_BYTES, ALIGN);
    static constexpr int SV_V_NCHUNK_SCALE_STAGE =
        align_up_constexpr(SV_V_NCHUNK_SCALE_LAYOUT + MX_V_NCHUNK_SCALE_LAYOUT_BYTES, ALIGN);
    static constexpr int SV_NCHUNK_STAGE_BYTES =
        align_up_constexpr(SV_V_NCHUNK_SCALE_STAGE + MX_SCALE_STAGE_BYTES, ALIGN);

    static constexpr int DOUBLE_BUFFERED_QK_BYTES = PIPE_STAGES * QK_STAGE_BYTES;
    static constexpr int DOUBLE_BUFFERED_SV_BYTES = PIPE_STAGES * SV_NCHUNK_STAGE_BYTES;
    static constexpr int DOUBLE_BUFFERED_SPLIT_STAGE_BYTES =
        DOUBLE_BUFFERED_QK_BYTES + DOUBLE_BUFFERED_SV_BYTES;
    static constexpr int SINGLE_BUFFERED_SPLIT_STAGE_BYTES =
        QK_STAGE_BYTES + SV_NCHUNK_STAGE_BYTES;
    static constexpr int SINGLE_BUFFERED_RESIDENT_BYTES =
        NativeMxProductionStorageLayout::PERSISTENT_BYTES +
        SINGLE_BUFFERED_SPLIT_STAGE_BYTES;
    static constexpr int DOUBLE_BUFFERED_RESIDENT_BYTES =
        NativeMxProductionStorageLayout::PERSISTENT_BYTES +
        DOUBLE_BUFFERED_SPLIT_STAGE_BYTES;
};

static_assert(NativeMxProductionStorageLayout::PERSISTENT_BYTES > MX_PERSISTENT_OPERAND_BYTES);
static_assert(NativeMxProductionStorageLayout::STAGE_BYTES > MX_STAGE_OPERAND_BYTES);
static_assert(NativeMxProductionStorageLayout::PERSISTENT_BYTES == 86528);
static_assert(NativeMxProductionStorageLayout::STAGE_BYTES == 168960);
static_assert(NativeMxProductionStorageLayout::TWO_STAGE_BYTES == 424448);
static_assert(MX_QK_LOGITS_DEBUG_BYTES == 65536);
static_assert(MX_OUT_DEBUG_BYTES == 262144);
static_assert(MX_QK_STAGE_BYTES == 84992);
static_assert(MX_SV_NCHUNK_STAGE_BYTES == 34816);
static_assert(NativeMxSplitStageStorageLayout::QK_STAGE_BYTES == MX_QK_STAGE_BYTES);
static_assert(NativeMxSplitStageStorageLayout::SV_NCHUNK_STAGE_BYTES == MX_SV_NCHUNK_STAGE_BYTES);
static_assert(NativeMxSplitStageStorageLayout::DOUBLE_BUFFERED_SPLIT_STAGE_BYTES == 239616);
static_assert(NativeMxSplitStageStorageLayout::SINGLE_BUFFERED_SPLIT_STAGE_BYTES == 119808);
static_assert(NativeMxSplitStageStorageLayout::SINGLE_BUFFERED_RESIDENT_BYTES == 206336);
static_assert(NativeMxSplitStageStorageLayout::DOUBLE_BUFFERED_RESIDENT_BYTES == 326144);

__device__ __forceinline__ uint8* native_mx_byte_ptr(uint8* base, int offset) {
    return base + offset;
}

template <class T>
__device__ __forceinline__ T* native_mx_ptr(uint8* base, int offset) {
    return reinterpret_cast<T*>(native_mx_byte_ptr(base, offset));
}

struct NativeMxProductionStorageView {
    /*
    Operand pointer view only.  It deliberately does not satisfy the
    PipelineStorage concept used by native_mxf8_prefill_pipeline_body_theory().
    That full pipeline still needs a B300-verified TMEM->softmax path for QK
    logits and a final O/TMEM epilogue path.
    */
    uint8* persistent_base;
    uint8* stage_base;

    __device__ __forceinline__ NativeFp8TilePointers persistent_operand_tile_ptrs() const {
        using L = NativeMxProductionStorageLayout;
        NativeFp8TilePointers tile = native_mx_empty_tile_pointers();
        tile.q_mx = native_mx_ptr<__nv_fp8_e4m3>(persistent_base, L::PERSISTENT_Q_MX);
        tile.q_scale_layout = native_mx_ptr<uint8>(persistent_base, L::PERSISTENT_Q_SCALE_LAYOUT);
        tile.q_scale_smem_stage = native_mx_ptr<uint8>(persistent_base, L::PERSISTENT_Q_SCALE_STAGE);
        tile.row_m = native_mx_ptr<float>(persistent_base, L::PERSISTENT_ROW_M);
        tile.row_l = native_mx_ptr<float>(persistent_base, L::PERSISTENT_ROW_L);
        tile.row_o_scale = native_mx_ptr<float>(persistent_base, L::PERSISTENT_ROW_O_SCALE);
        return tile;
    }

    __device__ __forceinline__ NativeFp8TilePointers stage_operand_tile_ptrs(int stage) const {
        using L = NativeMxProductionStorageLayout;
        uint8* base = stage_base + stage * L::STAGE_BYTES;
        NativeFp8TilePointers tile = native_mx_empty_tile_pointers();
        tile.k_mx = native_mx_ptr<__nv_fp8_e4m3>(base, L::STAGE_K_MX);
        tile.s_mx = native_mx_ptr<__nv_fp8_e4m3>(base, L::STAGE_S_MX);
        tile.v_mx = native_mx_ptr<__nv_fp8_e4m3>(base, L::STAGE_V_MX);
        tile.k_scale_layout = native_mx_ptr<uint8>(base, L::STAGE_K_SCALE_LAYOUT);
        tile.s_scale_layout = native_mx_ptr<uint8>(base, L::STAGE_S_SCALE_LAYOUT);
        tile.v_scale_layout = native_mx_ptr<uint8>(base, L::STAGE_V_SCALE_LAYOUT);
        tile.k_scale_smem_stage = native_mx_ptr<uint8>(base, L::STAGE_K_SCALE_STAGE);
        tile.s_scale_smem_stage = native_mx_ptr<uint8>(base, L::STAGE_S_SCALE_STAGE);
        tile.v_scale_smem_stage = native_mx_ptr<uint8>(base, L::STAGE_V_SCALE_STAGE);
        return tile;
    }
};

struct NativeMxSplitStageStorageView {
    /*
    Production-shaped pointer view for the native-MX path.

    This is the first storage contract that matches the dependency graph:

      persistent_base: Q_mx/Q_scale plus row state
      qk_stage_base:   double-buffered K_mx/K_scale for QK
      sv_stage_base:   double-buffered S_mx/S_scale plus one compact V chunk

    The SV stage intentionally has room for only one 128-wide value chunk.  The
    V producer and MMA warpgroup must therefore handshake on one physical chunk
    buffer while carrying n_chunk as the logical payload id:

      wait v_chunk_free -> transcode V chunk n -> v_chunk_ready(n)
      -> issue S@V chunk n -> signal v_chunk_free

    That is the concrete code-level version of "V3.2 scale cannot be passed
    directly to MX": each chunk owns its own 512B MX SFB scale block.
    */
    uint8* persistent_base;
    uint8* qk_stage_base;
    uint8* sv_stage_base;
    float* qk_tmem_debug_base;   // optional [physical_qk_stages, QK_M, QK_N]
    float* out_tmem_debug;       // optional [SV_M, SV_N]
    int qk_physical_stages;       // 0 means PIPE_STAGES
    int sv_physical_stages;       // 0 means PIPE_STAGES

    /*
    qk_tmem_debug_base is the temporary staging buffer used by
    FLASHMLA_ENABLE_THEORETICAL_TMEM_LOGITS_BRIDGE and by the non-PTX emulator.
    FLASHMLA_ENABLE_THEORETICAL_DIRECT_TMEM_SOFTMAX bypasses it and runs
    online softmax directly from the TMEM-loaded register row.

    qk_physical_stages/sv_physical_stages let a B300 bring-up choose the only
    currently plausible resident shared-memory footprint:

      persistent + one QK K stage + one SV S/Vchunk stage = 206,336 bytes

    The logical pipeline still passes stage = block % PIPE_STAGES.  When the
    physical count is one, both logical stages alias the same storage and the
    mbarriers must serialize reuse.  That gives less overlap than the ideal
    double-buffered sketch, but it is the path that can plausibly fit.
    */

    __device__ __forceinline__ int qk_stage_slot(int stage) const {
        FLASH_DEVICE_ASSERT(stage >= 0 && stage < PIPE_STAGES);
        const int count = qk_physical_stages > 0 ? qk_physical_stages : PIPE_STAGES;
        FLASH_DEVICE_ASSERT(count == 1 || count == PIPE_STAGES);
        return stage % count;
    }

    __device__ __forceinline__ int sv_stage_slot(int stage) const {
        FLASH_DEVICE_ASSERT(stage >= 0 && stage < PIPE_STAGES);
        const int count = sv_physical_stages > 0 ? sv_physical_stages : PIPE_STAGES;
        FLASH_DEVICE_ASSERT(count == 1 || count == PIPE_STAGES);
        return stage % count;
    }

    __device__ __forceinline__ NativeFp8TilePointers persistent_tile_ptrs() const {
        using L = NativeMxProductionStorageLayout;
        NativeFp8TilePointers tile = native_mx_empty_tile_pointers();
        tile.q_mx = native_mx_ptr<__nv_fp8_e4m3>(persistent_base, L::PERSISTENT_Q_MX);
        tile.q_scale_layout = native_mx_ptr<uint8>(persistent_base, L::PERSISTENT_Q_SCALE_LAYOUT);
        tile.q_scale_smem_stage = native_mx_ptr<uint8>(persistent_base, L::PERSISTENT_Q_SCALE_STAGE);
        tile.out_tmem_or_debug = out_tmem_debug;
        tile.row_m = native_mx_ptr<float>(persistent_base, L::PERSISTENT_ROW_M);
        tile.row_l = native_mx_ptr<float>(persistent_base, L::PERSISTENT_ROW_L);
        tile.row_o_scale = native_mx_ptr<float>(persistent_base, L::PERSISTENT_ROW_O_SCALE);
        return tile;
    }

    __device__ __forceinline__ NativeFp8TilePointers qk_stage_tile_ptrs(int stage) const {
        using L = NativeMxSplitStageStorageLayout;
        FLASH_DEVICE_ASSERT(stage >= 0 && stage < PIPE_STAGES);
        const int slot = qk_stage_slot(stage);
        uint8* base = qk_stage_base + slot * L::QK_STAGE_BYTES;
        NativeFp8TilePointers tile = native_mx_empty_tile_pointers();
        tile.k_mx = native_mx_ptr<__nv_fp8_e4m3>(base, L::QK_K_MX);
        tile.k_scale_layout = native_mx_ptr<uint8>(base, L::QK_K_SCALE_LAYOUT);
        tile.k_scale_smem_stage = native_mx_ptr<uint8>(base, L::QK_K_SCALE_STAGE);
        tile.qk_tmem_or_debug =
            qk_tmem_debug_base != nullptr ?
            qk_tmem_debug_base + slot * QK_M * QK_N :
            nullptr;
        return tile;
    }

    __device__ __forceinline__ NativeFp8TilePointers sv_stage_s_tile_ptrs(int stage) const {
        using S = NativeMxSplitStageStorageLayout;
        using P = NativeMxProductionStorageLayout;
        FLASH_DEVICE_ASSERT(stage >= 0 && stage < PIPE_STAGES);
        const int slot = sv_stage_slot(stage);
        uint8* base = sv_stage_base + slot * S::SV_NCHUNK_STAGE_BYTES;
        NativeFp8TilePointers tile = native_mx_empty_tile_pointers();
        tile.s_mx = native_mx_ptr<__nv_fp8_e4m3>(base, S::SV_S_MX);
        tile.s_scale_layout = native_mx_ptr<uint8>(base, S::SV_S_SCALE_LAYOUT);
        tile.s_scale_smem_stage = native_mx_ptr<uint8>(base, S::SV_S_SCALE_STAGE);
        tile.qk_tmem_or_debug =
            qk_tmem_debug_base != nullptr ?
            qk_tmem_debug_base + qk_stage_slot(stage) * QK_M * QK_N :
            nullptr;
        tile.out_tmem_or_debug = out_tmem_debug;
        tile.row_m = native_mx_ptr<float>(persistent_base, P::PERSISTENT_ROW_M);
        tile.row_l = native_mx_ptr<float>(persistent_base, P::PERSISTENT_ROW_L);
        tile.row_o_scale = native_mx_ptr<float>(persistent_base, P::PERSISTENT_ROW_O_SCALE);
        return tile;
    }

    __device__ __forceinline__ NativeMxValueChunkPointers sv_stage_v_chunk_ptrs(
        int stage,
        int n_chunk) const {
        using L = NativeMxSplitStageStorageLayout;
        FLASH_DEVICE_ASSERT(stage >= 0 && stage < PIPE_STAGES);
        FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);
        const int slot = sv_stage_slot(stage);
        uint8* base = sv_stage_base + slot * L::SV_NCHUNK_STAGE_BYTES;
        return {
            native_mx_ptr<__nv_fp8_e4m3>(base, L::SV_V_NCHUNK_MX),
            nullptr,
            native_mx_ptr<uint8>(base, L::SV_V_NCHUNK_SCALE_LAYOUT),
            native_mx_ptr<uint8>(base, L::SV_V_NCHUNK_SCALE_STAGE),
            n_chunk,
        };
    }
};

__device__ __forceinline__ NativeMxSplitStageStorageView
make_native_mx_single_buffer_split_stage_view(
    uint8* persistent_base,
    uint8* qk_stage_base,
    uint8* sv_stage_base,
    float* qk_tmem_debug_base,
    float* out_tmem_debug) {
    /*
    Resident shared-memory view for bring-up:

      persistent_base: NativeMxProductionStorageLayout::PERSISTENT_BYTES
      qk_stage_base:   NativeMxSplitStageStorageLayout::QK_STAGE_BYTES
      sv_stage_base:   NativeMxSplitStageStorageLayout::SV_NCHUNK_STAGE_BYTES

    Logical pipeline stages alias these physical buffers.  Correctness then
    depends on the producer/consumer barriers serializing reuse.
    */
    return {
        persistent_base,
        qk_stage_base,
        sv_stage_base,
        qk_tmem_debug_base,
        out_tmem_debug,
        1,
        1,
    };
}

__device__ __forceinline__ NativeMxSplitStageStorageView
make_native_mx_double_buffer_split_stage_view(
    uint8* persistent_base,
    uint8* qk_stage_base,
    uint8* sv_stage_base,
    float* qk_tmem_debug_base,
    float* out_tmem_debug) {
    /*
    Ideal overlap view.  The resident footprint is currently too large for the
    target sparse-prefill CTA shape unless the final B300 layout finds a smaller
    legal tile or moves part of the state out of shared memory.
    */
    return {
        persistent_base,
        qk_stage_base,
        sv_stage_base,
        qk_tmem_debug_base,
        out_tmem_debug,
        PIPE_STAGES,
        PIPE_STAGES,
    };
}

__device__ __forceinline__ NativeFp8TilePointers native_mx_attach_persistent_q(
    NativeFp8TilePointers stage_tile,
    NativeFp8TilePointers persistent_tile) {
    stage_tile.q_mx = persistent_tile.q_mx;
    stage_tile.q_scale = persistent_tile.q_scale;
    stage_tile.q_scale_layout = persistent_tile.q_scale_layout;
    stage_tile.q_scale_smem_stage = persistent_tile.q_scale_smem_stage;
    stage_tile.out_tmem_or_debug = persistent_tile.out_tmem_or_debug;
    stage_tile.row_m = persistent_tile.row_m;
    stage_tile.row_l = persistent_tile.row_l;
    stage_tile.row_o_scale = persistent_tile.row_o_scale;
    return stage_tile;
}

__device__ __forceinline__ bool native_mx_qk_debug_ready(NativeFp8TilePointers tile) {
    return
        tile.q_mx != nullptr &&
        tile.k_mx != nullptr &&
        (tile.q_scale != nullptr || tile.q_scale_layout != nullptr) &&
        (tile.k_scale != nullptr || tile.k_scale_layout != nullptr) &&
        tile.qk_tmem_or_debug != nullptr;
}

__device__ __forceinline__ bool native_mx_compact_sv_debug_ready(
    NativeFp8TilePointers s_tile,
    NativeMxValueChunkPointers v_chunk) {
    return
        s_tile.s_mx != nullptr &&
        (s_tile.s_scale != nullptr || s_tile.s_scale_layout != nullptr) &&
        v_chunk.v_mx != nullptr &&
        (v_chunk.v_scale != nullptr || v_chunk.v_scale_layout != nullptr) &&
        s_tile.out_tmem_or_debug != nullptr &&
        s_tile.row_o_scale != nullptr;
}

struct NativeMxPipelineStage {
    NativeFp8TilePointers tile;
    int topk_block;
};

struct NativeMxPipelineRole {
    static constexpr int Softmax = 0;
    static constexpr int KProducer = 1;
    static constexpr int VProducer = 2;
    static constexpr int Mma = 3;
};

struct NativeMxNamedBarrierId {
    static constexpr int ScaleStageWritten = 0;
};

struct NativeMxMmaCall {
    uint32_t d_tmem;
    uint64_t a_desc;
    uint64_t b_desc;
    uint64_t block_scaled_idesc;
    uint32_t a_scale_tmem_addr;
    uint32_t b_scale_tmem_addr;
    bool accumulate;
};

enum class NativeMxScaleOperand : uint8 {
    SFA,
    SFB,
};

struct NativeMxScaleTmemSpan {
    /*
    Packed scale bytes live in q/k/s/v_scale_layout.  Native block-scaled MMA
    does not consume that pointer directly.  The bytes for the current MMA tile
    must first be copied to the scale-factor TMEM fragment, and the fragment's
    address becomes the tsfa/tsfb operand in tcgen05.mma.

    This span describes exactly which logical scale[mn][k_group] window must
    be present at `tmem_addr` before the matching NativeMxMmaCall is issued.
    */
    const uint8* packed_layout;
    uint8* smem_stage;
    uint64_t smem_stage_desc;
    NativeMxScaleOperand operand;
    uint32_t tmem_addr;
    int total_mn;
    int mn_start;
    int mn_count;
    int k_group_start;
    int k_group_count;
};

struct NativeMxMmaTileSpec {
    int m;
    int n;
    int k;
    int k_dim_start;
    int n_start;
};

enum class NativeMxOperandRole : uint8 {
    Q,
    K,
    S,
    V,
};

struct NativeMxOperandAtomSpan {
    /*
    Logical slice described by an atom's smem descriptor.  This is metadata for
    the final CuTe descriptor layer and for debug assertions; the descriptor
    itself is still carried as `smem_desc`.

    For V, the physical storage is v_mx[token][value_dim], but MMA-B consumes it
    as logical [value_dim, token_k].  `transpose_for_mma` marks that contract.
    */
    const __nv_fp8_e4m3* base;
    uint64_t smem_desc;
    NativeMxOperandRole role;
    int row_start;
    int col_start;
    int rows;
    int cols;
    int ld;
    bool transpose_for_mma;
};

struct NativeMxMmaAtomPlan {
    NativeMxMmaCall call;
    NativeMxOperandAtomSpan a_operand;
    NativeMxOperandAtomSpan b_operand;
    NativeMxMmaTileSpec tile;
    int k_atom;
};

struct NativeMxMmaIssuePlan {
    NativeMxScaleTmemSpan a_scale_span;
    NativeMxScaleTmemSpan b_scale_span;
    NativeMxMmaTileSpec scale_block_tile;
    NativeMxMmaAtomPlan atoms[MX_SF_PER_MMA_K];
};

struct NativeMxMmaDescriptorSet {
    /*
    QK scale staging: one [128,128,128] scale block per 128-wide D_QK chunk.
    D_QK=576 is padded to 640, so the scale-copy sequence has five chunks.
    Actual tcgen05.mma issue is finer: four K32 atom calls per scale chunk.

    S@V scale staging: one [128,128,128] scale block per 128-wide value-dim
    chunk.  D_V=512 therefore uses four independent N chunks.  Actual
    tcgen05.mma issue is again four K32 atom calls per N chunk.
    */
    NativeMxMmaIssuePlan qk[MX_QK_K_CHUNKS];
    NativeMxMmaIssuePlan sv[MX_SV_N_CHUNKS];
};

struct NativeMxQkDescriptorSet {
    NativeMxMmaIssuePlan qk[MX_QK_K_CHUNKS];
};

struct NativeMxQkDescriptorInputs {
    /*
    Descriptor material for the split-stage QK path.

    Q comes from persistent storage, K comes from the QK stage.  The descriptor
    layer still has to produce atom-level K32 descriptors and scale-fragment
    TMEM addresses for every K128 scale-copy chunk.
    */
    uint32_t qk_tmem_d;

    uint64_t q_desc[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];
    uint64_t k_desc[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];
    uint64_t qk_block_scaled_idesc[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];

    uint32_t q_scale_tmem_block_addr[MX_QK_K_CHUNKS];
    uint32_t k_scale_tmem_block_addr[MX_QK_K_CHUNKS];
    uint32_t q_scale_tmem_addr[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];
    uint32_t k_scale_tmem_addr[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];

    uint64_t q_scale_smem_desc[MX_QK_K_CHUNKS];
    uint64_t k_scale_smem_desc[MX_QK_K_CHUNKS];
};

struct NativeMxRawDescriptorInputs {
    /*
    Low-level descriptor material that must be produced by the final CuTe/CUTLASS
    storage layer:

      - q/k/s/v_desc are swizzled shared-memory descriptors.
        Descriptors are K32 atom descriptors, not K128 scale-block descriptors.
        v_desc[n][k] must describe V as the MMA-B logical tensor
        [value_dim n_chunk, token_k_group32], even if the debug storage is
        indexed as v_mx[token][value_dim].
      - *_block_scaled_idesc are runtime instruction descriptors after calling
        the block-scaled descriptor helper with the atom's scale TMEM addresses.
      - *_scale_tmem_block_addr are the UTCCP destination base addresses for a
        K128 scale block.
      - *_scale_tmem_addr are the four K32 scale-factor TMEM fragment addresses
        derived from the matching K128 block base.  Callers may leave these zero
        when using native_mx_complete_scale_atom_tmem_addresses().
      - *_scale_smem_desc are the SMEM descriptors for the 512B scale staging
        blocks after materialize_mx_scale_span_to_smem_stage().
    */
    uint32_t qk_tmem_d;
    uint32_t sv_tmem_d[MX_SV_N_CHUNKS];

    uint64_t q_desc[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];
    uint64_t k_desc[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];
    uint64_t s_desc[MX_SF_PER_MMA_K];
    uint64_t v_desc[MX_SV_N_CHUNKS][MX_SF_PER_MMA_K];

    uint64_t qk_block_scaled_idesc[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];
    uint64_t sv_block_scaled_idesc[MX_SV_N_CHUNKS][MX_SF_PER_MMA_K];

    uint32_t q_scale_tmem_block_addr[MX_QK_K_CHUNKS];
    uint32_t k_scale_tmem_block_addr[MX_QK_K_CHUNKS];
    uint32_t s_scale_tmem_block_addr;
    uint32_t v_scale_tmem_block_addr[MX_SV_N_CHUNKS];

    uint32_t q_scale_tmem_addr[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];
    uint32_t k_scale_tmem_addr[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];
    uint32_t s_scale_tmem_addr[MX_SF_PER_MMA_K];
    uint32_t v_scale_tmem_addr[MX_SV_N_CHUNKS][MX_SF_PER_MMA_K];

    uint64_t q_scale_smem_desc[MX_QK_K_CHUNKS];
    uint64_t k_scale_smem_desc[MX_QK_K_CHUNKS];
    uint64_t s_scale_smem_desc;
    uint64_t v_scale_smem_desc[MX_SV_N_CHUNKS];
};

struct NativeMxCompactSvDescriptorInputs {
    /*
    Descriptor material for one compact S@V value chunk.

    S uses the usual [SV_M, SV_K] tile.  V uses compact [SV_K, 128] storage for
    logical output value chunk `n_chunk`.
    */
    int n_chunk;
    uint32_t sv_tmem_d;

    uint64_t s_desc[MX_SF_PER_MMA_K];
    uint64_t v_desc[MX_SF_PER_MMA_K];
    uint64_t sv_block_scaled_idesc[MX_SF_PER_MMA_K];

    uint32_t s_scale_tmem_block_addr;
    uint32_t v_scale_tmem_block_addr;
    uint32_t s_scale_tmem_addr[MX_SF_PER_MMA_K];
    uint32_t v_scale_tmem_addr[MX_SF_PER_MMA_K];

    uint64_t s_scale_smem_desc;
    uint64_t v_scale_smem_desc;
};

struct NativeMxQkSmemDescriptorMaterial {
    /*
    CuTe/CUTLASS material for one split-stage QK tile.

    This is deliberately separate from NativeMxTmemAddressPlan.  These fields
    describe how the staged operands and staged 512B scale blocks are viewed in
    shared memory.  The TMEM addresses used by tcgen05.mma are bound later, and
    the block-scaled instruction descriptor is then regenerated from those
    SFA/SFB TMEM addresses.
    */
    uint64_t q_desc[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];
    uint64_t k_desc[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];
    uint64_t q_scale_smem_desc[MX_QK_K_CHUNKS];
    uint64_t k_scale_smem_desc[MX_QK_K_CHUNKS];
};

struct NativeMxCompactSvSmemDescriptorMaterial {
    /*
    CuTe/CUTLASS material for one compact S@V value chunk.

    `n_chunk` is part of the material because V descriptors depend on the
    logical output-value chunk, even when every chunk reuses the same physical
    compact V buffer.
    */
    int n_chunk;
    uint64_t s_desc[MX_SF_PER_MMA_K];
    uint64_t v_desc[MX_SF_PER_MMA_K];
    uint64_t s_scale_smem_desc;
    uint64_t v_scale_smem_desc;
};

struct NativeMxTmemAddressPlan {
    /*
    The production kernel must allocate one 1SM TMEM arena and derive these
    addresses with the same CuTe tensors that create the MMA descriptors.

    Existing bf16 sparse-prefill kernels allocate 512 TMEM columns and use:

      O accumulator:      columns 0..255
      P/QK scratch:       columns 256..319 or later, depending on head shape

    Native MX additionally needs scale-factor TMEM fragments.  CUTLASS lays
    those out immediately after the accumulator tensor:

      tCtSFA = accumulator + find_tmem_tensor_col_offset(accumulator)
      tCtSFB = tCtSFA      + find_tmem_tensor_col_offset(tCtSFA)

    This struct is therefore not a replacement for CuTe.  It is the contract
    that the final CuTe descriptor layer must fill before this file can be
    registered in setup.py.  The block-base values are the real UTCCP destinations;
    the K32 atom addresses are deterministic projections of those bases, not
    independent inputs and not packed q/k/s/v scale-layout byte pointers.

    Important: scale-fragment TMEM addresses are not plain column numbers.
    CUTLASS' make_runtime_instr_desc_block_scaled() reads the top two bits of
    tCtSFA/tCtSFB addresses as the scale-fragment selector id.  Capacity checks
    therefore must mask those bits off and validate only the column payload.
    */
    uint32_t tmem_base;
    uint32_t o_tmem_d;
    uint32_t qk_tmem_d[PIPE_STAGES];
    uint32_t sv_tmem_d[MX_SV_N_CHUNKS];

    uint32_t q_scale_tmem_block_addr[MX_QK_K_CHUNKS];
    uint32_t k_scale_tmem_block_addr[MX_QK_K_CHUNKS];
    uint32_t s_scale_tmem_block_addr;
    uint32_t v_scale_tmem_block_addr[MX_SV_N_CHUNKS];

    uint32_t q_scale_tmem_addr[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];
    uint32_t k_scale_tmem_addr[MX_QK_K_CHUNKS][MX_SF_PER_MMA_K];
    uint32_t s_scale_tmem_addr[MX_SF_PER_MMA_K];
    uint32_t v_scale_tmem_addr[MX_SV_N_CHUNKS][MX_SF_PER_MMA_K];
};

__host__ __device__ __forceinline__ uint32_t native_mx_tmem_addr_column_bits(
    uint32_t addr) {
    return addr & NATIVE_MX_TMEM_ADDR_COLUMN_MASK;
}

__host__ __device__ __forceinline__ uint32_t native_mx_tmem_addr_sfid_bits(
    uint32_t addr) {
    return addr & NATIVE_MX_TMEM_ADDR_SFID_MASK;
}

__host__ __device__ __forceinline__ uint32_t native_mx_make_scale_tmem_block_addr(
    uint32_t sfid,
    uint32_t column) {
    return ((sfid & 0x3U) << NATIVE_MX_TMEM_ADDR_SFID_SHIFT) |
           (column & NATIVE_MX_TMEM_ADDR_COLUMN_MASK);
}

__host__ __device__ __forceinline__ uint32_t native_mx_scale_tmem_atom_addr_from_block(
    uint32_t block_addr,
    int k_atom) {
    return native_mx_tmem_addr_sfid_bits(block_addr) |
           (native_mx_tmem_addr_column_bits(block_addr) +
            static_cast<uint32_t>(k_atom) * NATIVE_MX_SCALE_TMEM_K32_STRIDE_COLS);
}

__host__ __device__ __forceinline__ bool native_mx_scale_tmem_block_has_k128_span(
    uint32_t block_addr) {
    return
        native_mx_tmem_addr_column_bits(block_addr) +
        NATIVE_MX_SCALE_TMEM_K128_SPAN_COLS <=
        NATIVE_MX_TMEM_CAPACITY_COLS;
}

__host__ __device__ __forceinline__ NativeMxTmemAddressPlan
native_mx_complete_scale_atom_tmem_addresses(NativeMxTmemAddressPlan tmem) {
    /*
    Callers only need to bind the K128 scale-block base addresses.  The four K32
    atom addresses are a deterministic projection of that base in the same shape
    CUTLASS uses for tCtSFA/tCtSFB(_,_,k_block): +0, +128, +256, +384 columns,
    preserving the high two scale-fragment selector bits.
    */
    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            tmem.q_scale_tmem_addr[k_chunk][k_atom] =
                native_mx_scale_tmem_atom_addr_from_block(
                    tmem.q_scale_tmem_block_addr[k_chunk],
                    k_atom);
            tmem.k_scale_tmem_addr[k_chunk][k_atom] =
                native_mx_scale_tmem_atom_addr_from_block(
                    tmem.k_scale_tmem_block_addr[k_chunk],
                    k_atom);
        }
    }

    for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
        tmem.s_scale_tmem_addr[k_atom] =
            native_mx_scale_tmem_atom_addr_from_block(
                tmem.s_scale_tmem_block_addr,
                k_atom);
    }

    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            tmem.v_scale_tmem_addr[n_chunk][k_atom] =
                native_mx_scale_tmem_atom_addr_from_block(
                    tmem.v_scale_tmem_block_addr[n_chunk],
                    k_atom);
        }
    }

    return tmem;
}

__host__ __device__ __forceinline__ NativeMxTmemAddressPlan
native_mx_make_single_buffer_tmem_address_plan_theory(uint32_t tmem_base) {
    /*
    Fixed bring-up TMEM layout for the single-buffer native-MX kernel.

    This is deliberately narrower than the final CuTe-derived production plan.
    It mirrors the existing SM100 sparse-prefill convention of allocating one
    512-column 1SM TMEM arena, keeps the compact S@V accumulator at column 0,
    and puts the QK logits scratch at column 256 so QK and SV never alias.

    MX scale blocks are different from accumulator columns: the high two bits are
    scale-fragment selector ids consumed by block-scaled tcgen05.mma.  A K128
    scale block spans columns base+0, base+128, base+256, base+384, so this
    fixed 512-column plan is only valid when the full allocation starts at a base
    for which base+512 fits in the 1SM TMEM capacity.  Q/S reuse SFA id 1 and K/V
    reuse SFB id 2 because QK and S@V are issued sequentially in this
    single-buffer bring-up path.

    This helper is a B300 bring-up scaffold.  The final registered kernel should
    replace it with the tCtSFA/tCtSFB addresses produced by the same CuTe tensors
    that build the MMA descriptors.
    */
    NativeMxTmemAddressPlan tmem{};
    tmem.tmem_base = tmem_base;
    tmem.o_tmem_d = tmem_base + NATIVE_MX_SINGLE_BUFFER_SV_TMEM_COL;
    for (int stage = 0; stage < PIPE_STAGES; ++stage) {
        tmem.qk_tmem_d[stage] = tmem_base + NATIVE_MX_SINGLE_BUFFER_QK_TMEM_COL;
    }
    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        tmem.sv_tmem_d[n_chunk] = tmem_base + NATIVE_MX_SINGLE_BUFFER_SV_TMEM_COL;
    }

    const uint32_t sfa_block_addr =
        native_mx_make_scale_tmem_block_addr(
            NATIVE_MX_SINGLE_BUFFER_SCALE_SFA_SFID,
            tmem_base + NATIVE_MX_SINGLE_BUFFER_SV_TMEM_COL);
    const uint32_t sfb_block_addr =
        native_mx_make_scale_tmem_block_addr(
            NATIVE_MX_SINGLE_BUFFER_SCALE_SFB_SFID,
            tmem_base + NATIVE_MX_SINGLE_BUFFER_SV_TMEM_COL);
    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        tmem.q_scale_tmem_block_addr[k_chunk] = sfa_block_addr;
        tmem.k_scale_tmem_block_addr[k_chunk] = sfb_block_addr;
    }
    tmem.s_scale_tmem_block_addr = sfa_block_addr;
    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        tmem.v_scale_tmem_block_addr[n_chunk] = sfb_block_addr;
    }

    return native_mx_complete_scale_atom_tmem_addresses(tmem);
}

struct NativeMxSplitStageDescriptorMaterial {
    /*
    Production-shaped descriptor material for the split-stage pipeline.

    This is the object the real B300 kernel should eventually build after:
      1. laying out the split shared storage,
      2. allocating the 1SM TMEM arena,
      3. deriving the CuTe SMEM descriptors and tCtSFA/tCtSFB addresses.

    It intentionally does not contain raw qk/sv_block_scaled_idesc values.
    Those depend on the TMEM scale-fragment address and are regenerated by
    bind_native_mx_*_descriptor_inputs() at the exact issue boundary.
    */
    NativeMxTmemAddressPlan tmem;
    int qk_material_stages;  // 0 means PIPE_STAGES
    int sv_material_stages;  // 0 means PIPE_STAGES
    NativeMxQkSmemDescriptorMaterial qk_smem_by_stage[PIPE_STAGES];
    NativeMxCompactSvSmemDescriptorMaterial
        sv_smem_by_stage[PIPE_STAGES][MX_SV_N_CHUNKS];
};

struct NativeMxSingleBufferDescriptorMaterialInputs {
    /*
    Minimal descriptor package for the first executable native-MX bring-up
    kernel.

    This deliberately does not ask callers for PIPE_STAGES worth of material:
    the single-buffer launch aliases both logical stages onto one physical QK
    stage and one physical SV stage.  The four SV entries are still distinct
    because each value-dim chunk has its own compact V descriptor and SFB scale
    fragment.
    */
    NativeMxTmemAddressPlan tmem;
    NativeMxQkSmemDescriptorMaterial qk_smem;
    NativeMxCompactSvSmemDescriptorMaterial sv_smem[MX_SV_N_CHUNKS];
};

inline NativeMxSplitStageDescriptorMaterial make_native_mx_single_buffer_descriptor_material(
    const NativeMxSingleBufferDescriptorMaterialInputs& inputs) {
    NativeMxSplitStageDescriptorMaterial material{};
    material.tmem = native_mx_complete_scale_atom_tmem_addresses(inputs.tmem);
    material.qk_material_stages = 1;
    material.sv_material_stages = 1;
    for (int stage = 1; stage < PIPE_STAGES; ++stage) {
        material.tmem.qk_tmem_d[stage] = material.tmem.qk_tmem_d[0];
    }
    material.qk_smem_by_stage[0] = inputs.qk_smem;
    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        material.sv_smem_by_stage[0][n_chunk] = inputs.sv_smem[n_chunk];
    }
    return material;
}

__device__ __forceinline__ int native_mx_physical_stage_count(int count) {
    const int normalized = count > 0 ? count : PIPE_STAGES;
    FLASH_DEVICE_ASSERT(normalized == 1 || normalized == PIPE_STAGES);
    return normalized;
}

__device__ __forceinline__ int native_mx_ready_phase_for_block(
    int block,
    int physical_stage_count) {
    /*
    Existing SM100 sparse-prefill kernels use phase = (k / NUM_BUFS) & 1 for
    producer->consumer readiness.  Native MX must use the same rule even when
    logical PIPE_STAGES aliases one physical buffer during bring-up.
    */
    FLASH_DEVICE_ASSERT(block >= 0);
    const int count = native_mx_physical_stage_count(physical_stage_count);
    return (block / count) & 1;
}

__device__ __forceinline__ bool native_mx_reuses_physical_stage(
    int block,
    int physical_stage_count) {
    FLASH_DEVICE_ASSERT(block >= 0);
    const int count = native_mx_physical_stage_count(physical_stage_count);
    return block >= count;
}

__device__ __forceinline__ int native_mx_reuse_phase_for_block(
    int block,
    int physical_stage_count) {
    /*
    For a free/reuse barrier, block b waits for the completion signal produced
    by the previous logical block that occupied the same physical slot:

      previous = b - physical_stage_count

    This is the piece that cannot be recovered from `stage = b % PIPE_STAGES`
    alone, and is why the split-stage hook signatures carry both block and
    stage.
    */
    FLASH_DEVICE_ASSERT(native_mx_reuses_physical_stage(block, physical_stage_count));
    const int count = native_mx_physical_stage_count(physical_stage_count);
    return ((block - count) / count) & 1;
}

__device__ __forceinline__ transac_bar_t* native_mx_stage_barrier(
    transac_bar_t* base,
    int slot) {
    return base == nullptr ? nullptr : base + slot;
}

__device__ __forceinline__ transac_bar_t* native_mx_sv_chunk_barrier(
    transac_bar_t* base,
    int slot,
    int n_chunk) {
    return
        base == nullptr ?
        nullptr :
        base + slot * MX_SV_N_CHUNKS + n_chunk;
}

__device__ __forceinline__ void native_mx_arrive_shared_barrier(transac_bar_t* bar) {
    if (bar == nullptr) {
        return;
    }
    fence_view_async_shared();
    bar->arrive();
}

__device__ __forceinline__ void native_mx_wait_shared_barrier(
    transac_bar_t* bar,
    int phase) {
    if (bar == nullptr) {
        return;
    }
    bar->wait(phase);
}

__device__ __forceinline__ void native_mx_arrive_tc_barrier(transac_bar_t* bar) {
    if (bar == nullptr) {
        return;
    }
    if ((threadIdx.x & (WARP_GROUP_THREADS - 1)) != 0) {
        return;
    }
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    ku::umma_arrive_noelect(*bar);
#else
    bar->arrive();
#endif
}

__device__ __forceinline__ void native_mx_wait_tc_barrier(
    transac_bar_t* bar,
    int phase) {
    if (bar == nullptr) {
        return;
    }
    bar->wait(phase);
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    ku::tcgen05_after_thread_sync();
#endif
}

struct NativeMxSplitStageBarrierStorage {
    /*
    Concrete mbarrier storage for the split-stage native-MX pipeline.

    The same physical slot count used by NativeMxSplitStageStorageView must be
    used here.  With one physical QK/SV stage, logical stage 0 and 1 alias the
    same storage and the phase helpers above serialize reuse with block number,
    not with `stage` alone.
    */
    transac_bar_t q_ready;
    transac_bar_t qk_stage_free[PIPE_STAGES];
    transac_bar_t k_ready[PIPE_STAGES];
    transac_bar_t qk_ready[PIPE_STAGES];
    transac_bar_t s_ready[PIPE_STAGES];
    transac_bar_t v_chunk_ready[PIPE_STAGES][MX_SV_N_CHUNKS];
    transac_bar_t sv_chunk_done[PIPE_STAGES][MX_SV_N_CHUNKS];
    transac_bar_t final_sv_done;
};

struct NativeMxSplitStageBarrierPlan {
    /*
    Pointer view over NativeMxSplitStageBarrierStorage.

    `sv_chunk_done` is deliberately named after the Tensor-Core event rather
    than "V free": after S@V completes, the compact V buffer may be reused and
    the corresponding O chunk is ready for a final TMEM load/store.  Both users
    must wait on the same tcgen05 completion event.
    */
    transac_bar_t* q_ready;
    transac_bar_t* qk_stage_free;   // [qk_physical_stages], tcgen05 QK done
    transac_bar_t* k_ready;         // [qk_physical_stages], producer stores done
    transac_bar_t* qk_ready;        // [qk_physical_stages], tcgen05 QK done
    transac_bar_t* s_ready;         // [sv_physical_stages], softmax stores done
    transac_bar_t* v_chunk_ready;   // [sv_physical_stages, MX_SV_N_CHUNKS]
    transac_bar_t* sv_chunk_done;   // [sv_physical_stages, MX_SV_N_CHUNKS]
    transac_bar_t* final_sv_done;
    int qk_physical_stages;         // 0 means PIPE_STAGES
    int sv_physical_stages;         // 0 means PIPE_STAGES
};

__device__ __forceinline__ NativeMxSplitStageBarrierPlan
make_native_mx_split_stage_barrier_plan(
    NativeMxSplitStageBarrierStorage* storage,
    int qk_physical_stages,
    int sv_physical_stages) {
    FLASH_DEVICE_ASSERT(storage != nullptr);
    const int qk_count = native_mx_physical_stage_count(qk_physical_stages);
    const int sv_count = native_mx_physical_stage_count(sv_physical_stages);
    return {
        &storage->q_ready,
        &storage->qk_stage_free[0],
        &storage->k_ready[0],
        &storage->qk_ready[0],
        &storage->s_ready[0],
        &storage->v_chunk_ready[0][0],
        &storage->sv_chunk_done[0][0],
        &storage->final_sv_done,
        qk_count,
        sv_count,
    };
}

__device__ __forceinline__ void init_native_mx_split_stage_barriers(
    NativeMxSplitStageBarrierStorage& storage,
    int qk_physical_stages,
    int sv_physical_stages) {
    /*
    Barrier counts match the pipeline roles:
      q_ready:       one producer warpgroup, 128 thread arrivals
      k_ready:       one KProducer warpgroup, 128 thread arrivals
      qk_ready:      one tcgen05 QK commit
      s_ready:       one Softmax warpgroup, 128 thread arrivals
      v_chunk_ready: one VProducer warpgroup, 128 thread arrivals
      sv_chunk_done: one tcgen05 S@V commit
      final_sv_done: one MMA warpgroup after final epilogue stores,
                     128 thread arrivals

    qk_stage_free/sv_chunk_done are not pre-signaled for block 0; producer code
    skips free waits until a physical slot is actually reused.

    Call this before the pipeline and then synchronize the CTA, matching the
    existing SM100 kernels' barrier-init pattern.
    */
    if (threadIdx.x != 0) {
        return;
    }

    const int qk_count = native_mx_physical_stage_count(qk_physical_stages);
    const int sv_count = native_mx_physical_stage_count(sv_physical_stages);

    storage.q_ready.init(WARP_GROUP_THREADS);
    for (int slot = 0; slot < qk_count; ++slot) {
        storage.qk_stage_free[slot].init(1);
        storage.k_ready[slot].init(WARP_GROUP_THREADS);
        storage.qk_ready[slot].init(1);
    }
    for (int slot = 0; slot < sv_count; ++slot) {
        storage.s_ready[slot].init(WARP_GROUP_THREADS);
        for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
            storage.v_chunk_ready[slot][n_chunk].init(WARP_GROUP_THREADS);
            storage.sv_chunk_done[slot][n_chunk].init(1);
        }
    }
    storage.final_sv_done.init(WARP_GROUP_THREADS);
    fence_barrier_init();
}

struct NativeMxSingleBufferSplitStageSharedStorage {
    /*
    Bring-up shared-storage shape for the only resident footprint that is close
    to practical with the current legal MX tile:

      persistent Q/row state: 86,528 bytes
      one QK K stage:         84,992 bytes
      one SV S/Vchunk stage:  34,816 bytes

    The logical pipeline still exposes PIPE_STAGES=2, but both logical stages
    alias one physical QK stage and one physical SV stage.  The barrier plan is
    therefore part of this storage contract; without it, aliasing is unsafe.
    */
    alignas(NativeMxProductionStorageLayout::ALIGN)
        uint8 persistent[NativeMxProductionStorageLayout::PERSISTENT_BYTES];
    alignas(NativeMxSplitStageStorageLayout::ALIGN)
        uint8 qk_stage[NativeMxSplitStageStorageLayout::QK_STAGE_BYTES];
    alignas(NativeMxSplitStageStorageLayout::ALIGN)
        uint8 sv_stage[NativeMxSplitStageStorageLayout::SV_NCHUNK_STAGE_BYTES];
    NativeMxSplitStageBarrierStorage barriers;
    uint32_t tmem_start_addr;
};

static constexpr int NATIVE_MX_SINGLE_BUFFER_OPERAND_BYTES =
    NativeMxSplitStageStorageLayout::SINGLE_BUFFERED_RESIDENT_BYTES;

static constexpr int NATIVE_MX_SINGLE_BUFFER_SHARED_STORAGE_BYTES =
    sizeof(NativeMxSingleBufferSplitStageSharedStorage);
static constexpr int NATIVE_MX_SINGLE_BUFFER_DYNAMIC_SMEM_BYTES =
    NATIVE_MX_SINGLE_BUFFER_SHARED_STORAGE_BYTES +
    NativeMxProductionStorageLayout::ALIGN - 1;

static_assert(
    NATIVE_MX_SINGLE_BUFFER_SHARED_STORAGE_BYTES >=
    NATIVE_MX_SINGLE_BUFFER_OPERAND_BYTES);
static_assert(NATIVE_MX_SINGLE_BUFFER_DYNAMIC_SMEM_BYTES < 227 * 1024);

__device__ __forceinline__ NativeMxSingleBufferSplitStageSharedStorage&
native_mx_single_buffer_storage_from_dynamic_smem(char* wksp_buf) {
    const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(wksp_buf);
    const std::uintptr_t aligned =
        (raw + NativeMxProductionStorageLayout::ALIGN - 1) &
        ~(std::uintptr_t(NativeMxProductionStorageLayout::ALIGN) - 1);
    return *reinterpret_cast<NativeMxSingleBufferSplitStageSharedStorage*>(aligned);
}

__device__ __forceinline__ NativeMxSplitStageStorageView
make_native_mx_single_buffer_split_stage_view(
    NativeMxSingleBufferSplitStageSharedStorage& storage,
    float* qk_tmem_debug_base,
    float* out_tmem_debug) {
    return make_native_mx_single_buffer_split_stage_view(
        &storage.persistent[0],
        &storage.qk_stage[0],
        &storage.sv_stage[0],
        qk_tmem_debug_base,
        out_tmem_debug);
}

__device__ __forceinline__ NativeMxSplitStageBarrierPlan
make_native_mx_single_buffer_split_stage_barrier_plan(
    NativeMxSingleBufferSplitStageSharedStorage& storage) {
    return make_native_mx_split_stage_barrier_plan(&storage.barriers, 1, 1);
}

__device__ __forceinline__ void init_native_mx_single_buffer_split_stage_barriers(
    NativeMxSingleBufferSplitStageSharedStorage& storage) {
    init_native_mx_split_stage_barriers(storage.barriers, 1, 1);
}

__device__ __forceinline__ bool native_mx_descriptor_is_bound(uint64_t desc) {
    return desc != 0;
}

__device__ __forceinline__ bool native_mx_tmem_addr_is_bound(uint32_t addr) {
    /*
    Address 0 is legal for a real accumulator base.  Scale fragments, however,
    must come from the non-accumulator TMEM region.  Treating zero as unbound
    here catches the common research-shell mistake: enabling theoretical PTX
    while still passing zero-filled descriptor inputs.
    */
    return addr != 0;
}

__device__ __forceinline__ uint32_t native_mx_tmem_addr_column(uint32_t addr) {
    return addr & NATIVE_MX_TMEM_ADDR_COLUMN_MASK;
}

__device__ __forceinline__ uint32_t native_mx_tmem_addr_sfid(uint32_t addr) {
    return
        (addr & NATIVE_MX_TMEM_ADDR_SFID_MASK) >>
        NATIVE_MX_TMEM_ADDR_SFID_SHIFT;
}

__device__ __forceinline__ bool native_mx_tmem_addr_in_capacity(uint32_t addr) {
    return native_mx_tmem_addr_column(addr) < NATIVE_MX_TMEM_CAPACITY_COLS;
}

__device__ __forceinline__ bool native_mx_tmem_accumulator_addr_in_capacity(
    uint32_t addr) {
    return native_mx_tmem_addr_sfid(addr) == 0 &&
           native_mx_tmem_addr_in_capacity(addr);
}

__device__ __forceinline__ bool native_mx_scale_tmem_atom_addr_matches_block(
    uint32_t block_addr,
    uint32_t atom_addr) {
    return native_mx_tmem_addr_is_bound(block_addr) &&
           native_mx_tmem_addr_is_bound(atom_addr) &&
           native_mx_scale_tmem_block_has_k128_span(block_addr) &&
           native_mx_tmem_addr_in_capacity(atom_addr) &&
           native_mx_tmem_addr_sfid(block_addr) == native_mx_tmem_addr_sfid(atom_addr);
}

__device__ __forceinline__ void validate_native_mx_tmem_address_plan(
    const NativeMxTmemAddressPlan& tmem) {
    /*
    Only scale-fragment addresses are required to be nonzero here.  Accumulator
    TMEM bases such as O or QK may legally be column 0, matching the existing
    bf16 sparse-prefill kernels.

    The exact column offsets still have to come from CuTe tensors on B300.  For
    scale-fragment addresses, native_mx_tmem_addr_in_capacity() validates the
    masked column payload and intentionally preserves the high sf-id bits that
    tcgen05.mma.kind::mxf8f6f4.block_scale consumes through the idesc.
    */
    FLASH_DEVICE_ASSERT(native_mx_tmem_accumulator_addr_in_capacity(tmem.tmem_base));
    FLASH_DEVICE_ASSERT(native_mx_tmem_accumulator_addr_in_capacity(tmem.o_tmem_d));
    for (int stage = 0; stage < PIPE_STAGES; ++stage) {
        FLASH_DEVICE_ASSERT(native_mx_tmem_accumulator_addr_in_capacity(tmem.qk_tmem_d[stage]));
    }
    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        FLASH_DEVICE_ASSERT(native_mx_tmem_accumulator_addr_in_capacity(tmem.sv_tmem_d[n_chunk]));
        for (int stage = 0; stage < PIPE_STAGES; ++stage) {
            FLASH_DEVICE_ASSERT(tmem.sv_tmem_d[n_chunk] != tmem.qk_tmem_d[stage]);
        }
    }

    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.q_scale_tmem_block_addr[k_chunk]));
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.k_scale_tmem_block_addr[k_chunk]));
        FLASH_DEVICE_ASSERT(tmem.q_scale_tmem_block_addr[k_chunk] != tmem.k_scale_tmem_block_addr[k_chunk]);
        FLASH_DEVICE_ASSERT(native_mx_scale_tmem_block_has_k128_span(tmem.q_scale_tmem_block_addr[k_chunk]));
        FLASH_DEVICE_ASSERT(native_mx_scale_tmem_block_has_k128_span(tmem.k_scale_tmem_block_addr[k_chunk]));

        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.q_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.k_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_tmem_addr_in_capacity(tmem.q_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_tmem_addr_in_capacity(tmem.k_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_scale_tmem_atom_addr_matches_block(
                tmem.q_scale_tmem_block_addr[k_chunk],
                tmem.q_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_scale_tmem_atom_addr_matches_block(
                tmem.k_scale_tmem_block_addr[k_chunk],
                tmem.k_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(
                tmem.q_scale_tmem_addr[k_chunk][k_atom] ==
                native_mx_scale_tmem_atom_addr_from_block(
                    tmem.q_scale_tmem_block_addr[k_chunk],
                    k_atom));
            FLASH_DEVICE_ASSERT(
                tmem.k_scale_tmem_addr[k_chunk][k_atom] ==
                native_mx_scale_tmem_atom_addr_from_block(
                    tmem.k_scale_tmem_block_addr[k_chunk],
                    k_atom));
        }
    }

    FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.s_scale_tmem_block_addr));
    FLASH_DEVICE_ASSERT(native_mx_scale_tmem_block_has_k128_span(tmem.s_scale_tmem_block_addr));
    for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.s_scale_tmem_addr[k_atom]));
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_in_capacity(tmem.s_scale_tmem_addr[k_atom]));
        FLASH_DEVICE_ASSERT(native_mx_scale_tmem_atom_addr_matches_block(
            tmem.s_scale_tmem_block_addr,
            tmem.s_scale_tmem_addr[k_atom]));
        FLASH_DEVICE_ASSERT(
            tmem.s_scale_tmem_addr[k_atom] ==
            native_mx_scale_tmem_atom_addr_from_block(
                tmem.s_scale_tmem_block_addr,
                k_atom));
    }

    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.v_scale_tmem_block_addr[n_chunk]));
        FLASH_DEVICE_ASSERT(tmem.v_scale_tmem_block_addr[n_chunk] != tmem.s_scale_tmem_block_addr);
        FLASH_DEVICE_ASSERT(native_mx_scale_tmem_block_has_k128_span(tmem.v_scale_tmem_block_addr[n_chunk]));

        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.v_scale_tmem_addr[n_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_tmem_addr_in_capacity(tmem.v_scale_tmem_addr[n_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_scale_tmem_atom_addr_matches_block(
                tmem.v_scale_tmem_block_addr[n_chunk],
                tmem.v_scale_tmem_addr[n_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(
                tmem.v_scale_tmem_addr[n_chunk][k_atom] ==
                native_mx_scale_tmem_atom_addr_from_block(
                    tmem.v_scale_tmem_block_addr[n_chunk],
                    k_atom));
        }
    }
}

__device__ __forceinline__ void validate_native_mx_qk_tmem_address_plan(
    const NativeMxTmemAddressPlan& tmem) {
    /*
    QK only needs Q/K scale fragments and the QK accumulator address.  It must
    not require S/V addresses, because S@V descriptor material may be produced
    later in a split-stage pipeline.
    */
    for (int stage = 0; stage < PIPE_STAGES; ++stage) {
        FLASH_DEVICE_ASSERT(native_mx_tmem_accumulator_addr_in_capacity(tmem.qk_tmem_d[stage]));
    }
    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.q_scale_tmem_block_addr[k_chunk]));
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.k_scale_tmem_block_addr[k_chunk]));
        FLASH_DEVICE_ASSERT(tmem.q_scale_tmem_block_addr[k_chunk] != tmem.k_scale_tmem_block_addr[k_chunk]);
        FLASH_DEVICE_ASSERT(native_mx_scale_tmem_block_has_k128_span(tmem.q_scale_tmem_block_addr[k_chunk]));
        FLASH_DEVICE_ASSERT(native_mx_scale_tmem_block_has_k128_span(tmem.k_scale_tmem_block_addr[k_chunk]));

        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.q_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.k_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_tmem_addr_in_capacity(tmem.q_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_tmem_addr_in_capacity(tmem.k_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_scale_tmem_atom_addr_matches_block(
                tmem.q_scale_tmem_block_addr[k_chunk],
                tmem.q_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_scale_tmem_atom_addr_matches_block(
                tmem.k_scale_tmem_block_addr[k_chunk],
                tmem.k_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(
                tmem.q_scale_tmem_addr[k_chunk][k_atom] ==
                native_mx_scale_tmem_atom_addr_from_block(
                    tmem.q_scale_tmem_block_addr[k_chunk],
                    k_atom));
            FLASH_DEVICE_ASSERT(
                tmem.k_scale_tmem_addr[k_chunk][k_atom] ==
                native_mx_scale_tmem_atom_addr_from_block(
                    tmem.k_scale_tmem_block_addr[k_chunk],
                    k_atom));
        }
    }
}

__device__ __forceinline__ void validate_native_mx_compact_sv_tmem_address_plan(
    const NativeMxTmemAddressPlan& tmem,
    int n_chunk) {
    /*
    Compact S@V only needs S fragments plus one V value-dim chunk.  This keeps
    the one-physical-V-chunk pipeline honest: each n_chunk validates exactly the
    SFB addresses it will pass to tcgen05.mma.
    */
    FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);
    FLASH_DEVICE_ASSERT(native_mx_tmem_accumulator_addr_in_capacity(tmem.sv_tmem_d[n_chunk]));
    for (int stage = 0; stage < PIPE_STAGES; ++stage) {
        FLASH_DEVICE_ASSERT(tmem.sv_tmem_d[n_chunk] != tmem.qk_tmem_d[stage]);
    }
    FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.s_scale_tmem_block_addr));
    FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.v_scale_tmem_block_addr[n_chunk]));
    FLASH_DEVICE_ASSERT(tmem.s_scale_tmem_block_addr != tmem.v_scale_tmem_block_addr[n_chunk]);
    FLASH_DEVICE_ASSERT(native_mx_scale_tmem_block_has_k128_span(tmem.s_scale_tmem_block_addr));
    FLASH_DEVICE_ASSERT(native_mx_scale_tmem_block_has_k128_span(tmem.v_scale_tmem_block_addr[n_chunk]));

    for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.s_scale_tmem_addr[k_atom]));
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(tmem.v_scale_tmem_addr[n_chunk][k_atom]));
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_in_capacity(tmem.s_scale_tmem_addr[k_atom]));
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_in_capacity(tmem.v_scale_tmem_addr[n_chunk][k_atom]));
        FLASH_DEVICE_ASSERT(native_mx_scale_tmem_atom_addr_matches_block(
            tmem.s_scale_tmem_block_addr,
            tmem.s_scale_tmem_addr[k_atom]));
        FLASH_DEVICE_ASSERT(native_mx_scale_tmem_atom_addr_matches_block(
            tmem.v_scale_tmem_block_addr[n_chunk],
            tmem.v_scale_tmem_addr[n_chunk][k_atom]));
        FLASH_DEVICE_ASSERT(
            tmem.s_scale_tmem_addr[k_atom] ==
            native_mx_scale_tmem_atom_addr_from_block(
                tmem.s_scale_tmem_block_addr,
                k_atom));
        FLASH_DEVICE_ASSERT(
            tmem.v_scale_tmem_addr[n_chunk][k_atom] ==
            native_mx_scale_tmem_atom_addr_from_block(
                tmem.v_scale_tmem_block_addr[n_chunk],
                k_atom));
    }
}

__device__ __forceinline__ void validate_native_mx_qk_smem_descriptor_material(
    const NativeMxQkSmemDescriptorMaterial& smem) {
    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(smem.q_scale_smem_desc[k_chunk]));
        FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(smem.k_scale_smem_desc[k_chunk]));

        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(smem.q_desc[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(smem.k_desc[k_chunk][k_atom]));
        }
    }
}

__device__ __forceinline__ void validate_native_mx_compact_sv_smem_descriptor_material(
    const NativeMxCompactSvSmemDescriptorMaterial& smem,
    int expected_n_chunk) {
    FLASH_DEVICE_ASSERT(expected_n_chunk >= 0 && expected_n_chunk < MX_SV_N_CHUNKS);
    FLASH_DEVICE_ASSERT(smem.n_chunk == expected_n_chunk);
    FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(smem.s_scale_smem_desc));
    FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(smem.v_scale_smem_desc));

    for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
        FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(smem.s_desc[k_atom]));
        FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(smem.v_desc[k_atom]));
    }
}

__device__ __forceinline__ void validate_native_mx_split_stage_descriptor_material(
    const NativeMxSplitStageDescriptorMaterial& material) {
    /*
    Full production-material validator.

    The granular validators above are intentionally used at issue time so QK
    does not depend on S/V.  This one is stricter and is meant for setup: it
    proves that the whole split-stage descriptor package is ready to run the
    complete native-MX prefill path.
    */
    validate_native_mx_tmem_address_plan(material.tmem);
    const int qk_count = native_mx_physical_stage_count(material.qk_material_stages);
    const int sv_count = native_mx_physical_stage_count(material.sv_material_stages);

    for (int stage = 0; stage < qk_count; ++stage) {
        validate_native_mx_qk_smem_descriptor_material(
            material.qk_smem_by_stage[stage]);
    }

    for (int stage = 0; stage < sv_count; ++stage) {
        for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
            validate_native_mx_compact_sv_smem_descriptor_material(
                material.sv_smem_by_stage[stage][n_chunk],
                n_chunk);
        }
    }
}

__device__ __forceinline__ void validate_native_mx_qk_descriptor_inputs(
    const NativeMxQkDescriptorInputs& raw) {
    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(raw.q_scale_tmem_block_addr[k_chunk]));
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(raw.k_scale_tmem_block_addr[k_chunk]));
        FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(raw.q_scale_smem_desc[k_chunk]));
        FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(raw.k_scale_smem_desc[k_chunk]));

        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(raw.q_desc[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(raw.k_desc[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(raw.qk_block_scaled_idesc[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(raw.q_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(raw.k_scale_tmem_addr[k_chunk][k_atom]));
            FLASH_DEVICE_ASSERT(
                raw.q_scale_tmem_addr[k_chunk][k_atom] ==
                native_mx_scale_tmem_atom_addr_from_block(
                    raw.q_scale_tmem_block_addr[k_chunk],
                    k_atom));
            FLASH_DEVICE_ASSERT(
                raw.k_scale_tmem_addr[k_chunk][k_atom] ==
                native_mx_scale_tmem_atom_addr_from_block(
                    raw.k_scale_tmem_block_addr[k_chunk],
                    k_atom));
        }
    }
}

__device__ __forceinline__ void validate_native_mx_compact_sv_descriptor_inputs(
    const NativeMxCompactSvDescriptorInputs& raw) {
    FLASH_DEVICE_ASSERT(raw.n_chunk >= 0 && raw.n_chunk < MX_SV_N_CHUNKS);
    FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(raw.s_scale_tmem_block_addr));
    FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(raw.v_scale_tmem_block_addr));
    FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(raw.s_scale_smem_desc));
    FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(raw.v_scale_smem_desc));

    for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
        FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(raw.s_desc[k_atom]));
        FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(raw.v_desc[k_atom]));
        FLASH_DEVICE_ASSERT(native_mx_descriptor_is_bound(raw.sv_block_scaled_idesc[k_atom]));
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(raw.s_scale_tmem_addr[k_atom]));
        FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(raw.v_scale_tmem_addr[k_atom]));
        FLASH_DEVICE_ASSERT(
            raw.s_scale_tmem_addr[k_atom] ==
            native_mx_scale_tmem_atom_addr_from_block(
                raw.s_scale_tmem_block_addr,
                k_atom));
        FLASH_DEVICE_ASSERT(
            raw.v_scale_tmem_addr[k_atom] ==
            native_mx_scale_tmem_atom_addr_from_block(
                raw.v_scale_tmem_block_addr,
                k_atom));
    }
}

__device__ __forceinline__ uint64_t make_native_mx_qk_block_scaled_idesc(
    uint32_t q_scale_tmem_addr,
    uint32_t k_scale_tmem_addr) {
    /*
    QK uses A=Q and B=K as K-major operands.  The returned idesc carries the
    scale-fragment selector bits derived from the two TMEM scale addresses.
    */
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    return cute::UMMA::make_runtime_instr_desc_block_scaled<
        cute::float_e4m3_t,
        cute::float_e4m3_t,
        float,
        cute::float_ue8m0_t,
        QK_M,
        QK_N,
        cute::UMMA::Major::K,
        cute::UMMA::Major::K>(q_scale_tmem_addr, k_scale_tmem_addr);
#else
    (void)q_scale_tmem_addr;
    (void)k_scale_tmem_addr;
    return 0;
#endif
}

__device__ __forceinline__ uint64_t make_native_mx_sv_block_scaled_idesc(
    uint32_t s_scale_tmem_addr,
    uint32_t v_scale_tmem_addr) {
    /*
    S@V uses A=S as K-major and B=V as MN-major, matching the bf16 prefill O
    path's logical V operand orientation.
    */
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    return cute::UMMA::make_runtime_instr_desc_block_scaled<
        cute::float_e4m3_t,
        cute::float_e4m3_t,
        float,
        cute::float_ue8m0_t,
        SV_M,
        MX_SV_N_TILE,
        cute::UMMA::Major::K,
        cute::UMMA::Major::MN>(s_scale_tmem_addr, v_scale_tmem_addr);
#else
    (void)s_scale_tmem_addr;
    (void)v_scale_tmem_addr;
    return 0;
#endif
}

__device__ __forceinline__ NativeMxQkDescriptorInputs bind_native_mx_qk_descriptor_inputs(
    const NativeMxTmemAddressPlan& tmem,
    const NativeMxQkSmemDescriptorMaterial& smem,
    int stage) {
    FLASH_DEVICE_ASSERT(stage >= 0 && stage < PIPE_STAGES);
    validate_native_mx_qk_tmem_address_plan(tmem);
    validate_native_mx_qk_smem_descriptor_material(smem);

    NativeMxQkDescriptorInputs raw{};
    raw.qk_tmem_d = tmem.qk_tmem_d[stage];

    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        raw.q_scale_tmem_block_addr[k_chunk] = tmem.q_scale_tmem_block_addr[k_chunk];
        raw.k_scale_tmem_block_addr[k_chunk] = tmem.k_scale_tmem_block_addr[k_chunk];
        raw.q_scale_smem_desc[k_chunk] = smem.q_scale_smem_desc[k_chunk];
        raw.k_scale_smem_desc[k_chunk] = smem.k_scale_smem_desc[k_chunk];

        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            raw.q_desc[k_chunk][k_atom] = smem.q_desc[k_chunk][k_atom];
            raw.k_desc[k_chunk][k_atom] = smem.k_desc[k_chunk][k_atom];
            raw.q_scale_tmem_addr[k_chunk][k_atom] =
                tmem.q_scale_tmem_addr[k_chunk][k_atom];
            raw.k_scale_tmem_addr[k_chunk][k_atom] =
                tmem.k_scale_tmem_addr[k_chunk][k_atom];
            raw.qk_block_scaled_idesc[k_chunk][k_atom] =
                make_native_mx_qk_block_scaled_idesc(
                    raw.q_scale_tmem_addr[k_chunk][k_atom],
                    raw.k_scale_tmem_addr[k_chunk][k_atom]);
        }
    }

    validate_native_mx_qk_descriptor_inputs(raw);
    return raw;
}

__device__ __forceinline__ NativeMxCompactSvDescriptorInputs
bind_native_mx_compact_sv_descriptor_inputs(
    const NativeMxTmemAddressPlan& tmem,
    const NativeMxCompactSvSmemDescriptorMaterial& smem,
    int n_chunk) {
    FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);
    validate_native_mx_compact_sv_tmem_address_plan(tmem, n_chunk);
    validate_native_mx_compact_sv_smem_descriptor_material(smem, n_chunk);

    NativeMxCompactSvDescriptorInputs raw{};
    raw.n_chunk = n_chunk;
    raw.sv_tmem_d = tmem.sv_tmem_d[n_chunk];
    raw.s_scale_tmem_block_addr = tmem.s_scale_tmem_block_addr;
    raw.v_scale_tmem_block_addr = tmem.v_scale_tmem_block_addr[n_chunk];
    raw.s_scale_smem_desc = smem.s_scale_smem_desc;
    raw.v_scale_smem_desc = smem.v_scale_smem_desc;

    for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
        raw.s_desc[k_atom] = smem.s_desc[k_atom];
        raw.v_desc[k_atom] = smem.v_desc[k_atom];
        raw.s_scale_tmem_addr[k_atom] = tmem.s_scale_tmem_addr[k_atom];
        raw.v_scale_tmem_addr[k_atom] = tmem.v_scale_tmem_addr[n_chunk][k_atom];
        raw.sv_block_scaled_idesc[k_atom] =
            make_native_mx_sv_block_scaled_idesc(
                raw.s_scale_tmem_addr[k_atom],
                raw.v_scale_tmem_addr[k_atom]);
    }

    validate_native_mx_compact_sv_descriptor_inputs(raw);
    return raw;
}

struct NativeMxSplitStageKernelStorage {
    /*
    Concrete split-stage storage contract for the theoretical kernel.

    The pointer view owns the actual operand locations.  The descriptor arrays
    are intentionally passed separately because final swizzled SMEM descriptors,
    TMEM accumulator addresses, and scale-fragment addresses must come from the
    CuTe/CUTLASS layout layer verified on B300.
    */
    NativeMxSplitStageStorageView ptrs;

    // Already-bound descriptor path, useful while porting from earlier shells.
    const NativeMxQkDescriptorInputs* qk_descs_by_stage;          // optional [PIPE_STAGES]
    const NativeMxCompactSvDescriptorInputs* sv_descs_by_stage;   // optional [PIPE_STAGES, MX_SV_N_CHUNKS]

    // Production-shaped path: CuTe creates SMEM material and TMEM addresses,
    // this shell binds them into the exact raw inputs consumed by issue code.
    const NativeMxTmemAddressPlan* tmem_plan;                     // optional
    const NativeMxQkSmemDescriptorMaterial* qk_smem_by_stage;     // optional [PIPE_STAGES]
    const NativeMxCompactSvSmemDescriptorMaterial* sv_smem_by_stage; // optional [PIPE_STAGES, MX_SV_N_CHUNKS]
    int qk_material_stages;                                       // 0 means PIPE_STAGES
    int sv_material_stages;                                       // 0 means PIPE_STAGES
    const NativeMxSplitStageBarrierPlan* barriers;                 // optional

    __device__ __forceinline__ NativeFp8TilePointers persistent_tile_ptrs() const {
        return ptrs.persistent_tile_ptrs();
    }

    __device__ __forceinline__ NativeFp8TilePointers qk_stage_tile_ptrs(int stage) const {
        return ptrs.qk_stage_tile_ptrs(stage);
    }

    __device__ __forceinline__ NativeFp8TilePointers sv_stage_s_tile_ptrs(int stage) const {
        return ptrs.sv_stage_s_tile_ptrs(stage);
    }

    __device__ __forceinline__ NativeMxValueChunkPointers sv_stage_v_chunk_ptrs(
        int stage,
        int n_chunk) const {
        return ptrs.sv_stage_v_chunk_ptrs(stage, n_chunk);
    }

    __device__ __forceinline__ int qk_material_slot(int stage) const {
        FLASH_DEVICE_ASSERT(stage >= 0 && stage < PIPE_STAGES);
        return stage % native_mx_physical_stage_count(qk_material_stages);
    }

    __device__ __forceinline__ int sv_material_slot(int stage) const {
        FLASH_DEVICE_ASSERT(stage >= 0 && stage < PIPE_STAGES);
        return stage % native_mx_physical_stage_count(sv_material_stages);
    }

    __device__ __forceinline__ void require_barriers_for_ptx() const {
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
        FLASH_DEVICE_ASSERT(barriers != nullptr);
#endif
    }

    __device__ __forceinline__ int qk_barrier_slot(int stage) const {
        FLASH_DEVICE_ASSERT(barriers != nullptr);
        FLASH_DEVICE_ASSERT(
            native_mx_physical_stage_count(barriers->qk_physical_stages) ==
            native_mx_physical_stage_count(ptrs.qk_physical_stages));
        return ptrs.qk_stage_slot(stage);
    }

    __device__ __forceinline__ int sv_barrier_slot(int stage) const {
        FLASH_DEVICE_ASSERT(barriers != nullptr);
        FLASH_DEVICE_ASSERT(
            native_mx_physical_stage_count(barriers->sv_physical_stages) ==
            native_mx_physical_stage_count(ptrs.sv_physical_stages));
        return ptrs.sv_stage_slot(stage);
    }

    /*
    Synchronization hooks for the split-stage pipeline.

    When `barriers` is null these remain no-ops for CUDA-free emulation.  When a
    NativeMxSplitStageBarrierPlan is provided, they execute the same style of
    mbarrier wait/arrive protocol used by the existing SM100 sparse-prefill
    kernels.
    */
    __device__ __forceinline__ void signal_q_ready() const {
        if (barriers != nullptr) {
            native_mx_arrive_shared_barrier(barriers->q_ready);
        }
    }
    __device__ __forceinline__ void wait_q_ready() const {
        require_barriers_for_ptx();
        if (barriers != nullptr) {
            native_mx_wait_shared_barrier(barriers->q_ready, 0);
        }
    }

    __device__ __forceinline__ void wait_qk_stage_free(int block, int stage) const {
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        if (barriers != nullptr &&
            native_mx_reuses_physical_stage(block, ptrs.qk_physical_stages)) {
            const int slot = qk_barrier_slot(stage);
            const int phase =
                native_mx_reuse_phase_for_block(block, ptrs.qk_physical_stages);
            native_mx_wait_tc_barrier(
                native_mx_stage_barrier(barriers->qk_stage_free, slot),
                phase);
        }
    }
    __device__ __forceinline__ void signal_qk_stage_free(int block, int stage) const {
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        if (barriers != nullptr) {
            const int slot = qk_barrier_slot(stage);
            native_mx_arrive_tc_barrier(
                native_mx_stage_barrier(barriers->qk_stage_free, slot));
        }
    }
    __device__ __forceinline__ void signal_k_ready(int block, int stage) const {
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        if (barriers != nullptr) {
            const int slot = qk_barrier_slot(stage);
            native_mx_arrive_shared_barrier(
                native_mx_stage_barrier(barriers->k_ready, slot));
        }
    }
    __device__ __forceinline__ void wait_k_ready(int block, int stage) const {
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        require_barriers_for_ptx();
        if (barriers != nullptr) {
            const int slot = qk_barrier_slot(stage);
            const int phase =
                native_mx_ready_phase_for_block(block, ptrs.qk_physical_stages);
            native_mx_wait_shared_barrier(
                native_mx_stage_barrier(barriers->k_ready, slot),
                phase);
        }
    }

    __device__ __forceinline__ void signal_qk_ready(int block, int stage) const {
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        if (barriers != nullptr) {
            const int slot = qk_barrier_slot(stage);
            native_mx_arrive_tc_barrier(
                native_mx_stage_barrier(barriers->qk_ready, slot));
        }
    }
    __device__ __forceinline__ void wait_qk_ready(int block, int stage) const {
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        require_barriers_for_ptx();
        if (barriers != nullptr) {
            const int slot = qk_barrier_slot(stage);
            const int phase =
                native_mx_ready_phase_for_block(block, ptrs.qk_physical_stages);
            native_mx_wait_tc_barrier(
                native_mx_stage_barrier(barriers->qk_ready, slot),
                phase);
        }
    }

    __device__ __forceinline__ void signal_s_ready(int block, int stage) const {
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        if (barriers != nullptr) {
            const int slot = sv_barrier_slot(stage);
            native_mx_arrive_shared_barrier(
                native_mx_stage_barrier(barriers->s_ready, slot));
        }
    }
    __device__ __forceinline__ void wait_s_ready(int block, int stage) const {
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        require_barriers_for_ptx();
        if (barriers != nullptr) {
            const int slot = sv_barrier_slot(stage);
            const int phase =
                native_mx_ready_phase_for_block(block, ptrs.sv_physical_stages);
            native_mx_wait_shared_barrier(
                native_mx_stage_barrier(barriers->s_ready, slot),
                phase);
        }
    }

    __device__ __forceinline__ void wait_sv_stage_free(int block, int stage) const {
        /*
        S_mx/S_scale live in the same physical SV stage as the compact V chunk.
        Before Softmax writes S for a reused physical stage, the previous block's
        final S@V value chunk must be complete.
        */
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        if (barriers != nullptr &&
            native_mx_reuses_physical_stage(block, ptrs.sv_physical_stages)) {
            const int slot = sv_barrier_slot(stage);
            const int phase =
                native_mx_reuse_phase_for_block(block, ptrs.sv_physical_stages);
            native_mx_wait_tc_barrier(
                native_mx_sv_chunk_barrier(
                    barriers->sv_chunk_done,
                    slot,
                    MX_SV_N_CHUNKS - 1),
                phase);
        }
    }

    __device__ __forceinline__ void wait_sv_stage_free_for_chunk(
        int block,
        int stage,
        int n_chunk) const {
        /*
        Single-value-chunk bring-up path.

        A CTA owns exactly one logical n_chunk and recomputes QK/softmax for
        that chunk.  It should wait for the previous block's S@V completion for
        the same logical chunk, not for n_chunk-1 in the current block.
        */
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);
        if (barriers != nullptr &&
            native_mx_reuses_physical_stage(block, ptrs.sv_physical_stages)) {
            const int slot = sv_barrier_slot(stage);
            const int phase =
                native_mx_reuse_phase_for_block(block, ptrs.sv_physical_stages);
            native_mx_wait_tc_barrier(
                native_mx_sv_chunk_barrier(barriers->sv_chunk_done, slot, n_chunk),
                phase);
        }
    }

    __device__ __forceinline__ void wait_v_chunk_free(int block, int stage, int n_chunk) const {
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);
        if (barriers == nullptr) {
            return;
        }

        const int slot = sv_barrier_slot(stage);
        if (n_chunk > 0) {
            const int phase =
                native_mx_ready_phase_for_block(block, ptrs.sv_physical_stages);
            native_mx_wait_tc_barrier(
                native_mx_sv_chunk_barrier(
                    barriers->sv_chunk_done,
                    slot,
                    n_chunk - 1),
                phase);
        } else if (native_mx_reuses_physical_stage(block, ptrs.sv_physical_stages)) {
            const int phase =
                native_mx_reuse_phase_for_block(block, ptrs.sv_physical_stages);
            native_mx_wait_tc_barrier(
                native_mx_sv_chunk_barrier(
                    barriers->sv_chunk_done,
                    slot,
                    MX_SV_N_CHUNKS - 1),
                phase);
        }
    }
    __device__ __forceinline__ void signal_v_chunk_free(int block, int stage, int n_chunk) const {
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);
        if (barriers != nullptr) {
            const int slot = sv_barrier_slot(stage);
            native_mx_arrive_tc_barrier(
                native_mx_sv_chunk_barrier(barriers->sv_chunk_done, slot, n_chunk));
        }
    }
    __device__ __forceinline__ void signal_v_chunk_ready(int block, int stage, int n_chunk) const {
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);
        if (barriers != nullptr) {
            const int slot = sv_barrier_slot(stage);
            native_mx_arrive_shared_barrier(
                native_mx_sv_chunk_barrier(barriers->v_chunk_ready, slot, n_chunk));
        }
    }
    __device__ __forceinline__ void wait_v_chunk_ready(int block, int stage, int n_chunk) const {
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);
        require_barriers_for_ptx();
        if (barriers != nullptr) {
            const int slot = sv_barrier_slot(stage);
            const int phase =
                native_mx_ready_phase_for_block(block, ptrs.sv_physical_stages);
            native_mx_wait_shared_barrier(
                native_mx_sv_chunk_barrier(barriers->v_chunk_ready, slot, n_chunk),
                phase);
        }
    }

    __device__ __forceinline__ void wait_sv_chunk_done(
        int block,
        int stage,
        int n_chunk) const {
        FLASH_DEVICE_ASSERT(stage == block % PIPE_STAGES);
        FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);
        require_barriers_for_ptx();
        if (barriers != nullptr) {
            const int slot = sv_barrier_slot(stage);
            const int phase =
                native_mx_ready_phase_for_block(block, ptrs.sv_physical_stages);
            native_mx_wait_tc_barrier(
                native_mx_sv_chunk_barrier(barriers->sv_chunk_done, slot, n_chunk),
                phase);
        }
    }

    __device__ __forceinline__ void signal_final_sv_done() const {
        if (barriers != nullptr) {
            native_mx_arrive_shared_barrier(barriers->final_sv_done);
        }
    }
    __device__ __forceinline__ void wait_final_sv_done() const {
        if (barriers != nullptr) {
            native_mx_wait_shared_barrier(barriers->final_sv_done, 0);
        }
    }

    __device__ __forceinline__ uint32_t qk_stage_tmem_d(int stage) const {
        FLASH_DEVICE_ASSERT(stage >= 0 && stage < PIPE_STAGES);
        if (tmem_plan != nullptr) {
            return tmem_plan->qk_tmem_d[qk_material_slot(stage)];
        }

        FLASH_DEVICE_ASSERT(qk_descs_by_stage != nullptr);
        return qk_descs_by_stage[qk_material_slot(stage)].qk_tmem_d;
    }

    __device__ __forceinline__ NativeMxQkDescriptorInputs qk_stage_raw_mma_descs(
        int stage) const {
        FLASH_DEVICE_ASSERT(stage >= 0 && stage < PIPE_STAGES);
        if (qk_descs_by_stage == nullptr) {
            FLASH_DEVICE_ASSERT(tmem_plan != nullptr);
            FLASH_DEVICE_ASSERT(qk_smem_by_stage != nullptr);
            return bind_native_mx_qk_descriptor_inputs(
                *tmem_plan,
                qk_smem_by_stage[qk_material_slot(stage)],
                qk_material_slot(stage));
        }

        NativeMxQkDescriptorInputs raw = qk_descs_by_stage[qk_material_slot(stage)];
        validate_native_mx_qk_descriptor_inputs(raw);
        return raw;
    }

    __device__ __forceinline__ NativeMxCompactSvDescriptorInputs sv_stage_raw_compact_descs(
        int stage,
        int n_chunk) const {
        FLASH_DEVICE_ASSERT(stage >= 0 && stage < PIPE_STAGES);
        FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);
        if (sv_descs_by_stage == nullptr) {
            FLASH_DEVICE_ASSERT(tmem_plan != nullptr);
            FLASH_DEVICE_ASSERT(sv_smem_by_stage != nullptr);
            return bind_native_mx_compact_sv_descriptor_inputs(
                *tmem_plan,
                sv_smem_by_stage[sv_material_slot(stage) * MX_SV_N_CHUNKS + n_chunk],
                n_chunk);
        }

        FLASH_DEVICE_ASSERT(sv_descs_by_stage != nullptr);
        NativeMxCompactSvDescriptorInputs raw =
            sv_descs_by_stage[sv_material_slot(stage) * MX_SV_N_CHUNKS + n_chunk];
        validate_native_mx_compact_sv_descriptor_inputs(raw);
        return raw;
    }
};

__device__ __forceinline__ NativeMxSplitStageKernelStorage
make_native_mx_bound_split_stage_storage(
    NativeMxSplitStageStorageView ptrs,
    const NativeMxQkDescriptorInputs* qk_descs_by_stage,
    const NativeMxCompactSvDescriptorInputs* sv_descs_by_stage,
    int qk_material_stages = PIPE_STAGES,
    int sv_material_stages = PIPE_STAGES,
    const NativeMxSplitStageBarrierPlan* barriers = nullptr) {
    /*
    Compatibility constructor for already-bound descriptor inputs.

    This is the old research-shell shape: callers provide raw descriptor inputs
    that already include SMEM descriptors, TMEM scale-fragment addresses, and
    block-scaled instruction descriptors.
    */
    FLASH_DEVICE_ASSERT(native_mx_physical_stage_count(qk_material_stages) ==
                        native_mx_physical_stage_count(ptrs.qk_physical_stages));
    FLASH_DEVICE_ASSERT(native_mx_physical_stage_count(sv_material_stages) ==
                        native_mx_physical_stage_count(ptrs.sv_physical_stages));
    if (barriers != nullptr) {
        FLASH_DEVICE_ASSERT(native_mx_physical_stage_count(barriers->qk_physical_stages) ==
                            native_mx_physical_stage_count(ptrs.qk_physical_stages));
        FLASH_DEVICE_ASSERT(native_mx_physical_stage_count(barriers->sv_physical_stages) ==
                            native_mx_physical_stage_count(ptrs.sv_physical_stages));
    }

    NativeMxSplitStageKernelStorage storage{};
    storage.ptrs = ptrs;
    storage.qk_descs_by_stage = qk_descs_by_stage;
    storage.sv_descs_by_stage = sv_descs_by_stage;
    storage.tmem_plan = nullptr;
    storage.qk_smem_by_stage = nullptr;
    storage.sv_smem_by_stage = nullptr;
    storage.qk_material_stages = qk_material_stages;
    storage.sv_material_stages = sv_material_stages;
    storage.barriers = barriers;
    return storage;
}

__device__ __forceinline__ NativeMxSplitStageKernelStorage
make_native_mx_material_split_stage_storage(
    NativeMxSplitStageStorageView ptrs,
    const NativeMxTmemAddressPlan* tmem_plan,
    const NativeMxQkSmemDescriptorMaterial* qk_smem_by_stage,
    const NativeMxCompactSvSmemDescriptorMaterial* sv_smem_by_stage,
    int qk_material_stages = PIPE_STAGES,
    int sv_material_stages = PIPE_STAGES,
    const NativeMxSplitStageBarrierPlan* barriers = nullptr) {
    /*
    Production-shaped constructor.

    Callers provide the two things a real SM100 kernel can derive locally:
      - TMEM addresses from Allocator1Sm plus CuTe tCtSFA/tCtSFB fragments
      - SMEM descriptors from the staged Q/K/S/V and scale staging layouts

    The storage accessors then bind these pieces into atom-level raw inputs at
    the point of issue.  This keeps V3.2->MX scale ownership in this file while
    leaving exact CuTe descriptor creation to the final B300-verified layer.
    */
    FLASH_DEVICE_ASSERT(native_mx_physical_stage_count(qk_material_stages) ==
                        native_mx_physical_stage_count(ptrs.qk_physical_stages));
    FLASH_DEVICE_ASSERT(native_mx_physical_stage_count(sv_material_stages) ==
                        native_mx_physical_stage_count(ptrs.sv_physical_stages));
    if (barriers != nullptr) {
        FLASH_DEVICE_ASSERT(native_mx_physical_stage_count(barriers->qk_physical_stages) ==
                            native_mx_physical_stage_count(ptrs.qk_physical_stages));
        FLASH_DEVICE_ASSERT(native_mx_physical_stage_count(barriers->sv_physical_stages) ==
                            native_mx_physical_stage_count(ptrs.sv_physical_stages));
    }

    NativeMxSplitStageKernelStorage storage{};
    storage.ptrs = ptrs;
    storage.qk_descs_by_stage = nullptr;
    storage.sv_descs_by_stage = nullptr;
    storage.tmem_plan = tmem_plan;
    storage.qk_smem_by_stage = qk_smem_by_stage;
    storage.sv_smem_by_stage = sv_smem_by_stage;
    storage.qk_material_stages = qk_material_stages;
    storage.sv_material_stages = sv_material_stages;
    storage.barriers = barriers;
    return storage;
}

__device__ __forceinline__ NativeMxSplitStageKernelStorage
make_native_mx_material_split_stage_storage(
    NativeMxSplitStageStorageView ptrs,
    const NativeMxSplitStageDescriptorMaterial* material,
    const NativeMxSplitStageBarrierPlan* barriers = nullptr) {
    /*
    Preferred production-shaped constructor.

    This makes the setup boundary harder to misuse: instead of passing three
    unrelated pointers, the caller passes the single descriptor package that
    owns the TMEM plan and every stage/chunk SMEM descriptor.
    */
    FLASH_DEVICE_ASSERT(material != nullptr);
    validate_native_mx_split_stage_descriptor_material(*material);

    return make_native_mx_material_split_stage_storage(
        ptrs,
        &material->tmem,
        &material->qk_smem_by_stage[0],
        &material->sv_smem_by_stage[0][0],
        native_mx_physical_stage_count(material->qk_material_stages),
        native_mx_physical_stage_count(material->sv_material_stages),
        barriers);
}

__device__ __forceinline__ int warpgroup_thread_idx() {
    return threadIdx.x & (WARP_GROUP_THREADS - 1);
}

__device__ __forceinline__ void native_mx_sync_scale_stage_written() {
    /*
    The MMA warpgroup materializes one 512B scale-factor block cooperatively:
    128 lanes write different bytes of q/k/s/v_scale_smem_stage, then one elected
    lane issues tcgen05.cp to move that block into the SFA/SFB TMEM fragment.

    Without this barrier the elected lane could start UTCCP while other lanes are
    still filling the staging block.  CUTLASS block-scaled kernels receive scale
    stages from a producer pipeline that has already synchronized; this research
    shell builds the stage locally, so it needs the same ordering point here.
    */
    fence_view_async_shared();
    NamedBarrier::arrive_and_wait(
        WARP_GROUP_THREADS,
        NativeMxNamedBarrierId::ScaleStageWritten);
}

__device__ __forceinline__ V32TokenParts split_token(V32TokenRef token) {
    return {
        token.base,
        reinterpret_cast<const float*>(token.base + V32_BYTES_NOPE),
        reinterpret_cast<const bf16*>(token.base + V32_BYTES_NOPE + V32_BYTES_SCALE),
    };
}

__device__ __forceinline__ V32TokenParts maybe_split_token(V32TokenRef token) {
    return token.valid ? split_token(token) : V32TokenParts{nullptr, nullptr, nullptr};
}

__device__ __forceinline__ V32TokenRef resolve_v32_token(
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
    return {
        kv_bytes +
            static_cast<int64_t>(block_idx) * params.stride_kv_block +
            static_cast<int64_t>(idx_in_block) * V32_BYTES_PER_TOKEN,
        true,
    };
}

__device__ __forceinline__ float load_v32_k_value_from_parts(
    V32TokenRef token,
    V32TokenParts parts,
    int dim) {
    if (!token.valid || dim >= D_QK) {
        return 0.0f;
    }

    if (dim < D_NOPE) {
        const auto* raw = reinterpret_cast<const __nv_fp8_e4m3*>(parts.nope);
        const float scale = __ldg(parts.scales + dim / V32_SCALE_GROUP);
        return static_cast<float>(raw[dim]) * scale;
    }
    return float(__ldg(parts.rope + (dim - D_NOPE)));
}

__device__ __forceinline__ float load_v32_k_value(V32TokenRef token, int dim) {
    return load_v32_k_value_from_parts(token, maybe_split_token(token), dim);
}

__device__ __forceinline__ float load_v32_v_value_from_parts(
    V32TokenRef token,
    V32TokenParts parts,
    int dim) {
    if (!token.valid || dim >= D_V) {
        return 0.0f;
    }

    const auto* raw = reinterpret_cast<const __nv_fp8_e4m3*>(parts.nope);
    const float scale = __ldg(parts.scales + dim / V32_SCALE_GROUP);
    return static_cast<float>(raw[dim]) * scale;
}

__device__ __forceinline__ float load_v32_v_value(V32TokenRef token, int dim) {
    return load_v32_v_value_from_parts(token, maybe_split_token(token), dim);
}

__device__ __forceinline__ float load_v32_v_value_with_scale_from_parts(
    V32TokenRef token,
    V32TokenParts parts,
    int dim,
    float scale) {
    if (!token.valid || dim >= D_V) {
        return 0.0f;
    }

    const auto* raw = reinterpret_cast<const __nv_fp8_e4m3*>(parts.nope);
    return static_cast<float>(raw[dim]) * scale;
}

__device__ __forceinline__ __nv_fp8_e4m3 load_v32_nope_raw_fp8_from_parts(
    V32TokenRef token,
    V32TokenParts parts,
    int dim) {
    if (!token.valid || dim >= D_NOPE) {
        return __nv_fp8_e4m3(0.0f);
    }

    const auto* raw = reinterpret_cast<const __nv_fp8_e4m3*>(parts.nope);
    return __ldg(raw + dim);
}

__device__ __forceinline__ __nv_fp8_e4m3 load_v32_nope_raw_fp8(
    V32TokenRef token,
    int dim) {
    return load_v32_nope_raw_fp8_from_parts(token, maybe_split_token(token), dim);
}

__device__ __forceinline__ float load_v32_nope_group_scale_from_parts(
    V32TokenRef token,
    V32TokenParts parts,
    int v32_scale_group) {
    if (!token.valid || v32_scale_group < 0 || v32_scale_group >= V32_NUM_SCALES) {
        return 1.0f;
    }

    return __ldg(parts.scales + v32_scale_group);
}

__device__ __forceinline__ float load_v32_nope_group_scale(
    V32TokenRef token,
    int v32_scale_group) {
    return load_v32_nope_group_scale_from_parts(
        token,
        maybe_split_token(token),
        v32_scale_group);
}

__device__ __forceinline__ float load_q_value(
    const SparseAttnFwdParams& params,
    int s_q_idx,
    int h_idx,
    int dim) {
    if (h_idx >= params.h_q || dim >= D_QK) {
        return 0.0f;
    }

    const bf16* q =
        params.q +
        static_cast<int64_t>(s_q_idx) * params.stride_q_s_q +
        static_cast<int64_t>(h_idx) * params.stride_q_h_q;
    return float(__ldg(q + dim));
}

__device__ __forceinline__ float ceil_to_power_of_two(float value) {
    const float safe_value = fmaxf(value, MIN_MX_SCALE);
    return exp2f(ceilf(log2f(safe_value)));
}

__device__ __forceinline__ uint8 encode_mx_scale_ue8m0(float scale) {
    /*
    Blackwell mx_float8_t uses float_ue8m0_t scale factors.  CUDA exposes the
    same encoding path used by the existing SM100 decode code:

      __nv_cvt_float2_to_e8m0x2(...)

    We encode a pair and keep the first byte.  The input scale is already
    rounded to a power of two, so cudaRoundZero does not change it.
    */
    fp8_e8m0 encoded[2];
    *reinterpret_cast<__nv_fp8x2_storage_t*>(encoded) =
        __nv_cvt_float2_to_e8m0x2(float2{scale, scale}, __NV_NOSAT, cudaRoundZero);
    return *reinterpret_cast<uint8*>(encoded);
}

__device__ __forceinline__ MxScale make_mx_scale_from_real_scale(float scale) {
    const float pow2_scale = ceil_to_power_of_two(scale);
    return {pow2_scale, encode_mx_scale_ue8m0(pow2_scale)};
}

__device__ __forceinline__ MxScale make_mx_scale_from_amax(float amax) {
    // Scale maps dequantized real values to e4m3 payload values:
    //
    //   real_value ~= e4m3_payload * scale
    //
    // MX scale factors are e8m0, so the real scale is rounded up to a power of
    // two.  This mirrors tests/quant.py::_cast_scale_inv_to_ue8m0().
    return make_mx_scale_from_real_scale(amax / E4M3_MAX_FINITE);
}

__device__ __forceinline__ __nv_fp8_e4m3 quantize_to_mx_fp8(float value, MxScale scale) {
    const float scaled = value / scale.value;
    return __nv_fp8_e4m3(scaled);
}

__device__ __forceinline__ float dequant_from_mx_fp8(__nv_fp8_e4m3 value, MxScale scale) {
    return static_cast<float>(value) * scale.value;
}

__device__ __forceinline__ int row_major(int row, int col, int ld) {
    return row * ld + col;
}

#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
template<int MN, int K, int SWIZZLE, cute::UMMA::Major MAJOR>
__device__ __forceinline__ int native_mx_canonical_operand_offset(
    int mn_index,
    int k_index) {
    auto layout =
        ku::make_umma_canonical_layout<MAJOR, MN, K, SWIZZLE, uint8>();
    return layout(mn_index, k_index);
}
#endif

__device__ __forceinline__ int native_mx_q_operand_offset(int h, int dim) {
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    return native_mx_canonical_operand_offset<
        QK_M,
        D_QK_PAD,
        128,
        cute::UMMA::Major::K>(h, dim);
#else
    return row_major(h, dim, D_QK_PAD);
#endif
}

__device__ __forceinline__ int native_mx_k_operand_offset(int local_token, int dim) {
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    return native_mx_canonical_operand_offset<
        QK_N,
        D_QK_PAD,
        128,
        cute::UMMA::Major::K>(local_token, dim);
#else
    return row_major(local_token, dim, D_QK_PAD);
#endif
}

__device__ __forceinline__ int native_mx_s_operand_offset(int h, int local_token) {
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    return native_mx_canonical_operand_offset<
        SV_M,
        SV_K,
        128,
        cute::UMMA::Major::K>(h, local_token);
#else
    return row_major(h, local_token, SV_K);
#endif
}

__device__ __forceinline__ int native_mx_v_full_operand_offset(
    int local_token,
    int dim) {
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    return native_mx_canonical_operand_offset<
        SV_N,
        SV_K,
        128,
        cute::UMMA::Major::MN>(dim, local_token);
#else
    return row_major(local_token, dim, SV_N);
#endif
}

__device__ __forceinline__ int native_mx_v_compact_operand_offset(
    int local_token,
    int local_dim) {
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    return native_mx_canonical_operand_offset<
        MX_SV_N_TILE,
        SV_K,
        128,
        cute::UMMA::Major::MN>(local_dim, local_token);
#else
    return row_major(local_token, local_dim, MX_SV_N_TILE);
#endif
}

#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
template<int MN, int K, int SWIZZLE, cute::UMMA::Major MAJOR>
__device__ __forceinline__ uint64_t native_mx_make_operand_smem_desc(
    uint8* base,
    int mn_start,
    int k_start) {
    /*
    Build the same canonical UMMA descriptor family used by the existing SM100
    bf16 kernels, but for 8-bit MX payload storage.  The descriptor offset is
    derived from the actual CuTe layout instead of a hand-written row-major
    formula; this is what keeps the V3.2->MX transcode stores and tcgen05.mma
    operands in the same address space.

    This covers Q/K/S/V payload descriptors.  Scale-factor SMEM descriptors are
    a different UTCCP source tensor and remain explicit material fields.
    */
    auto layout =
        ku::make_umma_canonical_layout<MAJOR, MN, K, SWIZZLE, uint8>();
    auto tensor = cute::make_tensor(cute::make_smem_ptr(base), layout);
    cute::UMMA::SmemDescriptor desc =
        cute::UMMA::make_umma_desc<MAJOR>(tensor);
    const int element_offset = layout(mn_start, k_start);
    FLASH_DEVICE_ASSERT((element_offset % 16) == 0);
    return static_cast<uint64_t>(desc + (element_offset / 16));
}

__device__ __forceinline__ uint64_t native_mx_make_q_smem_desc(
    NativeFp8TilePointers tile,
    int k_start) {
    return native_mx_make_operand_smem_desc<
        QK_M,
        D_QK_PAD,
        128,
        cute::UMMA::Major::K>(
            reinterpret_cast<uint8*>(tile.q_mx),
            0,
            k_start);
}

__device__ __forceinline__ uint64_t native_mx_make_k_smem_desc(
    NativeFp8TilePointers tile,
    int k_start) {
    return native_mx_make_operand_smem_desc<
        QK_N,
        D_QK_PAD,
        128,
        cute::UMMA::Major::K>(
            reinterpret_cast<uint8*>(tile.k_mx),
            0,
            k_start);
}

__device__ __forceinline__ uint64_t native_mx_make_s_smem_desc(
    NativeFp8TilePointers tile,
    int k_start) {
    return native_mx_make_operand_smem_desc<
        SV_M,
        SV_K,
        128,
        cute::UMMA::Major::K>(
            reinterpret_cast<uint8*>(tile.s_mx),
            0,
            k_start);
}

__device__ __forceinline__ uint64_t native_mx_make_v_compact_smem_desc(
    NativeMxValueChunkPointers v_chunk,
    int k_start) {
    return native_mx_make_operand_smem_desc<
        MX_SV_N_TILE,
        SV_K,
        128,
        cute::UMMA::Major::MN>(
            reinterpret_cast<uint8*>(v_chunk.v_mx),
            0,
            k_start);
}
#endif

struct NativeMxSingleBufferScaleSmemDescriptorMaterial {
    /*
    Transitional descriptor material for the scale-factor staging blocks.

    Payload SMEM descriptors can be generated from NativeMxSplitStageStorageView
    because Q/K/S/V addresses are normal shared-memory pointers.  Scale factors
    are different: tcgen05.cp wants the source descriptor produced by CuTe's
    UTCCP copy partition, not a plain pointer to the packed scale layout.

    This struct narrows the remaining external material to only those UTCCP
    source descriptors.  The next B300 step is to replace it with a builder based
    on UMMA::tmem_sf_frg and get_utccp_smem_desc_tensor().
    */
    uint64_t q_scale_smem_desc[MX_QK_K_CHUNKS];
    uint64_t k_scale_smem_desc[MX_QK_K_CHUNKS];
    uint64_t s_scale_smem_desc;
    uint64_t v_scale_smem_desc[MX_SV_N_CHUNKS];
};

#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_REBUILD_SCALE_SMEM_DESCS)
template <bool IsSFA>
__device__ __forceinline__ uint64_t
native_mx_make_scale_smem_stage_desc_from_utccp_partition_side_theory(
    uint8* smem_stage) {
    /*
    CUTLASS does not hand tcgen05.cp a descriptor made directly from the raw scale
    pointer.  It first creates the UTCCP tiled-copy object from the tCtSFA/SFB TMEM
    fragment, partitions the source SMEM tensor, and only then calls
    get_utccp_smem_desc_tensor().

    This helper mirrors that sequence for our single 512B stage.  `IsSFA`
    selects the same tmem_sf_frg<..., Is_SFA> side that CUTLASS uses for
    q/s scales versus k/v scales.  NativeMxScaleTmemShape represents one
    128-MN-position x K128 scale block as four K32 atom-scale slices.
    */
    using ScaleTmemFragment = cute::conditional_t<
        IsSFA,
        NativeMxScaleSfaTmemFragment,
        NativeMxScaleSfbTmemFragment>;

    auto tCsSF = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<cute::float_ue8m0_t*>(smem_stage)),
        NativeMxScaleStageLayout{});
    auto tCsSF_compact =
        cute::make_tensor(tCsSF.data(), cute::filter_zeros(tCsSF.layout()));

    auto tCtSF = cute::make_tensor<ScaleTmemFragment>(NativeMxScaleTmemShape{});
    auto tCtSF_compact =
        cute::make_tensor(tCtSF.data(), cute::filter_zeros(tCtSF.layout()));

    auto tiled_copy_s2t = cute::make_utccp_copy(NativeMxScaleUtccpOp{}, tCtSF_compact);
    auto thr_copy_s2t = tiled_copy_s2t.get_slice(0);
    auto thr_tCsSF_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact);
    auto thr_tCsSF_s2t =
        cute::get_utccp_smem_desc_tensor<NativeMxScaleUtccpOp>(thr_tCsSF_s2t_);
    return static_cast<uint64_t>(*thr_tCsSF_s2t.data());
}

__device__ __forceinline__ uint64_t
native_mx_make_scale_smem_stage_desc_from_utccp_partition_theory(
    uint8* smem_stage,
    NativeMxScaleOperand operand) {
    FLASH_DEVICE_ASSERT(
        operand == NativeMxScaleOperand::SFA ||
        operand == NativeMxScaleOperand::SFB);
    if (operand == NativeMxScaleOperand::SFA) {
        return native_mx_make_scale_smem_stage_desc_from_utccp_partition_side_theory<true>(
            smem_stage);
    }
    return native_mx_make_scale_smem_stage_desc_from_utccp_partition_side_theory<false>(
        smem_stage);
}

__device__ __forceinline__ uint64_t native_mx_make_scale_smem_stage_desc_theory(
    uint8* smem_stage,
    NativeMxScaleOperand operand) {
    /*
    Experimental direct source descriptor for the 512B scale staging block.

    CUTLASS normally reaches this descriptor through:

      make_utccp_copy(SM100_UTCCP_4x32dp128bit_1cta, tCtSFA/SFB)
      -> partition_S(filter_zeros(sSFA/SFB))
      -> get_utccp_smem_desc_tensor(...)

    The staging block that materialize_mx_scale_span_to_smem_stage() writes
    already uses Sm1xxBlockScaledBasicChunk<32>::SfKMajorAtom byte ownership.
    This helper expresses that same 512B block as a K-major UMMA descriptor.
    It is intentionally behind FLASHMLA_ENABLE_THEORETICAL_MXF8_REBUILD_SCALE_SMEM_DESCS
    until B300 confirms it is bit-identical to get_utccp_smem_desc_tensor().

    Define FLASHMLA_ENABLE_THEORETICAL_MXF8_CUTE_UTCCP_SCALE_DESCS to use the
    closer CUTLASS-style partition path above.  The fallback remains here because
    this machine cannot compile the CuTe UTCCP tensor stack.
    */
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_CUTE_UTCCP_SCALE_DESCS)
    return native_mx_make_scale_smem_stage_desc_from_utccp_partition_theory(
        smem_stage,
        operand);
#else
    (void)operand;
    auto tensor =
        cute::make_tensor(cute::make_smem_ptr(smem_stage), NativeMxScaleStageLayout{});
    cute::UMMA::SmemDescriptor desc =
        cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(tensor);
    return static_cast<uint64_t>(desc);
#endif
}

__device__ __forceinline__ NativeMxSingleBufferScaleSmemDescriptorMaterial
make_native_mx_single_buffer_scale_smem_descriptor_material_from_smem_theory(
    NativeMxSplitStageStorageView ptrs) {
    NativeMxSingleBufferScaleSmemDescriptorMaterial scale_descs{};

    NativeFp8TilePointers persistent = ptrs.persistent_tile_ptrs();
    NativeFp8TilePointers qk_tile = ptrs.qk_stage_tile_ptrs(0);
    NativeFp8TilePointers s_tile = ptrs.sv_stage_s_tile_ptrs(0);

    const uint64_t q_scale_desc =
        native_mx_make_scale_smem_stage_desc_theory(
            persistent.q_scale_smem_stage,
            NativeMxScaleOperand::SFA);
    const uint64_t k_scale_desc =
        native_mx_make_scale_smem_stage_desc_theory(
            qk_tile.k_scale_smem_stage,
            NativeMxScaleOperand::SFB);
    const uint64_t s_scale_desc =
        native_mx_make_scale_smem_stage_desc_theory(
            s_tile.s_scale_smem_stage,
            NativeMxScaleOperand::SFA);

    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        scale_descs.q_scale_smem_desc[k_chunk] = q_scale_desc;
        scale_descs.k_scale_smem_desc[k_chunk] = k_scale_desc;
    }
    scale_descs.s_scale_smem_desc = s_scale_desc;

    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        NativeMxValueChunkPointers v_chunk =
            ptrs.sv_stage_v_chunk_ptrs(0, n_chunk);
        scale_descs.v_scale_smem_desc[n_chunk] =
            native_mx_make_scale_smem_stage_desc_theory(
                v_chunk.v_scale_smem_stage,
                NativeMxScaleOperand::SFB);
    }

    return scale_descs;
}
#endif

__device__ __forceinline__ NativeMxSingleBufferScaleSmemDescriptorMaterial
native_mx_extract_single_buffer_scale_smem_descriptor_material(
    const NativeMxSplitStageDescriptorMaterial& material) {
    NativeMxSingleBufferScaleSmemDescriptorMaterial scale_descs{};

    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        scale_descs.q_scale_smem_desc[k_chunk] =
            material.qk_smem_by_stage[0].q_scale_smem_desc[k_chunk];
        scale_descs.k_scale_smem_desc[k_chunk] =
            material.qk_smem_by_stage[0].k_scale_smem_desc[k_chunk];
    }
    scale_descs.s_scale_smem_desc =
        material.sv_smem_by_stage[0][0].s_scale_smem_desc;
    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        scale_descs.v_scale_smem_desc[n_chunk] =
            material.sv_smem_by_stage[0][n_chunk].v_scale_smem_desc;
    }

    return scale_descs;
}

__device__ __forceinline__ NativeMxQkSmemDescriptorMaterial
make_native_mx_qk_smem_descriptor_material_theory(
    NativeFp8TilePointers qk_tile,
    const NativeMxSingleBufferScaleSmemDescriptorMaterial& scale_descs) {
    NativeMxQkSmemDescriptorMaterial material{};

    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        material.q_scale_smem_desc[k_chunk] = scale_descs.q_scale_smem_desc[k_chunk];
        material.k_scale_smem_desc[k_chunk] = scale_descs.k_scale_smem_desc[k_chunk];

        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            const int k_start = k_chunk * MX_MMA_K + k_atom * MX_GROUP_K;
            material.q_desc[k_chunk][k_atom] =
                native_mx_make_q_smem_desc(qk_tile, k_start);
            material.k_desc[k_chunk][k_atom] =
                native_mx_make_k_smem_desc(qk_tile, k_start);
        }
    }

    validate_native_mx_qk_smem_descriptor_material(material);
    return material;
}

__device__ __forceinline__ NativeMxCompactSvSmemDescriptorMaterial
make_native_mx_compact_sv_smem_descriptor_material_theory(
    NativeFp8TilePointers s_tile,
    NativeMxValueChunkPointers v_chunk,
    const NativeMxSingleBufferScaleSmemDescriptorMaterial& scale_descs) {
    NativeMxCompactSvSmemDescriptorMaterial material{};
    material.n_chunk = v_chunk.n_chunk;
    material.s_scale_smem_desc = scale_descs.s_scale_smem_desc;
    material.v_scale_smem_desc = scale_descs.v_scale_smem_desc[v_chunk.n_chunk];

    for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
        const int k_start = k_atom * MX_GROUP_K;
        material.s_desc[k_atom] = native_mx_make_s_smem_desc(s_tile, k_start);
        material.v_desc[k_atom] =
            native_mx_make_v_compact_smem_desc(v_chunk, k_start);
    }

    validate_native_mx_compact_sv_smem_descriptor_material(
        material,
        v_chunk.n_chunk);
    return material;
}

__device__ __forceinline__ NativeMxSplitStageDescriptorMaterial
make_native_mx_single_buffer_descriptor_material_from_smem_theory(
    NativeMxSplitStageStorageView ptrs,
    const NativeMxTmemAddressPlan& tmem,
    const NativeMxSingleBufferScaleSmemDescriptorMaterial& scale_descs) {
    /*
    Device-side material builder for the single-buffer native-MX path.

    This removes the invalid assumption that host code can precompute Q/K/S/V
    SMEM descriptors.  The descriptors below are built from the actual dynamic
    shared-memory addresses chosen for this CTA.  Scale staging descriptors are
    still passed in separately because they come from the UTCCP scale-copy
    tensor, not from the payload tensor layout.
    */
    NativeMxSplitStageDescriptorMaterial material{};
    material.tmem = native_mx_complete_scale_atom_tmem_addresses(tmem);
    material.qk_material_stages = 1;
    material.sv_material_stages = 1;
    for (int stage = 1; stage < PIPE_STAGES; ++stage) {
        material.tmem.qk_tmem_d[stage] = material.tmem.qk_tmem_d[0];
    }

    NativeFp8TilePointers persistent = ptrs.persistent_tile_ptrs();
    NativeFp8TilePointers qk_tile =
        native_mx_attach_persistent_q(ptrs.qk_stage_tile_ptrs(0), persistent);
    material.qk_smem_by_stage[0] =
        make_native_mx_qk_smem_descriptor_material_theory(qk_tile, scale_descs);

    NativeFp8TilePointers s_tile = ptrs.sv_stage_s_tile_ptrs(0);
    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        NativeMxValueChunkPointers v_chunk =
            ptrs.sv_stage_v_chunk_ptrs(0, n_chunk);
        material.sv_smem_by_stage[0][n_chunk] =
            make_native_mx_compact_sv_smem_descriptor_material_theory(
                s_tile,
                v_chunk,
                scale_descs);
    }

    validate_native_mx_split_stage_descriptor_material(material);
    return material;
}
#endif

__device__ __forceinline__ int mx_scale_idx(int mn_index, int k_group, int num_k_groups) {
    return mn_index * num_k_groups + k_group;
}

constexpr int mx_scale_factor_local_byte_offset_constexpr(
    int local_mn,
    int local_k_group) {
    /*
    CUTLASS Sm1xxBlockScaledBasicChunk<32>::SfKMajorAtom uses:

      Shape  : ((32, 4), (32, 4))
      Stride : ((16, 4), (0, 1))

    The 32-wide data-vector coordinate has stride 0 because a single scale
    applies to all 32 values in that vector.  The unique byte offset in one
    512B basic block is therefore:

      (mn % 32) * 16 + (mn / 32) * 4 + k_group

    This is not plain row-major `mn * 4 + k_group`.
    */
    return (local_mn % 32) * 16 +
           (local_mn / 32) * MX_SCALE_K_GROUPS_PER_BASIC_BLOCK +
           local_k_group;
}

static_assert(mx_scale_factor_local_byte_offset_constexpr(0, 0) == 0);
static_assert(mx_scale_factor_local_byte_offset_constexpr(1, 0) == 16);
static_assert(mx_scale_factor_local_byte_offset_constexpr(32, 0) == 4);
static_assert(mx_scale_factor_local_byte_offset_constexpr(127, 3) == 511);

__device__ __forceinline__ int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

__device__ __forceinline__ int mx_scale_factor_local_byte_offset(
    int local_mn,
    int local_k_group) {
    return (local_mn % 32) * 16 +
           (local_mn / 32) * MX_SCALE_K_GROUPS_PER_BASIC_BLOCK +
           local_k_group;
}

__device__ __forceinline__ int mx_scale_factor_byte_offset(
    int mn_index,
    int k_group,
    int total_mn) {
    /*
    SM100 block-scaled scale-factor layout:

      one 512B basic block = 128 M/N positions x 4 K-groups x 1 byte

    The order inside one basic block follows CUTLASS' SfKMajorAtom:

      byte = (mn % 32) * 16 + (mn / 32) * 4 + local_k_group

    When both M/N and scale-K exceed one basic block, the basic blocks are
    arranged K-major.  This helper turns a logical scale[mn][k_group] into the
    byte slot that tcgen05/CUTLASS scale descriptors should see.
    */
    const int mn_blocks = ceil_div_int(total_mn, MX_SCALE_MN_PER_BASIC_BLOCK);
    const int mn_block = mn_index / MX_SCALE_MN_PER_BASIC_BLOCK;
    const int local_mn = mn_index % MX_SCALE_MN_PER_BASIC_BLOCK;
    const int k_block = k_group / MX_SCALE_K_GROUPS_PER_BASIC_BLOCK;
    const int local_k = k_group % MX_SCALE_K_GROUPS_PER_BASIC_BLOCK;
    const int basic_block = k_block * mn_blocks + mn_block;
    return basic_block * MX_SCALE_BASIC_BLOCK_BYTES +
           mx_scale_factor_local_byte_offset(local_mn, local_k);
}

__device__ __forceinline__ void store_mx_scale_factor(
    MxScale* debug_scales,
    uint8* packed_layout,
    int total_mn,
    int mn_index,
    int k_group,
    int num_k_groups,
    MxScale scale) {
    FLASH_DEVICE_ASSERT(packed_layout != nullptr);
    if (debug_scales != nullptr) {
        debug_scales[mx_scale_idx(mn_index, k_group, num_k_groups)] = scale;
    }
    packed_layout[mx_scale_factor_byte_offset(mn_index, k_group, total_mn)] = scale.encoded_ue8m0;
}

__device__ __forceinline__ float decode_mx_scale_ue8m0(uint8 encoded) {
    /*
    Debug-side inverse of encode_mx_scale_ue8m0().

    Production tcgen05.mma consumes the raw ue8m0 byte through TMEM scale
    fragments.  The emulator needs the real float value, but it should read it
    from the same packed layout bytes rather than from the optional MxScale
    side array.  That keeps the debug path honest about the MX descriptor data.
    */
    const unsigned short encoded_pair =
        static_cast<unsigned short>(encoded) |
        (static_cast<unsigned short>(encoded) << 8);

    __nv_bfloat16 decoded_pair[2];
    *reinterpret_cast<__nv_bfloat162_raw*>(decoded_pair) =
        __nv_cvt_e8m0x2_to_bf162raw(encoded_pair);
    return __bfloat162float(decoded_pair[0]);
}

__device__ __forceinline__ MxScale load_mx_scale_factor(
    const MxScale* debug_scales,
    const uint8* packed_layout,
    int total_mn,
    int mn_index,
    int k_group,
    int num_k_groups) {
    if (debug_scales != nullptr) {
        return debug_scales[mx_scale_idx(mn_index, k_group, num_k_groups)];
    }

    FLASH_DEVICE_ASSERT(packed_layout != nullptr);
    const uint8 encoded =
        packed_layout[mx_scale_factor_byte_offset(mn_index, k_group, total_mn)];
    return {decode_mx_scale_ue8m0(encoded), encoded};
}

__device__ __forceinline__ const uint8* mx_scale_span_byte_ptr(
    NativeMxScaleTmemSpan span,
    int local_mn,
    int local_k_group) {
    return span.packed_layout +
           mx_scale_factor_byte_offset(
               span.mn_start + local_mn,
               span.k_group_start + local_k_group,
               span.total_mn);
}

__device__ __forceinline__ int mx_scale_span_bytes(NativeMxScaleTmemSpan span) {
    /*
    This fixed research path only issues [128, *, 128] dense MX MMAs.  Therefore
    each SFA/SFB span is one full 128 x 4 scale-factor basic block: 512 bytes.
    The value is also the source byte footprint for the UTCCP copy.
    */
    return span.mn_count * span.k_group_count * sizeof(uint8);
}

__device__ __forceinline__ void validate_mx_scale_tmem_span_contract(
    NativeMxScaleTmemSpan span) {
    FLASH_DEVICE_ASSERT(span.packed_layout != nullptr);
    FLASH_DEVICE_ASSERT(span.smem_stage != nullptr);
    FLASH_DEVICE_ASSERT(native_mx_tmem_addr_is_bound(span.tmem_addr));
    FLASH_DEVICE_ASSERT(span.total_mn > 0);
    FLASH_DEVICE_ASSERT(span.mn_start >= 0);
    FLASH_DEVICE_ASSERT(span.mn_count == MX_SCALE_MN_PER_BASIC_BLOCK);
    FLASH_DEVICE_ASSERT(span.mn_start + span.mn_count <= span.total_mn);
    FLASH_DEVICE_ASSERT(span.k_group_start >= 0);
    FLASH_DEVICE_ASSERT(span.k_group_count == MX_SCALE_K_GROUPS_PER_BASIC_BLOCK);
    FLASH_DEVICE_ASSERT(mx_scale_span_bytes(span) == MX_SCALE_STAGE_BYTES);
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX) && defined(FLASHMLA_ENABLE_THEORETICAL_UTCCP)
    FLASH_DEVICE_ASSERT(span.smem_stage_desc != 0);
#endif
}

__device__ void materialize_mx_scale_span_to_smem_stage(
    NativeMxScaleTmemSpan span) {
    /*
    For the fixed [128, *, 128] dense MX MMA tiles used here, each SFA/SFB
    operand consumes exactly one 512B scale-factor basic block.  The producers
    write the full q/k/s/v scale layouts, while this helper slices the block
    needed by the current MMA and materializes it into the per-operand SMEM
    staging block that UTCCP will copy into TMEM.

    This is real data movement for the scale bytes.  The remaining production
    step is generating the exact CuTe SMEM descriptor for this staging block and
    verifying the UTCCP/TMEM address contract on B300.
    */
    validate_mx_scale_tmem_span_contract(span);

    const int lane = warpgroup_thread_idx();

    for (int task = lane; task < span.mn_count * span.k_group_count; task += WARP_GROUP_THREADS) {
        const int local_mn = task / span.k_group_count;
        const int local_k = task % span.k_group_count;
        const int dst = mx_scale_factor_local_byte_offset(local_mn, local_k);
        span.smem_stage[dst] = *mx_scale_span_byte_ptr(span, local_mn, local_k);
    }

    native_mx_sync_scale_stage_written();
}

__device__ __forceinline__ void copy_mx_scale_smem_stage_to_tmem_theory(
    NativeMxScaleTmemSpan span) {
    /*
    CUTLASS issues the scale-factor SMEM->TMEM transfer with:

      tcgen05.cp.cta_group::1.32x128b.warpx4 [tmem_dst], smem_desc;

    The source operand is not a raw pointer.  It is the descriptor returned by
    CuTe's get_utccp_smem_desc_tensor(...) for the staged SFA/SFB tensor.  This
    wrapper makes the final storage contract explicit without requiring the full
    CuTe tensor stack in this research shell.
    */
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX) && defined(FLASHMLA_ENABLE_THEORETICAL_UTCCP)
    FLASH_DEVICE_ASSERT(span.smem_stage_desc != 0);
    if (warpgroup_thread_idx() == 0) {
        cute::SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(
            span.smem_stage_desc,
            span.tmem_addr);
    }
#else
    (void)span;
#endif
}

__device__ __forceinline__ NativeMxScaleTmemSpan make_qk_q_scale_span(
    NativeFp8TilePointers tile,
    int k_chunk,
    uint32_t tmem_addr,
    uint64_t smem_stage_desc) {
    return {
        tile.q_scale_layout,
        tile.q_scale_smem_stage,
        smem_stage_desc,
        NativeMxScaleOperand::SFA,
        tmem_addr,
        QK_M,
        0,
        QK_M,
        k_chunk * (MX_MMA_K / MX_GROUP_K),
        MX_MMA_K / MX_GROUP_K,
    };
}

__device__ __forceinline__ NativeMxScaleTmemSpan make_qk_k_scale_span(
    NativeFp8TilePointers tile,
    int k_chunk,
    uint32_t tmem_addr,
    uint64_t smem_stage_desc) {
    return {
        tile.k_scale_layout,
        tile.k_scale_smem_stage,
        smem_stage_desc,
        NativeMxScaleOperand::SFB,
        tmem_addr,
        QK_N,
        0,
        QK_N,
        k_chunk * (MX_MMA_K / MX_GROUP_K),
        MX_MMA_K / MX_GROUP_K,
    };
}

__device__ __forceinline__ NativeMxScaleTmemSpan make_sv_s_scale_span(
    NativeFp8TilePointers tile,
    uint32_t tmem_addr,
    uint64_t smem_stage_desc) {
    return {
        tile.s_scale_layout,
        tile.s_scale_smem_stage,
        smem_stage_desc,
        NativeMxScaleOperand::SFA,
        tmem_addr,
        SV_M,
        0,
        SV_M,
        0,
        MX_SV_K_GROUPS,
    };
}

__device__ __forceinline__ NativeMxScaleTmemSpan make_sv_v_scale_span(
    NativeFp8TilePointers tile,
    int n_chunk,
    uint32_t tmem_addr,
    uint64_t smem_stage_desc) {
    return {
        tile.v_scale_layout,
        tile.v_scale_smem_stage,
        smem_stage_desc,
        NativeMxScaleOperand::SFB,
        tmem_addr,
        SV_N,
        n_chunk * MX_SV_N_TILE,
        MX_SV_N_TILE,
        0,
        MX_SV_K_GROUPS,
    };
}

__device__ __forceinline__ NativeMxScaleTmemSpan make_sv_v_compact_scale_span(
    NativeMxValueChunkPointers v_chunk,
    uint32_t tmem_addr,
    uint64_t smem_stage_desc) {
    return {
        v_chunk.v_scale_layout,
        v_chunk.v_scale_smem_stage,
        smem_stage_desc,
        NativeMxScaleOperand::SFB,
        tmem_addr,
        MX_SV_N_TILE,
        0,
        MX_SV_N_TILE,
        0,
        MX_SV_K_GROUPS,
    };
}

__device__ __forceinline__ NativeMxMmaTileSpec make_qk_tile_spec(int k_chunk) {
    return {
        QK_M,
        QK_N,
        MX_MMA_K,
        k_chunk * MX_MMA_K,
        0,
    };
}

__device__ __forceinline__ NativeMxMmaTileSpec make_qk_atom_tile_spec(
    int k_chunk,
    int k_atom) {
    return {
        QK_M,
        QK_N,
        MX_GROUP_K,
        k_chunk * MX_MMA_K + k_atom * MX_GROUP_K,
        0,
    };
}

__device__ __forceinline__ NativeMxMmaTileSpec make_sv_tile_spec(int n_chunk) {
    return {
        SV_M,
        MX_SV_N_TILE,
        SV_K,
        0,
        n_chunk * MX_SV_N_TILE,
    };
}

__device__ __forceinline__ NativeMxMmaTileSpec make_sv_atom_tile_spec(
    int n_chunk,
    int k_atom) {
    return {
        SV_M,
        MX_SV_N_TILE,
        MX_GROUP_K,
        k_atom * MX_GROUP_K,
        n_chunk * MX_SV_N_TILE,
    };
}

__device__ __forceinline__ NativeMxOperandAtomSpan make_q_atom_span(
    NativeFp8TilePointers tile,
    int k_chunk,
    int k_atom,
    uint64_t smem_desc) {
    const int k_start = k_chunk * MX_MMA_K + k_atom * MX_GROUP_K;
    return {
        tile.q_mx,
        smem_desc,
        NativeMxOperandRole::Q,
        0,
        k_start,
        QK_M,
        MX_GROUP_K,
        D_QK_PAD,
        false,
    };
}

__device__ __forceinline__ NativeMxOperandAtomSpan make_k_atom_span(
    NativeFp8TilePointers tile,
    int k_chunk,
    int k_atom,
    uint64_t smem_desc) {
    const int k_start = k_chunk * MX_MMA_K + k_atom * MX_GROUP_K;
    return {
        tile.k_mx,
        smem_desc,
        NativeMxOperandRole::K,
        0,
        k_start,
        QK_N,
        MX_GROUP_K,
        D_QK_PAD,
        false,
    };
}

__device__ __forceinline__ NativeMxOperandAtomSpan make_s_atom_span(
    NativeFp8TilePointers tile,
    int k_atom,
    uint64_t smem_desc) {
    const int k_start = k_atom * MX_GROUP_K;
    return {
        tile.s_mx,
        smem_desc,
        NativeMxOperandRole::S,
        0,
        k_start,
        SV_M,
        MX_GROUP_K,
        SV_K,
        false,
    };
}

__device__ __forceinline__ NativeMxOperandAtomSpan make_v_atom_span(
    NativeFp8TilePointers tile,
    int n_chunk,
    int k_atom,
    uint64_t smem_desc) {
    return {
        tile.v_mx,
        smem_desc,
        NativeMxOperandRole::V,
        k_atom * MX_GROUP_K,
        n_chunk * MX_SV_N_TILE,
        MX_GROUP_K,
        MX_SV_N_TILE,
        SV_N,
        true,
    };
}

__device__ __forceinline__ NativeMxOperandAtomSpan make_v_compact_atom_span(
    NativeMxValueChunkPointers v_chunk,
    int k_atom,
    uint64_t smem_desc) {
    return {
        v_chunk.v_mx,
        smem_desc,
        NativeMxOperandRole::V,
        k_atom * MX_GROUP_K,
        0,
        MX_GROUP_K,
        MX_SV_N_TILE,
        MX_SV_N_TILE,
        true,
    };
}

__device__ __forceinline__ NativeMxMmaDescriptorSet make_native_mx_mma_descriptor_set(
    NativeFp8TilePointers tile,
    const NativeMxRawDescriptorInputs& raw) {
    NativeMxMmaDescriptorSet descs{};

    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        descs.qk[k_chunk].a_scale_span =
            make_qk_q_scale_span(
                tile,
                k_chunk,
                raw.q_scale_tmem_block_addr[k_chunk],
                raw.q_scale_smem_desc[k_chunk]);
        descs.qk[k_chunk].b_scale_span =
            make_qk_k_scale_span(
                tile,
                k_chunk,
                raw.k_scale_tmem_block_addr[k_chunk],
                raw.k_scale_smem_desc[k_chunk]);
        descs.qk[k_chunk].scale_block_tile = make_qk_tile_spec(k_chunk);

        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            descs.qk[k_chunk].atoms[k_atom] = {
                {
                    raw.qk_tmem_d,
                    raw.q_desc[k_chunk][k_atom],
                    raw.k_desc[k_chunk][k_atom],
                    raw.qk_block_scaled_idesc[k_chunk][k_atom],
                    raw.q_scale_tmem_addr[k_chunk][k_atom],
                    raw.k_scale_tmem_addr[k_chunk][k_atom],
                    k_chunk != 0 || k_atom != 0,
                },
                make_q_atom_span(tile, k_chunk, k_atom, raw.q_desc[k_chunk][k_atom]),
                make_k_atom_span(tile, k_chunk, k_atom, raw.k_desc[k_chunk][k_atom]),
                make_qk_atom_tile_spec(k_chunk, k_atom),
                k_atom,
            };
        }
    }

    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        descs.sv[n_chunk].a_scale_span =
            make_sv_s_scale_span(
                tile,
                raw.s_scale_tmem_block_addr,
                raw.s_scale_smem_desc);
        descs.sv[n_chunk].b_scale_span =
            make_sv_v_scale_span(
                tile,
                n_chunk,
                raw.v_scale_tmem_block_addr[n_chunk],
                raw.v_scale_smem_desc[n_chunk]);
        descs.sv[n_chunk].scale_block_tile = make_sv_tile_spec(n_chunk);

        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            descs.sv[n_chunk].atoms[k_atom] = {
                {
                    raw.sv_tmem_d[n_chunk],
                    raw.s_desc[k_atom],
                    raw.v_desc[n_chunk][k_atom],
                    raw.sv_block_scaled_idesc[n_chunk][k_atom],
                    raw.s_scale_tmem_addr[k_atom],
                    raw.v_scale_tmem_addr[n_chunk][k_atom],
                    false,
                },
                make_s_atom_span(tile, k_atom, raw.s_desc[k_atom]),
                make_v_atom_span(tile, n_chunk, k_atom, raw.v_desc[n_chunk][k_atom]),
                make_sv_atom_tile_spec(n_chunk, k_atom),
                k_atom,
            };
        }
    }

    return descs;
}

__device__ __forceinline__ NativeMxQkDescriptorSet make_native_mx_qk_descriptor_set(
    NativeFp8TilePointers qk_tile,
    const NativeMxQkDescriptorInputs& raw) {
    validate_native_mx_qk_descriptor_inputs(raw);
    NativeMxQkDescriptorSet descs{};

    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        descs.qk[k_chunk].a_scale_span =
            make_qk_q_scale_span(
                qk_tile,
                k_chunk,
                raw.q_scale_tmem_block_addr[k_chunk],
                raw.q_scale_smem_desc[k_chunk]);
        descs.qk[k_chunk].b_scale_span =
            make_qk_k_scale_span(
                qk_tile,
                k_chunk,
                raw.k_scale_tmem_block_addr[k_chunk],
                raw.k_scale_smem_desc[k_chunk]);
        descs.qk[k_chunk].scale_block_tile = make_qk_tile_spec(k_chunk);

        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            descs.qk[k_chunk].atoms[k_atom] = {
                {
                    raw.qk_tmem_d,
                    raw.q_desc[k_chunk][k_atom],
                    raw.k_desc[k_chunk][k_atom],
                    raw.qk_block_scaled_idesc[k_chunk][k_atom],
                    raw.q_scale_tmem_addr[k_chunk][k_atom],
                    raw.k_scale_tmem_addr[k_chunk][k_atom],
                    k_chunk != 0 || k_atom != 0,
                },
                make_q_atom_span(qk_tile, k_chunk, k_atom, raw.q_desc[k_chunk][k_atom]),
                make_k_atom_span(qk_tile, k_chunk, k_atom, raw.k_desc[k_chunk][k_atom]),
                make_qk_atom_tile_spec(k_chunk, k_atom),
                k_atom,
            };
        }
    }

    return descs;
}

__device__ __forceinline__ NativeMxMmaIssuePlan make_native_mx_compact_sv_issue_plan(
    NativeFp8TilePointers s_tile,
    NativeMxValueChunkPointers v_chunk,
    const NativeMxCompactSvDescriptorInputs& raw) {
    validate_native_mx_compact_sv_descriptor_inputs(raw);
    NativeMxMmaIssuePlan plan{};
    FLASH_DEVICE_ASSERT(raw.n_chunk == v_chunk.n_chunk);
    FLASH_DEVICE_ASSERT(raw.n_chunk >= 0 && raw.n_chunk < MX_SV_N_CHUNKS);

    plan.a_scale_span =
        make_sv_s_scale_span(
            s_tile,
            raw.s_scale_tmem_block_addr,
            raw.s_scale_smem_desc);
    plan.b_scale_span =
        make_sv_v_compact_scale_span(
            v_chunk,
            raw.v_scale_tmem_block_addr,
            raw.v_scale_smem_desc);
    plan.scale_block_tile = make_sv_tile_spec(raw.n_chunk);

    for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
        plan.atoms[k_atom] = {
            {
                raw.sv_tmem_d,
                raw.s_desc[k_atom],
                raw.v_desc[k_atom],
                raw.sv_block_scaled_idesc[k_atom],
                raw.s_scale_tmem_addr[k_atom],
                raw.v_scale_tmem_addr[k_atom],
                false,
            },
            make_s_atom_span(s_tile, k_atom, raw.s_desc[k_atom]),
            make_v_compact_atom_span(v_chunk, k_atom, raw.v_desc[k_atom]),
            make_sv_atom_tile_spec(raw.n_chunk, k_atom),
            k_atom,
        };
    }

    return plan;
}

__device__ __forceinline__ int topk_index(
    const SparseAttnFwdParams& params,
    int s_q_idx,
    int topk_pos) {
    const int* indices =
        params.indices +
        static_cast<int64_t>(s_q_idx) * params.stride_indices_s_q;
    return __ldg(indices + topk_pos);
}

__device__ __forceinline__ V32TokenRef resolve_v32_token_at_topk_pos(
    const SparseAttnFwdParams& params,
    int s_q_idx,
    int topk_pos,
    int effective_topk_length) {
    if (topk_pos >= effective_topk_length) {
        return {nullptr, false};
    }

    const int token_idx = topk_index(params, s_q_idx, topk_pos);
    return resolve_v32_token(params, token_idx, topk_pos, effective_topk_length);
}

__device__ void transcode_q_to_mx(
    const SparseAttnFwdParams& params,
    NativeFp8TilePointers tile,
    int s_q_idx) {
    constexpr int NUM_K_GROUPS = MX_QK_K_GROUPS;
    const int h = warpgroup_thread_idx();
    const bool valid_head = h < params.h_q;
    const bf16* q =
        valid_head ?
        params.q +
            static_cast<int64_t>(s_q_idx) * params.stride_q_s_q +
            static_cast<int64_t>(h) * params.stride_q_h_q :
        nullptr;

    /*
    One lane owns one logical MX row.  The previous task-major loop revisited the
    same Q row for every K group and recomputed the row pointer through
    load_q_value(); row-major ownership keeps the persistent Q transcode cheap
    enough to be amortized over all topk blocks.
    */
    for (int kg = 0; kg < NUM_K_GROUPS; ++kg) {
        float values[MX_GROUP_K];

        float amax = 0.0f;
#pragma unroll
        for (int kk = 0; kk < MX_GROUP_K; ++kk) {
            const int dim = kg * MX_GROUP_K + kk;
            const float value =
                (valid_head && dim < D_QK) ? float(__ldg(q + dim)) : 0.0f;
            values[kk] = value;
            amax = fmaxf(amax, fabsf(value));
        }

        const MxScale scale = make_mx_scale_from_amax(amax);
        store_mx_scale_factor(tile.q_scale, tile.q_scale_layout, QK_M, h, kg, NUM_K_GROUPS, scale);

#pragma unroll
        for (int kk = 0; kk < MX_GROUP_K; ++kk) {
            const int dim = kg * MX_GROUP_K + kk;
            tile.q_mx[native_mx_q_operand_offset(h, dim)] =
                quantize_to_mx_fp8(values[kk], scale);
        }
    }
}

__device__ void transcode_k_tile_to_mx(
    const SparseAttnFwdParams& params,
    NativeFp8TilePointers tile,
    int s_q_idx,
    int topk_block,
    int effective_topk_length) {
    constexpr int NUM_K_GROUPS = MX_QK_K_GROUPS;
    const int local_token = warpgroup_thread_idx();
    const int topk_pos = topk_block * QK_N + local_token;
    const V32TokenRef token =
        resolve_v32_token_at_topk_pos(params, s_q_idx, topk_pos, effective_topk_length);
    const V32TokenParts parts = maybe_split_token(token);

    /*
    QK_N is 128, exactly one warpgroup.  Give each lane one topk token and let it
    walk all 20 MX K-groups.  This preserves work balance but avoids resolving the
    same sparse index once per group, which was 20x redundant in the task-major
    sketch.
    */
    for (int kg = 0; kg < NUM_K_GROUPS; ++kg) {
        float values[MX_GROUP_K];

        /*
        NoPE: V3.2 stores e4m3 payload plus one fp32 scale per 128 dimensions.
        Blackwell MX wants one ue8m0 scale per 32-wide GEMM-K group in the 512B
        basic-block layout.  The safe conversion is therefore:

          V3.2 raw fp8 * V3.2 dim128 fp32 scale -> real values
          real values over this 32-dim group    -> new MX ue8m0 scale
          real values / MX scale                -> MX e4m3 payload

        For NoPE the cheap path is also the right baseline: use the existing
        V3.2 dim128 scale as four identical MX dim32 scales and copy the raw e4m3
        payload.  That makes MX dequantization reconstruct the same K values as
        the current V3.2 bf16-dequant route.  Define
        FLASHMLA_ENABLE_THEORETICAL_MXF8_V32_NOPE_REQUANT only to study the more
        expensive "tighten each 32-wide group" variant.
        */
        if (kg < V32_NOPE_MX_K_GROUPS) {
#if !defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_V32_NOPE_REQUANT)
            const int v32_scale_group = kg / V32_MX_GROUPS_PER_SCALE;
            const MxScale scale =
                make_mx_scale_from_real_scale(
                    load_v32_nope_group_scale_from_parts(
                        token,
                        parts,
                        v32_scale_group));

            store_mx_scale_factor(
                tile.k_scale,
                tile.k_scale_layout,
                QK_N,
                local_token,
                kg,
                NUM_K_GROUPS,
                scale);

#pragma unroll
            for (int kk = 0; kk < MX_GROUP_K; ++kk) {
                const int dim = kg * MX_GROUP_K + kk;
                tile.k_mx[native_mx_k_operand_offset(local_token, dim)] =
                    load_v32_nope_raw_fp8_from_parts(token, parts, dim);
            }
#else
            float amax = 0.0f;
#pragma unroll
            for (int kk = 0; kk < MX_GROUP_K; ++kk) {
                const int dim = kg * MX_GROUP_K + kk;
                const float value = load_v32_k_value_from_parts(token, parts, dim);
                values[kk] = value;
                amax = fmaxf(amax, fabsf(value));
            }

            const MxScale scale = make_mx_scale_from_amax(amax);
            store_mx_scale_factor(
                tile.k_scale,
                tile.k_scale_layout,
                QK_N,
                local_token,
                kg,
                NUM_K_GROUPS,
                scale);

#pragma unroll
            for (int kk = 0; kk < MX_GROUP_K; ++kk) {
                const int dim = kg * MX_GROUP_K + kk;
                tile.k_mx[native_mx_k_operand_offset(local_token, dim)] =
                    quantize_to_mx_fp8(values[kk], scale);
            }
#endif
            continue;
        }

        /*
        RoPE: V3.2 keeps the last 64 dims as bf16, so native MX has to create
        two new 32-wide K groups.  The final two 32-wide groups are zero padding
        for the D_QK=576 -> D_QK_PAD=640 tile shape.
        */
        float amax = 0.0f;
#pragma unroll
        for (int kk = 0; kk < MX_GROUP_K; ++kk) {
            const int dim = kg * MX_GROUP_K + kk;
            const float value = load_v32_k_value_from_parts(token, parts, dim);
            values[kk] = value;
            amax = fmaxf(amax, fabsf(value));
        }

        const MxScale scale = make_mx_scale_from_amax(amax);
        store_mx_scale_factor(tile.k_scale, tile.k_scale_layout, QK_N, local_token, kg, NUM_K_GROUPS, scale);

#pragma unroll
        for (int kk = 0; kk < MX_GROUP_K; ++kk) {
            const int dim = kg * MX_GROUP_K + kk;
            tile.k_mx[native_mx_k_operand_offset(local_token, dim)] =
                quantize_to_mx_fp8(values[kk], scale);
        }
    }
}

__device__ void transcode_v_nchunk_to_mx(
    const SparseAttnFwdParams& params,
    NativeFp8TilePointers tile,
    int s_q_idx,
    int topk_block,
    int n_chunk,
    int effective_topk_length) {
    // For S@V, the MMA K dimension is token/topk.  V3.2 scales vary by token,
    // but MX requires one scale for each output dimension and 32-token group.
    // This is the expensive layout-reconciliation pass.  The function handles
    // one 128-wide value-dimension chunk so a production pipeline can avoid
    // keeping full D_V=512 V_mx resident while QK/softmax are running.
    constexpr int NUM_K_GROUPS = MX_SV_K_GROUPS;
    FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);
    FLASH_DEVICE_ASSERT(tile.v_mx != nullptr);
    FLASH_DEVICE_ASSERT(tile.v_scale_layout != nullptr);
    const int lane = warpgroup_thread_idx();
    const int dim_start = n_chunk * MX_SV_N_TILE;

    for (int task = lane; task < MX_SV_N_TILE * NUM_K_GROUPS; task += WARP_GROUP_THREADS) {
        const int local_dim = task / NUM_K_GROUPS;
        const int dim = dim_start + local_dim;
        const int kg = task % NUM_K_GROUPS;
        float values[MX_GROUP_K];

        float amax = 0.0f;
#pragma unroll
        for (int kk = 0; kk < MX_GROUP_K; ++kk) {
            const int local_token = kg * MX_GROUP_K + kk;
            const int topk_pos = topk_block * SV_K + local_token;
            const V32TokenRef token =
                resolve_v32_token_at_topk_pos(params, s_q_idx, topk_pos, effective_topk_length);
            const float value =
                load_v32_v_value_from_parts(token, maybe_split_token(token), dim);
            values[kk] = value;
            amax = fmaxf(amax, fabsf(value));
        }

        const MxScale scale = make_mx_scale_from_amax(amax);
        store_mx_scale_factor(tile.v_scale, tile.v_scale_layout, SV_N, dim, kg, NUM_K_GROUPS, scale);

#pragma unroll
        for (int kk = 0; kk < MX_GROUP_K; ++kk) {
            const int local_token = kg * MX_GROUP_K + kk;
            // This still writes into the full [SV_K, SV_N] debug/storage view.
            // A smaller production SV stage should switch this to local_dim and
            // a per-chunk V scale layout.
            tile.v_mx[native_mx_v_full_operand_offset(local_token, dim)] =
                quantize_to_mx_fp8(values[kk], scale);
        }
    }
}

__device__ void transcode_v_nchunk_to_compact_mx(
    const SparseAttnFwdParams& params,
    NativeMxValueChunkPointers v_chunk,
    int s_q_idx,
    int topk_block,
    int effective_topk_length) {
    /*
    Production-oriented V transcode for one 128-wide value chunk.

    This is the compact version of transcode_v_nchunk_to_mx(): scale ownership
    is rebuilt as [local_value_dim_0..127][topk_k_group_0..3], and v_mx uses
    local_dim as its column.  That makes the V operand resident size 16KB plus
    one 512B scale block instead of the full 64KB + 2KB V layout.
    */
    constexpr int NUM_K_GROUPS = MX_SV_K_GROUPS;
    FLASH_DEVICE_ASSERT(v_chunk.n_chunk >= 0 && v_chunk.n_chunk < MX_SV_N_CHUNKS);
    FLASH_DEVICE_ASSERT(v_chunk.v_mx != nullptr);
    FLASH_DEVICE_ASSERT(v_chunk.v_scale_layout != nullptr);

    const int local_dim = warpgroup_thread_idx();
    const int dim_start = v_chunk.n_chunk * MX_SV_N_TILE;
    const int dim = dim_start + local_dim;
    const int warp_lane = threadIdx.x & 31;
    const int v32_scale_group = v_chunk.n_chunk;

    /*
    One lane owns one output value dimension within the compact 128-wide chunk.
    It then walks the four MX token-groups.  This matches the scale ownership
    v_scale[local_dim][token_group] and keeps the dequantized 32-token group in
    registers so the V3.2 cache is not read twice for amax and quantization.
    */
    for (int kg = 0; kg < NUM_K_GROUPS; ++kg) {
        float values[MX_GROUP_K];

        float amax = 0.0f;
#pragma unroll
        for (int kk = 0; kk < MX_GROUP_K; ++kk) {
            const int local_token = kg * MX_GROUP_K + kk;
            const int topk_pos = topk_block * SV_K + local_token;
            const V32TokenRef token =
                resolve_v32_token_at_topk_pos(params, s_q_idx, topk_pos, effective_topk_length);
            const V32TokenParts parts = maybe_split_token(token);
            float token_scale = 1.0f;
            if (warp_lane == 0) {
                token_scale =
                    load_v32_nope_group_scale_from_parts(
                        token,
                        parts,
                        v32_scale_group);
            }
            token_scale = __shfl_sync(0xffffffff, token_scale, 0);
            const float value =
                load_v32_v_value_with_scale_from_parts(token, parts, dim, token_scale);
            values[kk] = value;
            amax = fmaxf(amax, fabsf(value));
        }

        const MxScale scale = make_mx_scale_from_amax(amax);
        store_mx_scale_factor(
            v_chunk.v_scale,
            v_chunk.v_scale_layout,
            MX_SV_N_TILE,
            local_dim,
            kg,
            NUM_K_GROUPS,
            scale);

#pragma unroll
        for (int kk = 0; kk < MX_GROUP_K; ++kk) {
            const int local_token = kg * MX_GROUP_K + kk;
            v_chunk.v_mx[native_mx_v_compact_operand_offset(local_token, local_dim)] =
                quantize_to_mx_fp8(values[kk], scale);
        }
    }
}

__device__ void transcode_v_tile_to_mx(
    const SparseAttnFwdParams& params,
    NativeFp8TilePointers tile,
    int s_q_idx,
    int topk_block,
    int effective_topk_length) {
    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        transcode_v_nchunk_to_mx(
            params,
            tile,
            s_q_idx,
            topk_block,
            n_chunk,
            effective_topk_length);
    }
}

__device__ void emulate_mx_qk_mma_to_float(NativeFp8TilePointers tile) {
    // Debug fallback for CUDA-free development.  Production should replace this
    // with tcgen05.mma.kind::mxf8f6f4.block_scale.
    constexpr int NUM_K_GROUPS = MX_QK_K_GROUPS;
    FLASH_DEVICE_ASSERT(tile.q_scale != nullptr || tile.q_scale_layout != nullptr);
    FLASH_DEVICE_ASSERT(tile.k_scale != nullptr || tile.k_scale_layout != nullptr);
    FLASH_DEVICE_ASSERT(tile.qk_tmem_or_debug != nullptr);
    const int lane = warpgroup_thread_idx();

    for (int task = lane; task < QK_M * QK_N; task += WARP_GROUP_THREADS) {
        const int h = task / QK_N;
        const int n = task % QK_N;

        float sum = 0.0f;
        for (int dim = 0; dim < D_QK_PAD; ++dim) {
            const int k_group = dim / MX_GROUP_K;
            const MxScale q_scale =
                load_mx_scale_factor(
                    tile.q_scale,
                    tile.q_scale_layout,
                    QK_M,
                    h,
                    k_group,
                    NUM_K_GROUPS);
            const MxScale k_scale =
                load_mx_scale_factor(
                    tile.k_scale,
                    tile.k_scale_layout,
                    QK_N,
                    n,
                    k_group,
                    NUM_K_GROUPS);
            const float q =
                dequant_from_mx_fp8(tile.q_mx[native_mx_q_operand_offset(h, dim)], q_scale);
            const float k =
                dequant_from_mx_fp8(tile.k_mx[native_mx_k_operand_offset(n, dim)], k_scale);
            sum += q * k;
        }
        tile.qk_tmem_or_debug[row_major(h, n, QK_N)] = sum;
    }

}

__device__ void transcode_online_softmax_to_mx(
    NativeFp8TilePointers tile,
    float sm_scale_div_log2,
    int topk_block,
    int effective_topk_length,
    bool first_topk_block) {
    constexpr int NUM_K_GROUPS = MX_SV_K_GROUPS;
    const int lane = warpgroup_thread_idx();

    for (int h = lane; h < SV_M; h += WARP_GROUP_THREADS) {
        float block_m = -CUDART_INF_F;
        for (int n = 0; n < SV_K; ++n) {
            const int topk_pos = topk_block * SV_K + n;
            const bool valid = topk_pos < effective_topk_length;
            const float logit =
                valid ?
                tile.qk_tmem_or_debug[row_major(h, n, QK_N)] * sm_scale_div_log2 :
                -CUDART_INF_F;
            block_m = fmaxf(block_m, logit);
        }

        const float old_m = first_topk_block ? -CUDART_INF_F : tile.row_m[h];
        const float old_l = first_topk_block ? 0.0f : tile.row_l[h];
        const float new_m = fmaxf(old_m, block_m);
        const float old_o_scale = first_topk_block ? 0.0f : exp2f(old_m - new_m);

        float row_sum = 0.0f;
        for (int n = 0; n < SV_K; ++n) {
            const int topk_pos = topk_block * SV_K + n;
            const bool valid = topk_pos < effective_topk_length;
            const float logit =
                valid ?
                tile.qk_tmem_or_debug[row_major(h, n, QK_N)] * sm_scale_div_log2 :
                -CUDART_INF_F;
            const float p = valid ? exp2f(logit - new_m) : 0.0f;
            tile.qk_tmem_or_debug[row_major(h, n, QK_N)] = p;
            row_sum += p;
        }

        tile.row_m[h] = new_m;
        tile.row_l[h] = old_l * old_o_scale + row_sum;
        tile.row_o_scale[h] = old_o_scale;

        for (int kg = 0; kg < NUM_K_GROUPS; ++kg) {
            float amax = 0.0f;
            for (int kk = 0; kk < MX_GROUP_K; ++kk) {
                const int n = kg * MX_GROUP_K + kk;
                amax = fmaxf(amax, fabsf(tile.qk_tmem_or_debug[row_major(h, n, QK_N)]));
            }

            const MxScale scale = make_mx_scale_from_amax(amax);
            store_mx_scale_factor(tile.s_scale, tile.s_scale_layout, SV_M, h, kg, NUM_K_GROUPS, scale);

            for (int kk = 0; kk < MX_GROUP_K; ++kk) {
                const int n = kg * MX_GROUP_K + kk;
                const float value = tile.qk_tmem_or_debug[row_major(h, n, QK_N)];
                tile.s_mx[native_mx_s_operand_offset(h, n)] =
                    quantize_to_mx_fp8(value, scale);
            }
        }
    }
}

struct NativeMxQkTmemLogitsBridgePlan {
    /*
    Staging contract for the production QK path:

      tcgen05.mma writes QK logits to TMEM at qk_tmem_d.
      online softmax needs one row of 128 logits in registers.

    The direct-register path below loads the row from TMEM and immediately runs
    softmax/S quantization.  This staging plan remains as a transitional bridge
    for validating the same TMEM values with the existing debug softmax path.
    */
    uint32_t qk_tmem_d;
    float* qk_logits_staging;  // [QK_M, QK_N]
};

__device__ __forceinline__ void stage_qk_tmem_logits_to_debug_buffer_theory(
    NativeMxQkTmemLogitsBridgePlan plan) {
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX) && \
    defined(FLASHMLA_ENABLE_THEORETICAL_TMEM_LOGITS_BRIDGE)
    FLASH_DEVICE_ASSERT(plan.qk_logits_staging != nullptr);

    float logits[QK_N];
    ku::tmem_ld_32dp32bNx<QK_N>(plan.qk_tmem_d, logits);
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();

    const int h = warpgroup_thread_idx();
    for (int n = 0; n < QK_N; ++n) {
        plan.qk_logits_staging[row_major(h, n, QK_N)] = logits[n];
    }
#else
    (void)plan;
#endif
}

__device__ __forceinline__ void transcode_online_softmax_from_qk_tmem_registers_theory(
    NativeFp8TilePointers s_tile,
    uint32_t qk_tmem_d,
    float sm_scale_div_log2,
    int topk_block,
    int effective_topk_length,
    bool first_topk_block) {
    /*
    Direct-register version of the QK TMEM bridge.

    This is the production direction: load one logical QK row from TMEM into
    registers, apply the sparse topk mask, update online softmax state, and
    quantize S directly into the MX A operand plus packed MX SFA scale layout.
    No qk_tmem_or_debug staging buffer is required.

    The exact TMEM row mapping still needs B300 validation.  The fixed contract
    here is that one warpgroup lane owns one logical head row h in [0, 127].
    */
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX) && \
    defined(FLASHMLA_ENABLE_THEORETICAL_DIRECT_TMEM_SOFTMAX)
    FLASH_DEVICE_ASSERT(s_tile.s_mx != nullptr);
    FLASH_DEVICE_ASSERT(s_tile.s_scale_layout != nullptr);
    FLASH_DEVICE_ASSERT(s_tile.row_m != nullptr);
    FLASH_DEVICE_ASSERT(s_tile.row_l != nullptr);
    FLASH_DEVICE_ASSERT(s_tile.row_o_scale != nullptr);

    float logits[QK_N];
    ku::tmem_ld_32dp32bNx<QK_N>(qk_tmem_d, logits);
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();

    const int h = warpgroup_thread_idx();

    float block_m = -CUDART_INF_F;
    for (int n = 0; n < SV_K; ++n) {
        const int topk_pos = topk_block * SV_K + n;
        const bool valid = topk_pos < effective_topk_length;
        const float logit = valid ? logits[n] * sm_scale_div_log2 : -CUDART_INF_F;
        block_m = fmaxf(block_m, logit);
    }

    const float old_m = first_topk_block ? -CUDART_INF_F : s_tile.row_m[h];
    const float old_l = first_topk_block ? 0.0f : s_tile.row_l[h];
    const float new_m = fmaxf(old_m, block_m);
    const float old_o_scale = first_topk_block ? 0.0f : exp2f(old_m - new_m);

    float row_sum = 0.0f;
    for (int n = 0; n < SV_K; ++n) {
        const int topk_pos = topk_block * SV_K + n;
        const bool valid = topk_pos < effective_topk_length;
        const float logit = valid ? logits[n] * sm_scale_div_log2 : -CUDART_INF_F;
        const float p = valid ? exp2f(logit - new_m) : 0.0f;
        logits[n] = p;
        row_sum += p;
    }

    s_tile.row_m[h] = new_m;
    s_tile.row_l[h] = old_l * old_o_scale + row_sum;
    s_tile.row_o_scale[h] = old_o_scale;

    for (int kg = 0; kg < MX_SV_K_GROUPS; ++kg) {
        float amax = 0.0f;
        for (int kk = 0; kk < MX_GROUP_K; ++kk) {
            const int n = kg * MX_GROUP_K + kk;
            amax = fmaxf(amax, fabsf(logits[n]));
        }

        const MxScale scale = make_mx_scale_from_amax(amax);
        store_mx_scale_factor(
            s_tile.s_scale,
            s_tile.s_scale_layout,
            SV_M,
            h,
            kg,
            MX_SV_K_GROUPS,
            scale);

        for (int kk = 0; kk < MX_GROUP_K; ++kk) {
            const int n = kg * MX_GROUP_K + kk;
            s_tile.s_mx[native_mx_s_operand_offset(h, n)] =
                quantize_to_mx_fp8(logits[n], scale);
        }
    }
#else
    FLASH_DEVICE_ASSERT(false);
    (void)s_tile;
    (void)qk_tmem_d;
    (void)sm_scale_div_log2;
    (void)topk_block;
    (void)effective_topk_length;
    (void)first_topk_block;
#endif
}

__device__ void transcode_online_softmax_from_qk_tmem_to_mx_theory(
    NativeFp8TilePointers s_tile,
    uint32_t qk_tmem_d,
    float sm_scale_div_log2,
    int topk_block,
    int effective_topk_length,
    bool first_topk_block) {
    /*
    Production split-stage hook.

    The native MX QK path writes logits into TMEM, not into the debug
    qk_tmem_or_debug float buffer.  The final B300 kernel therefore needs an
    explicit TMEM -> online-softmax -> S_mx/S_scale bridge.  That bridge is
    intentionally separated from transcode_online_softmax_to_mx(), which reads a
    normal float pointer and is only valid for CUDA-free emulation.

    Until the exact TMEM load/reduction sequence is verified on B300, this
    function only documents the boundary.  The non-PTX path can still use the
    debug buffer when a storage implementation provides one.
    */
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
#if defined(FLASHMLA_ENABLE_THEORETICAL_DIRECT_TMEM_SOFTMAX)
    transcode_online_softmax_from_qk_tmem_registers_theory(
        s_tile,
        qk_tmem_d,
        sm_scale_div_log2,
        topk_block,
        effective_topk_length,
        first_topk_block);
#elif defined(FLASHMLA_ENABLE_THEORETICAL_TMEM_LOGITS_BRIDGE)
    if (s_tile.qk_tmem_or_debug != nullptr) {
        stage_qk_tmem_logits_to_debug_buffer_theory({qk_tmem_d, s_tile.qk_tmem_or_debug});
        transcode_online_softmax_to_mx(
            s_tile,
            sm_scale_div_log2,
            topk_block,
            effective_topk_length,
            first_topk_block);
    }
#else
    (void)s_tile;
    (void)qk_tmem_d;
    (void)sm_scale_div_log2;
    (void)topk_block;
    (void)effective_topk_length;
    (void)first_topk_block;
#endif
#else
    (void)qk_tmem_d;
    if (s_tile.qk_tmem_or_debug != nullptr) {
        transcode_online_softmax_to_mx(
            s_tile,
            sm_scale_div_log2,
            topk_block,
            effective_topk_length,
            first_topk_block);
    }
#endif
}

__device__ __forceinline__ void scale_compact_sv_tmem_accumulator_theory(
    NativeFp8TilePointers s_tile,
    uint32_t sv_tmem_d,
    bool first_topk_block) {
    /*
    Online softmax changes the normalization base at every topk block:

      O_old *= exp2(old_m - new_m) == row_o_scale[h]

    The debug emulator applies that factor before adding the current S@V block.
    Native MX must do the same to the TMEM accumulator before issuing the next
    block's S@V MMA with accumulate=true.

    The compact S@V path owns one 128-wide value chunk at a time, so this scales
    one [head, value_chunk128] accumulator tile.  The exact row mapping is the
    same contract used by transcode_online_softmax_from_qk_tmem_registers_theory:
    one warpgroup lane owns one logical head row.
    */
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    if (first_topk_block) {
        return;
    }
    FLASH_DEVICE_ASSERT(s_tile.row_o_scale != nullptr);

    float o[MX_SV_N_TILE];
    ku::tmem_ld_32dp32bNx<MX_SV_N_TILE>(sv_tmem_d, o);
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();

    const int h = warpgroup_thread_idx();
    const float scale = s_tile.row_o_scale[h];
    for (int i = 0; i < MX_SV_N_TILE; ++i) {
        o[i] *= scale;
    }

    ku::tmem_st_32dp32bNx<MX_SV_N_TILE>(sv_tmem_d, o);
#else
    (void)s_tile;
    (void)sv_tmem_d;
    (void)first_topk_block;
#endif
}

__device__ __forceinline__ void store_compact_sv_tmem_accumulator_to_global_theory(
    const SparseAttnFwdParams& params,
    NativeFp8TilePointers s_tile,
    uint32_t sv_tmem_d,
    int s_q_idx,
    int n_chunk,
    int effective_topk_length) {
    /*
    Final compact-SV epilogue for the theoretical PTX path.

    Each n_chunk accumulates O[:, n_chunk*128 : (n_chunk+1)*128] in TMEM.
    After the last topk block, normalize by row_l plus the optional attention
    sink and write the bf16 output chunk.  This mirrors finalize_debug_o_to_global()
    but avoids the debug float O buffer.
    */
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    FLASH_DEVICE_ASSERT(params.out != nullptr);
    FLASH_DEVICE_ASSERT(s_tile.row_m != nullptr);
    FLASH_DEVICE_ASSERT(s_tile.row_l != nullptr);
    FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);

    const int h = warpgroup_thread_idx();
    if (h >= params.h_q) {
        return;
    }

    float o[MX_SV_N_TILE];
    ku::tmem_ld_32dp32bNx<MX_SV_N_TILE>(sv_tmem_d, o);
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();

    const bool no_valid_topk = effective_topk_length <= 0;
    const float mi = no_valid_topk ? -CUDART_INF_F : s_tile.row_m[h];
    const float li = no_valid_topk ? 0.0f : s_tile.row_l[h];
    const float attn_sink =
        params.attn_sink == nullptr ?
        -CUDART_INF_F :
        __ldg(params.attn_sink + h) * CUDART_L2E_F;
    const float denom = li + exp2f(attn_sink - mi);
    const float output_scale = denom > 0.0f ? __fdividef(1.0f, denom) : 1.0f;
    const int dim_start = n_chunk * MX_SV_N_TILE;

    for (int local_dim = 0; local_dim < MX_SV_N_TILE; ++local_dim) {
        const int dim = dim_start + local_dim;
        const float value = no_valid_topk ? 0.0f : o[local_dim] * output_scale;
        params.out[
            static_cast<int64_t>(s_q_idx) * params.h_q * params.d_v +
            static_cast<int64_t>(h) * params.d_v +
            dim] = bf16(value);
    }

    if (n_chunk == 0) {
        const int global_index = s_q_idx * params.h_q + h;
        float cur_lse = no_valid_topk ? -CUDART_INF_F : logf(li) + mi * CUDART_LN2_F;
        cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;

        if (params.max_logits != nullptr) {
            params.max_logits[global_index] = mi * CUDART_LN2_F;
        }
        if (params.lse != nullptr) {
            params.lse[global_index] = cur_lse;
        }
    }
#else
    (void)params;
    (void)s_tile;
    (void)sv_tmem_d;
    (void)s_q_idx;
    (void)n_chunk;
    (void)effective_topk_length;
#endif
}

__device__ void emulate_mx_sv_mma_to_float(NativeFp8TilePointers tile, bool first_topk_block) {
    constexpr int NUM_K_GROUPS = MX_SV_K_GROUPS;
    FLASH_DEVICE_ASSERT(tile.s_scale != nullptr || tile.s_scale_layout != nullptr);
    FLASH_DEVICE_ASSERT(tile.v_scale != nullptr || tile.v_scale_layout != nullptr);
    FLASH_DEVICE_ASSERT(tile.out_tmem_or_debug != nullptr);
    const int lane = warpgroup_thread_idx();

    for (int task = lane; task < SV_M * SV_N; task += WARP_GROUP_THREADS) {
        const int h = task / SV_N;
        const int dim = task % SV_N;

        float sum =
            first_topk_block ?
            0.0f :
            tile.out_tmem_or_debug[row_major(h, dim, SV_N)] * tile.row_o_scale[h];
        for (int n = 0; n < SV_K; ++n) {
            const int k_group = n / MX_GROUP_K;
            const MxScale s_scale =
                load_mx_scale_factor(
                    tile.s_scale,
                    tile.s_scale_layout,
                    SV_M,
                    h,
                    k_group,
                    NUM_K_GROUPS);
            const MxScale v_scale =
                load_mx_scale_factor(
                    tile.v_scale,
                    tile.v_scale_layout,
                    SV_N,
                    dim,
                    k_group,
                    NUM_K_GROUPS);
            const float s =
                dequant_from_mx_fp8(tile.s_mx[native_mx_s_operand_offset(h, n)], s_scale);
            const float v =
                dequant_from_mx_fp8(tile.v_mx[native_mx_v_full_operand_offset(n, dim)], v_scale);
            sum += s * v;
        }
        tile.out_tmem_or_debug[row_major(h, dim, SV_N)] = sum;
    }

}

__device__ void emulate_mx_compact_sv_nchunk_to_float(
    NativeFp8TilePointers s_tile,
    NativeMxValueChunkPointers v_chunk,
    bool first_topk_block) {
    /*
    Debug fallback for the split-stage compact S@V path.

    This uses the same packed scale layouts that tcgen05.mma would consume:
      S scale: [head][token_k_group32]
      V scale: [local_value_dim_0..127][token_k_group32]

    It therefore checks the exact V3.2 -> MX scale rewrite for the compact
    value chunk, instead of relying on the optional MxScale debug side arrays.
    */
    constexpr int NUM_K_GROUPS = MX_SV_K_GROUPS;
    FLASH_DEVICE_ASSERT(s_tile.s_mx != nullptr);
    FLASH_DEVICE_ASSERT(s_tile.s_scale != nullptr || s_tile.s_scale_layout != nullptr);
    FLASH_DEVICE_ASSERT(v_chunk.v_mx != nullptr);
    FLASH_DEVICE_ASSERT(v_chunk.v_scale != nullptr || v_chunk.v_scale_layout != nullptr);
    FLASH_DEVICE_ASSERT(s_tile.out_tmem_or_debug != nullptr);
    FLASH_DEVICE_ASSERT(v_chunk.n_chunk >= 0 && v_chunk.n_chunk < MX_SV_N_CHUNKS);

    const int lane = warpgroup_thread_idx();
    const int dim_start = v_chunk.n_chunk * MX_SV_N_TILE;

    for (int task = lane; task < SV_M * MX_SV_N_TILE; task += WARP_GROUP_THREADS) {
        const int h = task / MX_SV_N_TILE;
        const int local_dim = task % MX_SV_N_TILE;
        const int dim = dim_start + local_dim;

        float sum =
            first_topk_block ?
            0.0f :
            s_tile.out_tmem_or_debug[row_major(h, dim, SV_N)] * s_tile.row_o_scale[h];
        for (int n = 0; n < SV_K; ++n) {
            const int k_group = n / MX_GROUP_K;
            const MxScale s_scale =
                load_mx_scale_factor(
                    s_tile.s_scale,
                    s_tile.s_scale_layout,
                    SV_M,
                    h,
                    k_group,
                    NUM_K_GROUPS);
            const MxScale v_scale =
                load_mx_scale_factor(
                    v_chunk.v_scale,
                    v_chunk.v_scale_layout,
                    MX_SV_N_TILE,
                    local_dim,
                    k_group,
                    NUM_K_GROUPS);
            const float s =
                dequant_from_mx_fp8(s_tile.s_mx[native_mx_s_operand_offset(h, n)], s_scale);
            const float v =
                dequant_from_mx_fp8(
                    v_chunk.v_mx[native_mx_v_compact_operand_offset(n, local_dim)],
                    v_scale);
            sum += s * v;
        }
        s_tile.out_tmem_or_debug[row_major(h, dim, SV_N)] = sum;
    }
}

__device__ void finalize_debug_o_to_global(
    const SparseAttnFwdParams& params,
    NativeFp8TilePointers persistent,
    int s_q_idx,
    int effective_topk_length) {
    /*
    Debug epilogue for the native-MX shell.

    The MX QK/S@V path accumulates:

      O_accum[h, d] = sum_tokens exp2(logit_base2 - row_m[h]) * V[token, d]
      row_l[h]      = sum_tokens exp2(logit_base2 - row_m[h])

    The final output is normalized by row_l plus the optional attention sink,
    matching the existing bf16 prefill epilogue.  Production should read O from
    TMEM and store through the existing TMA epilogue path; this helper exists so
    the non-PTX split-stage emulator has a closed attention computation.
    */
    FLASH_DEVICE_ASSERT(params.out != nullptr);
    FLASH_DEVICE_ASSERT(params.h_q <= QK_M);
    FLASH_DEVICE_ASSERT(params.d_v == D_V);
    FLASH_DEVICE_ASSERT(persistent.out_tmem_or_debug != nullptr);
    FLASH_DEVICE_ASSERT(persistent.row_m != nullptr);
    FLASH_DEVICE_ASSERT(persistent.row_l != nullptr);

    const int lane = warpgroup_thread_idx();
    const bool no_valid_topk = effective_topk_length <= 0;

    for (int task = lane; task < params.h_q * D_V; task += WARP_GROUP_THREADS) {
        const int h = task / D_V;
        const int dim = task % D_V;
        const float mi = no_valid_topk ? -CUDART_INF_F : persistent.row_m[h];
        const float li = no_valid_topk ? 0.0f : persistent.row_l[h];
        const float attn_sink =
            params.attn_sink == nullptr ?
            -CUDART_INF_F :
            __ldg(params.attn_sink + h) * CUDART_L2E_F;
        const float denom = li + exp2f(attn_sink - mi);
        const float output_scale = denom > 0.0f ? __fdividef(1.0f, denom) : 1.0f;
        const float value =
            no_valid_topk ?
            0.0f :
            persistent.out_tmem_or_debug[row_major(h, dim, SV_N)] * output_scale;
        params.out[
            static_cast<int64_t>(s_q_idx) * params.h_q * params.d_v +
            static_cast<int64_t>(h) * params.d_v +
            dim] = bf16(value);
    }

    for (int h = lane; h < params.h_q; h += WARP_GROUP_THREADS) {
        const int global_index = s_q_idx * params.h_q + h;
        const float mi = no_valid_topk ? -CUDART_INF_F : persistent.row_m[h];
        const float li = no_valid_topk ? 0.0f : persistent.row_l[h];
        float cur_lse = no_valid_topk ? -CUDART_INF_F : logf(li) + mi * CUDART_LN2_F;
        cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;

        if (params.max_logits != nullptr) {
            params.max_logits[global_index] = mi * CUDART_LN2_F;
        }
        if (params.lse != nullptr) {
            params.lse[global_index] = cur_lse;
        }
    }
}

#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
template <cute::UMMA::Major BMajor>
__device__ __forceinline__ void issue_mxf8_mma_1sm_dense_cutlass_theory(
    NativeMxMmaCall call) {
    /*
    CUTLASS' SM100_MMA_MXF8F6F4_SS::fma is the canonical wrapper for:

      tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale

    Use it instead of duplicating raw asm so the idesc-hi extraction, elected
    lane rule, and B-major contract stay aligned with CUTLASS.  QK instantiates
    BMajor=K for K-major K tiles; S@V instantiates BMajor=MN for value tiles.
    */
    static_assert(QK_M == SV_M);
    static_assert(QK_N == MX_SV_N_TILE);
    using MmaOp = cute::SM100_MMA_MXF8F6F4_SS<
        cute::float_e4m3_t,
        cute::float_e4m3_t,
        float,
        cute::float_ue8m0_t,
        QK_M,
        QK_N,
        cute::UMMA::Major::K,
        BMajor>;
    const uint32_t scale_c = call.accumulate ? 1U : 0U;
    MmaOp::fma(
        call.a_desc,
        call.b_desc,
        call.d_tmem,
        scale_c,
        call.block_scaled_idesc,
        call.a_scale_tmem_addr,
        call.b_scale_tmem_addr);
}
#endif

__device__ __forceinline__ void issue_mxf8_qk_mma_1sm_dense_theory(
    NativeMxMmaCall call) {
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    issue_mxf8_mma_1sm_dense_cutlass_theory<cute::UMMA::Major::K>(call);
#else
    (void)call;
#endif
}

__device__ __forceinline__ void issue_mxf8_sv_mma_1sm_dense_theory(
    NativeMxMmaCall call) {
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    issue_mxf8_mma_1sm_dense_cutlass_theory<cute::UMMA::Major::MN>(call);
#else
    (void)call;
#endif
}

__device__ __forceinline__ void prepare_mx_scale_tmem_span_theory(
    NativeMxScaleTmemSpan span) {
    /*
    Stage one scale-factor block for the required TMEM upload.

    CUTLASS' SM100 block-scaled mainloop does this in
    sm100_blockscaled_mma_array_warpspecialized.hpp:

      - make_tensor<typename TiledMma::FrgTypeSFA/SFB>(...)
      - make_utccp_copy(SM100_UTCCP_4x32dp128bit_1cta, tCtSFA/SFB)
      - get_utccp_smem_desc_tensor(...)
      - copy(tiled_copy_s2t_SF, smem_sf_desc, tmem_sf_fragment)

    What must happen here in a B300-verified FlashMLA kernel:

      1. Materialize the bytes selected by mx_scale_span_byte_ptr(span, ...)
         into a CUTLASS-compatible SmemLayoutAtomSFA/SFB staging tensor.
         span.operand selects the A-side SFA or B-side SFB staging layout.
      2. Use the UTCCP 4x32dp128bit path above to copy the staging tensor into
         the UMMA::tmem_sf_frg fragment.
      3. Use span.tmem_addr, i.e. raw_pointer_cast(tCtSFA/SFB.data()), as the
         tsfa/tsfb operand in tcgen05.mma.

    A plain pointer to q/k/s/v_scale_layout is not a valid tcgen05 operand.
    The SMEM staging copy is implemented below.  When
    FLASHMLA_ENABLE_THEORETICAL_UTCCP is enabled, this function also emits the
    tcgen05.cp scale-fragment copy using span.smem_stage_desc and span.tmem_addr.
    */
    materialize_mx_scale_span_to_smem_stage(span);
    copy_mx_scale_smem_stage_to_tmem_theory(span);
    const bool is_a_scale = span.operand == NativeMxScaleOperand::SFA;
    (void)is_a_scale;
    (void)mx_scale_span_bytes(span);
    (void)span;
}

__device__ __forceinline__ void validate_mx_mma_atom_plan_theory(
    NativeMxMmaAtomPlan atom,
    NativeMxOperandRole expected_a,
    NativeMxOperandRole expected_b,
    bool expected_b_transpose) {
    /*
    Descriptor correctness cannot be proven in this CUDA-free environment, but
    the issue plan can still enforce its internal contract: call descriptors
    must match the operand spans, and every atom is K32.
    */
    FLASH_DEVICE_ASSERT(atom.call.a_desc == atom.a_operand.smem_desc);
    FLASH_DEVICE_ASSERT(atom.call.b_desc == atom.b_operand.smem_desc);
    FLASH_DEVICE_ASSERT(atom.tile.k == MX_GROUP_K);
    FLASH_DEVICE_ASSERT(atom.k_atom >= 0 && atom.k_atom < MX_SF_PER_MMA_K);
    FLASH_DEVICE_ASSERT(atom.a_operand.base != nullptr);
    FLASH_DEVICE_ASSERT(atom.b_operand.base != nullptr);
    FLASH_DEVICE_ASSERT(atom.a_operand.role == expected_a);
    FLASH_DEVICE_ASSERT(atom.b_operand.role == expected_b);
    FLASH_DEVICE_ASSERT(!atom.a_operand.transpose_for_mma);
    FLASH_DEVICE_ASSERT(atom.b_operand.transpose_for_mma == expected_b_transpose);
}

__device__ __forceinline__ void issue_mxf8_qk_tile_theory(
    const NativeMxMmaDescriptorSet& descs) {
    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        prepare_mx_scale_tmem_span_theory(descs.qk[k_chunk].a_scale_span);
        prepare_mx_scale_tmem_span_theory(descs.qk[k_chunk].b_scale_span);

        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            NativeMxMmaAtomPlan atom = descs.qk[k_chunk].atoms[k_atom];
            validate_mx_mma_atom_plan_theory(
                atom,
                NativeMxOperandRole::Q,
                NativeMxOperandRole::K,
                false);
            NativeMxMmaCall call = atom.call;
            call.accumulate = k_chunk != 0 || k_atom != 0;
            issue_mxf8_qk_mma_1sm_dense_theory(call);
        }
    }
}

__device__ __forceinline__ void issue_mxf8_qk_tile_theory(
    const NativeMxQkDescriptorSet& descs) {
    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        prepare_mx_scale_tmem_span_theory(descs.qk[k_chunk].a_scale_span);
        prepare_mx_scale_tmem_span_theory(descs.qk[k_chunk].b_scale_span);

        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            NativeMxMmaAtomPlan atom = descs.qk[k_chunk].atoms[k_atom];
            validate_mx_mma_atom_plan_theory(
                atom,
                NativeMxOperandRole::Q,
                NativeMxOperandRole::K,
                false);
            NativeMxMmaCall call = atom.call;
            call.accumulate = k_chunk != 0 || k_atom != 0;
            issue_mxf8_qk_mma_1sm_dense_theory(call);
        }
    }
}

__device__ __forceinline__ void issue_mxf8_sv_tile_theory(
    const NativeMxMmaDescriptorSet& descs,
    bool first_topk_block) {
    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        prepare_mx_scale_tmem_span_theory(descs.sv[n_chunk].a_scale_span);
        prepare_mx_scale_tmem_span_theory(descs.sv[n_chunk].b_scale_span);

        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            NativeMxMmaAtomPlan atom = descs.sv[n_chunk].atoms[k_atom];
            validate_mx_mma_atom_plan_theory(
                atom,
                NativeMxOperandRole::S,
                NativeMxOperandRole::V,
                true);
            NativeMxMmaCall call = atom.call;
            call.accumulate = !first_topk_block || k_atom != 0;
            issue_mxf8_sv_mma_1sm_dense_theory(call);
        }
    }
}

__device__ __forceinline__ void issue_mxf8_compact_sv_nchunk_theory(
    const NativeMxMmaIssuePlan& plan,
    bool first_topk_block) {
    prepare_mx_scale_tmem_span_theory(plan.a_scale_span);
    prepare_mx_scale_tmem_span_theory(plan.b_scale_span);

    for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
        NativeMxMmaAtomPlan atom = plan.atoms[k_atom];
        validate_mx_mma_atom_plan_theory(
            atom,
            NativeMxOperandRole::S,
            NativeMxOperandRole::V,
            true);
        NativeMxMmaCall call = atom.call;
        call.accumulate = !first_topk_block || k_atom != 0;
        issue_mxf8_sv_mma_1sm_dense_theory(call);
    }
}

template<class PipelineStorage>
__device__ void native_mxf8_prefill_pipeline_body_theory(
    const SparseAttnFwdParams& params,
    PipelineStorage storage,
    int s_q_idx,
    int effective_topk_length,
    int num_topk_blocks) {
    /*
    Improved native-MX pipeline.

    The first research shell was fully serial:

      Q transcode -> K transcode -> QK -> softmax/S transcode -> V transcode -> SV

    That is a bad schedule because K/V transcode is independent of current QK/SV
    and because V can be prepared while QK and softmax run.  A single kv_ready
    barrier would also be too conservative: it would force QK to wait for V even
    though QK only needs K.  The intended B300 dependency graph is double
    buffered with separate K and V readiness.  The storage budget above shows
    that this cannot be copied literally into shared memory as two full K/S/V
    stages plus persistent Q; production needs finer-grained staging or a split
    kernel.

      persistent:
        Q_mx, Q_scale packed once per query token
        O_tmem, row_m, row_l

      per stage b % 2:
        K_mx[b], K_scale[b]
        V_mx[b], V_scale[b]
        P_tmem_or_debug[b]
        S_mx[b], S_scale[b]

    Warpgroup roles:

      WG0 Softmax:
        waits qk_ready[stage]
        updates online row_m/row_l and old-O scale
        transcodes P -> S_mx + S_scale_layout
        signals s_ready[stage]

      WG1 KProducer:
        waits k_stage_free[stage]
        transcodes K_b to MX fp8 and packs K scale layout
        signals k_ready[stage]

      WG2 VProducer:
        waits v_stage_free[stage]
        transcodes V_b to MX fp8 and packs V scale layout
        research wrapper still builds all value chunks, but the intended
        production direction is V_nchunk[stage][value_dim128]
        signals v_ready[stage] for full-V staging, or v_chunk_ready[stage]
        carrying n_chunk for compact-V staging
        in production this also absorbs exact scale-factor byte encoding

      WG3 MMA:
        waits q_ready once
        waits k_ready[stage]
        issues QK mxf8 MMA
        signals qk_ready[stage]
        signals k_stage_free[stage]
        waits s_ready[stage]
        waits v_ready[stage]
        rescales old O using row_o_scale
        issues S@V mxf8 MMA
        signals v_stage_free[stage]

    KProducer and VProducer work on block b+1 while WG3/WG0 consume block b.
    QK and S@V both use Tensor Cores, so they are intentionally serialized
    inside WG3.  The overlap we want is:

      K transcode(next)      || QK(current)
      V transcode(current)   || QK(current) + softmax/S(current)
      V_nchunk transcode     || SV(current, previous value chunk)
      K/V transcode(next)    || SV(current)

    `PipelineStorage` is expected to provide:

      persistent_tile_ptrs()
      stage_tile_ptrs(int stage)
      stage_raw_mma_descs(int stage) when FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX is set

    where each returned NativeFp8TilePointers points at the right swizzled
    shared-memory/TMEM/debug buffers for that stage.  stage_raw_mma_descs()
    returns the low-level descriptors and scale-TMEM addresses; this file then
    builds the qk[5] and sv[4] issue plans with
    make_native_mx_mma_descriptor_set() so V3.2->MX scale ownership stays here.
    The actual mbarrier fields are not spelled out here because they must be
    verified with CUDA headers on B300; the wait/signal comments below are the
    contract.
    */

    const int warpgroup_idx = threadIdx.x / WARP_GROUP_THREADS;

    if (warpgroup_idx == NativeMxPipelineRole::Softmax) {
        NativeFp8TilePointers persistent = storage.persistent_tile_ptrs();

        transcode_q_to_mx(params, persistent, s_q_idx);
        // signal q_ready after all WG0 lanes finish

        for (int block = 0; block < num_topk_blocks; ++block) {
            const int stage = block % PIPE_STAGES;
            NativeFp8TilePointers tile = storage.stage_tile_ptrs(stage);
            tile.q_mx = persistent.q_mx;
            tile.q_scale = persistent.q_scale;
            tile.q_scale_layout = persistent.q_scale_layout;
            tile.q_scale_smem_stage = persistent.q_scale_smem_stage;
            tile.out_tmem_or_debug = persistent.out_tmem_or_debug;
            tile.row_m = persistent.row_m;
            tile.row_l = persistent.row_l;
            tile.row_o_scale = persistent.row_o_scale;

            // wait qk_ready[stage]
            transcode_online_softmax_to_mx(
                tile,
                params.sm_scale_div_log2,
                block,
                effective_topk_length,
                block == 0);
            // signal s_ready[stage] after all WG0 lanes finish
        }
    } else if (warpgroup_idx == NativeMxPipelineRole::KProducer) {
        for (int block = 0; block < num_topk_blocks; ++block) {
            const int stage = block % PIPE_STAGES;
            NativeFp8TilePointers tile = storage.stage_tile_ptrs(stage);

            // wait k_stage_free[stage]
            transcode_k_tile_to_mx(params, tile, s_q_idx, block, effective_topk_length);
            // signal k_ready[stage] after all WG1 lanes finish
        }
    } else if (warpgroup_idx == NativeMxPipelineRole::VProducer) {
        for (int block = 0; block < num_topk_blocks; ++block) {
            const int stage = block % PIPE_STAGES;
            NativeFp8TilePointers tile = storage.stage_tile_ptrs(stage);

            // wait v_stage_free[stage]
            transcode_v_tile_to_mx(params, tile, s_q_idx, block, effective_topk_length);
            // signal v_ready[stage] after all WG2 lanes finish
        }
    } else if (warpgroup_idx == NativeMxPipelineRole::Mma) {
        // wait q_ready
        for (int block = 0; block < num_topk_blocks; ++block) {
            const int stage = block % PIPE_STAGES;
            NativeFp8TilePointers persistent = storage.persistent_tile_ptrs();
            NativeFp8TilePointers tile = storage.stage_tile_ptrs(stage);
            tile.q_mx = persistent.q_mx;
            tile.q_scale = persistent.q_scale;
            tile.q_scale_layout = persistent.q_scale_layout;
            tile.q_scale_smem_stage = persistent.q_scale_smem_stage;
            tile.out_tmem_or_debug = persistent.out_tmem_or_debug;
            tile.row_m = persistent.row_m;
            tile.row_l = persistent.row_l;
            tile.row_o_scale = persistent.row_o_scale;

            // wait k_ready[stage]
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
            NativeMxRawDescriptorInputs raw_mma_descs = storage.stage_raw_mma_descs(stage);
            NativeMxMmaDescriptorSet mma_descs =
                make_native_mx_mma_descriptor_set(tile, raw_mma_descs);
            issue_mxf8_qk_tile_theory(mma_descs);
#else
            emulate_mx_qk_mma_to_float(tile);
#endif
            // signal qk_ready[stage]
            // signal k_stage_free[stage]

            // wait s_ready[stage]
            // wait v_ready[stage]
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
            issue_mxf8_sv_tile_theory(mma_descs, block == 0);
#else
            emulate_mx_sv_mma_to_float(tile, block == 0);
#endif
            // signal v_stage_free[stage]
        }
    }
}

template<class SplitStorage>
__device__ void native_mxf8_prefill_split_stage_pipeline_body_theory(
    const SparseAttnFwdParams& params,
    SplitStorage storage,
    int s_q_idx,
    int effective_topk_length,
    int num_topk_blocks) {
    /*
    Split-stage native-MX pipeline.

    This is the production-shaped pipeline for the V3.2 native MX experiment.
    It replaces the earlier full-stage skeleton above with a contract that
    keeps the resident operands bounded:

      persistent:
        Q_mx/Q_scale_layout and online row state

      qk stage:
        K_mx/K_scale_layout for one topk block

      sv stage:
        S_mx/S_scale_layout for one topk block
        one compact V_mx/V_scale_layout value chunk at a time

    Storage contract expected from SplitStorage:

      persistent_tile_ptrs()
      qk_stage_tile_ptrs(stage)
      sv_stage_s_tile_ptrs(stage)
      sv_stage_v_chunk_ptrs(stage, n_chunk)

      qk_stage_tmem_d(stage)
      qk_stage_raw_mma_descs(stage)
      sv_stage_raw_compact_descs(stage, n_chunk)

    The two descriptor methods are deliberately split: QK descriptors bind the
    persistent Q tile plus staged K tile, while each S@V descriptor binds the
    staged S tile plus exactly one compact V value chunk.  This forces the
    V3.2 -> MX scale rewrite to happen at the same granularity as tcgen05.mma.
    The compact V pointer may be the same physical address for every n_chunk;
    n_chunk selects the logical output dimension range and descriptor set.
    */

    const int warpgroup_idx = threadIdx.x / WARP_GROUP_THREADS;

    if (warpgroup_idx == NativeMxPipelineRole::Softmax) {
        NativeFp8TilePointers persistent = storage.persistent_tile_ptrs();

        transcode_q_to_mx(params, persistent, s_q_idx);
        storage.signal_q_ready();

        for (int block = 0; block < num_topk_blocks; ++block) {
            const int stage = block % PIPE_STAGES;
            NativeFp8TilePointers s_tile = storage.sv_stage_s_tile_ptrs(stage);

            storage.wait_qk_ready(block, stage);
            storage.wait_sv_stage_free(block, stage);
            uint32_t qk_tmem_d = 0;
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
            qk_tmem_d = storage.qk_stage_tmem_d(stage);
#endif
            transcode_online_softmax_from_qk_tmem_to_mx_theory(
                s_tile,
                qk_tmem_d,
                params.sm_scale_div_log2,
                block,
                effective_topk_length,
                block == 0);
            storage.signal_s_ready(block, stage);
        }

        storage.wait_final_sv_done();
#if !defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
        finalize_debug_o_to_global(params, persistent, s_q_idx, effective_topk_length);
#endif
    } else if (warpgroup_idx == NativeMxPipelineRole::KProducer) {
        for (int block = 0; block < num_topk_blocks; ++block) {
            const int stage = block % PIPE_STAGES;
            NativeFp8TilePointers qk_tile = storage.qk_stage_tile_ptrs(stage);

            storage.wait_qk_stage_free(block, stage);
            transcode_k_tile_to_mx(params, qk_tile, s_q_idx, block, effective_topk_length);
            storage.signal_k_ready(block, stage);
        }
    } else if (warpgroup_idx == NativeMxPipelineRole::VProducer) {
        for (int block = 0; block < num_topk_blocks; ++block) {
            const int stage = block % PIPE_STAGES;

            for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
                NativeMxValueChunkPointers v_chunk =
                    storage.sv_stage_v_chunk_ptrs(stage, n_chunk);

                storage.wait_v_chunk_free(block, stage, n_chunk);
                transcode_v_nchunk_to_compact_mx(
                    params,
                    v_chunk,
                    s_q_idx,
                    block,
                    effective_topk_length);
                storage.signal_v_chunk_ready(block, stage, n_chunk);
            }
        }
    } else if (warpgroup_idx == NativeMxPipelineRole::Mma) {
        NativeFp8TilePointers persistent = storage.persistent_tile_ptrs();

        storage.wait_q_ready();
        for (int block = 0; block < num_topk_blocks; ++block) {
            const int stage = block % PIPE_STAGES;
            NativeFp8TilePointers qk_tile =
                native_mx_attach_persistent_q(storage.qk_stage_tile_ptrs(stage), persistent);

            storage.wait_k_ready(block, stage);
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
            NativeMxQkDescriptorInputs raw_qk_descs = storage.qk_stage_raw_mma_descs(stage);
            NativeMxQkDescriptorSet qk_descs =
                make_native_mx_qk_descriptor_set(qk_tile, raw_qk_descs);
            issue_mxf8_qk_tile_theory(qk_descs);
#else
            if (native_mx_qk_debug_ready(qk_tile)) {
                emulate_mx_qk_mma_to_float(qk_tile);
            }
#endif
            storage.signal_qk_ready(block, stage);
            storage.signal_qk_stage_free(block, stage);

            storage.wait_s_ready(block, stage);
            NativeFp8TilePointers s_tile = storage.sv_stage_s_tile_ptrs(stage);
            for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
                NativeMxValueChunkPointers v_chunk =
                    storage.sv_stage_v_chunk_ptrs(stage, n_chunk);

                storage.wait_v_chunk_ready(block, stage, n_chunk);
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
                NativeMxCompactSvDescriptorInputs raw_sv_descs =
                    storage.sv_stage_raw_compact_descs(stage, n_chunk);
                scale_compact_sv_tmem_accumulator_theory(
                    s_tile,
                    raw_sv_descs.sv_tmem_d,
                    block == 0);
                NativeMxMmaIssuePlan sv_plan =
                    make_native_mx_compact_sv_issue_plan(s_tile, v_chunk, raw_sv_descs);
                issue_mxf8_compact_sv_nchunk_theory(sv_plan, block == 0);
                storage.signal_v_chunk_free(block, stage, n_chunk);
                if (block + 1 == num_topk_blocks) {
                    storage.wait_sv_chunk_done(block, stage, n_chunk);
                    store_compact_sv_tmem_accumulator_to_global_theory(
                        params,
                        s_tile,
                        raw_sv_descs.sv_tmem_d,
                        s_q_idx,
                        n_chunk,
                        effective_topk_length);
                }
#else
                if (native_mx_compact_sv_debug_ready(s_tile, v_chunk)) {
                    emulate_mx_compact_sv_nchunk_to_float(s_tile, v_chunk, block == 0);
                }
                storage.signal_v_chunk_free(block, stage, n_chunk);
#endif
            }
        }
        storage.signal_final_sv_done();
    }
}

template<class TileStorage>
__global__ void sparse_attn_v32_native_mxf8_prefill_kernel_theory(
    __grid_constant__ const SparseAttnFwdParams params,
    TileStorage storage) {
    /*
    One CTA handles one query token in this research shell.  The body below is
    no longer the original serial proof-of-concept or the full K/S/V stage
    sketch; it uses the split-stage producer/softmax/MMA pipeline described in
    native_mxf8_prefill_split_stage_pipeline_body_theory().
    */

    const int s_q_idx = blockIdx.x;
    const int requested_topk_length =
        params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int clamped_topk_length = requested_topk_length > 0 ? requested_topk_length : 0;
    const int effective_topk_length =
        clamped_topk_length < params.topk ? clamped_topk_length : params.topk;
    const int raw_num_topk_blocks = (effective_topk_length + QK_N - 1) / QK_N;
    const int num_topk_blocks = raw_num_topk_blocks > 0 ? raw_num_topk_blocks : 1;

    native_mxf8_prefill_split_stage_pipeline_body_theory(
        params,
        storage,
        s_q_idx,
        effective_topk_length,
        num_topk_blocks);
}

template<class SplitStorage>
__device__ void native_mxf8_prefill_single_vchunk_pipeline_body_theory(
    const SparseAttnFwdParams& params,
    SplitStorage storage,
    int s_q_idx,
    int n_chunk,
    int effective_topk_length,
    int num_topk_blocks) {
    /*
    Correctness-first single-value-chunk native-MX pipeline.

    The fully fused split-stage pipeline computes QK/softmax once per topk
    block and updates all four D_V=512 chunks.  That is the desired performance
    direction, but it requires all O chunks to remain live in TMEM while QK also
    needs a TMEM scratch tile.  The single-buffer bring-up kernel below avoids
    that TMEM pressure by launching one logical value chunk per CTA:

      grid.y = n_chunk in [0, 4)

    Each chunk recomputes QK and S, then updates/stores only its own O chunk.
    This is more expensive, but it keeps the native-MX scale/descriptor path
    executable while the final compact TMEM layout is still being verified.
    */
    FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);
    const int warpgroup_idx = threadIdx.x / WARP_GROUP_THREADS;

    if (warpgroup_idx == NativeMxPipelineRole::Softmax) {
        NativeFp8TilePointers persistent = storage.persistent_tile_ptrs();

        transcode_q_to_mx(params, persistent, s_q_idx);
        storage.signal_q_ready();

        for (int block = 0; block < num_topk_blocks; ++block) {
            const int stage = block % PIPE_STAGES;
            NativeFp8TilePointers s_tile = storage.sv_stage_s_tile_ptrs(stage);

            storage.wait_qk_ready(block, stage);
            storage.wait_sv_stage_free_for_chunk(block, stage, n_chunk);
            uint32_t qk_tmem_d = 0;
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
            qk_tmem_d = storage.qk_stage_tmem_d(stage);
#endif
            transcode_online_softmax_from_qk_tmem_to_mx_theory(
                s_tile,
                qk_tmem_d,
                params.sm_scale_div_log2,
                block,
                effective_topk_length,
                block == 0);
            storage.signal_s_ready(block, stage);
        }

        storage.wait_final_sv_done();
#if !defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
        if (n_chunk == 0) {
            finalize_debug_o_to_global(params, persistent, s_q_idx, effective_topk_length);
        }
#endif
    } else if (warpgroup_idx == NativeMxPipelineRole::KProducer) {
        for (int block = 0; block < num_topk_blocks; ++block) {
            const int stage = block % PIPE_STAGES;
            NativeFp8TilePointers qk_tile = storage.qk_stage_tile_ptrs(stage);

            storage.wait_qk_stage_free(block, stage);
            transcode_k_tile_to_mx(params, qk_tile, s_q_idx, block, effective_topk_length);
            storage.signal_k_ready(block, stage);
        }
    } else if (warpgroup_idx == NativeMxPipelineRole::VProducer) {
        for (int block = 0; block < num_topk_blocks; ++block) {
            const int stage = block % PIPE_STAGES;
            NativeMxValueChunkPointers v_chunk =
                storage.sv_stage_v_chunk_ptrs(stage, n_chunk);

            storage.wait_sv_stage_free_for_chunk(block, stage, n_chunk);
            transcode_v_nchunk_to_compact_mx(
                params,
                v_chunk,
                s_q_idx,
                block,
                effective_topk_length);
            storage.signal_v_chunk_ready(block, stage, n_chunk);
        }
    } else if (warpgroup_idx == NativeMxPipelineRole::Mma) {
        NativeFp8TilePointers persistent = storage.persistent_tile_ptrs();

        storage.wait_q_ready();
        for (int block = 0; block < num_topk_blocks; ++block) {
            const int stage = block % PIPE_STAGES;
            NativeFp8TilePointers qk_tile =
                native_mx_attach_persistent_q(storage.qk_stage_tile_ptrs(stage), persistent);

            storage.wait_k_ready(block, stage);
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
            NativeMxQkDescriptorInputs raw_qk_descs = storage.qk_stage_raw_mma_descs(stage);
            NativeMxQkDescriptorSet qk_descs =
                make_native_mx_qk_descriptor_set(qk_tile, raw_qk_descs);
            issue_mxf8_qk_tile_theory(qk_descs);
#else
            if (native_mx_qk_debug_ready(qk_tile)) {
                emulate_mx_qk_mma_to_float(qk_tile);
            }
#endif
            storage.signal_qk_ready(block, stage);
            storage.signal_qk_stage_free(block, stage);

            storage.wait_s_ready(block, stage);
            storage.wait_v_chunk_ready(block, stage, n_chunk);
            NativeFp8TilePointers s_tile = storage.sv_stage_s_tile_ptrs(stage);
            NativeMxValueChunkPointers v_chunk =
                storage.sv_stage_v_chunk_ptrs(stage, n_chunk);

#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
            NativeMxCompactSvDescriptorInputs raw_sv_descs =
                storage.sv_stage_raw_compact_descs(stage, n_chunk);
            scale_compact_sv_tmem_accumulator_theory(
                s_tile,
                raw_sv_descs.sv_tmem_d,
                block == 0);
            NativeMxMmaIssuePlan sv_plan =
                make_native_mx_compact_sv_issue_plan(s_tile, v_chunk, raw_sv_descs);
            issue_mxf8_compact_sv_nchunk_theory(sv_plan, block == 0);
            storage.signal_v_chunk_free(block, stage, n_chunk);
            if (block + 1 == num_topk_blocks) {
                storage.wait_sv_chunk_done(block, stage, n_chunk);
                store_compact_sv_tmem_accumulator_to_global_theory(
                    params,
                    s_tile,
                    raw_sv_descs.sv_tmem_d,
                    s_q_idx,
                    n_chunk,
                    effective_topk_length);
            }
#else
            if (native_mx_compact_sv_debug_ready(s_tile, v_chunk)) {
                emulate_mx_compact_sv_nchunk_to_float(s_tile, v_chunk, block == 0);
            }
            storage.signal_v_chunk_free(block, stage, n_chunk);
#endif
        }
        storage.signal_final_sv_done();
    }
}

__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
sparse_attn_v32_native_mxf8_prefill_single_buffer_kernel_theory(
    __grid_constant__ const SparseAttnFwdParams params,
    __grid_constant__ const NativeMxSplitStageDescriptorMaterial material) {
    /*
    First launch-shaped native-MX bring-up kernel.

    This is still not the final registered production kernel: scale-factor
    UTCCP source descriptors and SFA/SFB TMEM fragment addresses must be built
    by a B300-verified CuTe descriptor layer.  With
    FLASHMLA_ENABLE_THEORETICAL_MXF8_REBUILD_SMEM_DESCS, Q/K/S/V payload SMEM
    descriptors are rebuilt from this CTA's actual dynamic shared-memory
    addresses.  What this kernel fixes is the surrounding execution contract:

      - one CTA per query token
      - grid.y selects one D_V=128 value chunk
      - dynamic shared storage using the single-buffer resident footprint
      - real split-stage barriers for the aliased QK/SV physical buffers
      - the same V3.2 -> MX transcode and tcgen05 issue pipeline as the shell

    In other words, the remaining gap is the scale-factor UTCCP/TMEM descriptor
    layer, not the high-level launch/storage/pipeline shape.
    */
    extern __shared__ char wksp_buf[];
    NativeMxSingleBufferSplitStageSharedStorage& smem =
        native_mx_single_buffer_storage_from_dynamic_smem(wksp_buf);

#if !defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_PTX)
    if (threadIdx.x == 0) {
        FLASH_DEVICE_ASSERT(false);
    }
    return;
#else
    init_native_mx_single_buffer_split_stage_barriers(smem);
    if (threadIdx.x < 32) {
        cute::TMEM::Allocator1Sm().allocate(512, &smem.tmem_start_addr);
#if !defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_SINGLE_BUFFER_DEFAULT_TMEM_PLAN)
        FLASH_DEVICE_ASSERT(smem.tmem_start_addr == material.tmem.tmem_base);
#endif
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    NativeMxTmemAddressPlan effective_tmem = material.tmem;
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_SINGLE_BUFFER_DEFAULT_TMEM_PLAN)
    effective_tmem =
        native_mx_make_single_buffer_tmem_address_plan_theory(smem.tmem_start_addr);
    validate_native_mx_tmem_address_plan(effective_tmem);
    NativeMxSplitStageDescriptorMaterial material_with_effective_tmem = material;
    material_with_effective_tmem.tmem = effective_tmem;
    const NativeMxSplitStageDescriptorMaterial& input_material =
        material_with_effective_tmem;
#else
    const NativeMxSplitStageDescriptorMaterial& input_material = material;
#endif

    NativeMxSplitStageStorageView ptrs =
        make_native_mx_single_buffer_split_stage_view(
            smem,
            nullptr,
            nullptr);
    NativeMxSplitStageBarrierPlan barriers =
        make_native_mx_single_buffer_split_stage_barrier_plan(smem);

#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_REBUILD_SMEM_DESCS)
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_REBUILD_SCALE_SMEM_DESCS)
    NativeMxSingleBufferScaleSmemDescriptorMaterial scale_descs =
        make_native_mx_single_buffer_scale_smem_descriptor_material_from_smem_theory(ptrs);
#else
    NativeMxSingleBufferScaleSmemDescriptorMaterial scale_descs =
        native_mx_extract_single_buffer_scale_smem_descriptor_material(input_material);
#endif
    NativeMxSplitStageDescriptorMaterial rebuilt_material =
        make_native_mx_single_buffer_descriptor_material_from_smem_theory(
            ptrs,
            effective_tmem,
            scale_descs);
    const NativeMxSplitStageDescriptorMaterial* effective_material =
        &rebuilt_material;
#else
    const NativeMxSplitStageDescriptorMaterial* effective_material = &input_material;
#endif

    NativeMxSplitStageKernelStorage storage =
        make_native_mx_material_split_stage_storage(
            ptrs,
            effective_material,
            &barriers);

    const int s_q_idx = blockIdx.x;
    const int n_chunk = blockIdx.y;
    FLASH_DEVICE_ASSERT(n_chunk >= 0 && n_chunk < MX_SV_N_CHUNKS);
    const int requested_topk_length =
        params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int clamped_topk_length = requested_topk_length > 0 ? requested_topk_length : 0;
    const int effective_topk_length =
        clamped_topk_length < params.topk ? clamped_topk_length : params.topk;
    const int raw_num_topk_blocks = (effective_topk_length + QK_N - 1) / QK_N;
    const int num_topk_blocks = raw_num_topk_blocks > 0 ? raw_num_topk_blocks : 1;

    native_mxf8_prefill_single_vchunk_pipeline_body_theory(
        params,
        storage,
        s_q_idx,
        n_chunk,
        effective_topk_length,
        num_topk_blocks);
#endif
}

inline int native_mx_host_stage_count(int count) {
    return count > 0 ? count : PIPE_STAGES;
}

inline bool native_mx_host_tmem_addr_in_capacity(uint32_t addr) {
    return native_mx_tmem_addr_column_bits(addr) < NATIVE_MX_TMEM_CAPACITY_COLS;
}

inline bool native_mx_host_tmem_accumulator_addr_in_capacity(uint32_t addr) {
    return native_mx_tmem_addr_sfid_bits(addr) == 0 &&
           native_mx_host_tmem_addr_in_capacity(addr);
}

inline uint32_t native_mx_host_tmem_addr_sfid(uint32_t addr) {
    return
        native_mx_tmem_addr_sfid_bits(addr) >>
        NATIVE_MX_TMEM_ADDR_SFID_SHIFT;
}

inline bool native_mx_host_scale_tmem_atom_addr_matches_block(
    uint32_t block_addr,
    uint32_t atom_addr) {
    return block_addr != 0 &&
           atom_addr != 0 &&
           native_mx_scale_tmem_block_has_k128_span(block_addr) &&
           native_mx_host_tmem_addr_in_capacity(atom_addr) &&
           native_mx_host_tmem_addr_sfid(block_addr) ==
               native_mx_host_tmem_addr_sfid(atom_addr);
}

inline bool native_mx_host_tmem_address_plan_looks_bound(
    const NativeMxTmemAddressPlan& tmem) {
    if (!native_mx_host_tmem_accumulator_addr_in_capacity(tmem.tmem_base) ||
        !native_mx_host_tmem_accumulator_addr_in_capacity(tmem.o_tmem_d)) {
        return false;
    }
    for (int stage = 0; stage < PIPE_STAGES; ++stage) {
        if (!native_mx_host_tmem_accumulator_addr_in_capacity(tmem.qk_tmem_d[stage])) {
            return false;
        }
    }
    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        if (!native_mx_host_tmem_accumulator_addr_in_capacity(tmem.sv_tmem_d[n_chunk])) {
            return false;
        }
        for (int stage = 0; stage < PIPE_STAGES; ++stage) {
            if (tmem.sv_tmem_d[n_chunk] == tmem.qk_tmem_d[stage]) {
                return false;
            }
        }
    }

    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        if (tmem.q_scale_tmem_block_addr[k_chunk] == 0 ||
            tmem.k_scale_tmem_block_addr[k_chunk] == 0 ||
            tmem.q_scale_tmem_block_addr[k_chunk] ==
                tmem.k_scale_tmem_block_addr[k_chunk] ||
            !native_mx_scale_tmem_block_has_k128_span(tmem.q_scale_tmem_block_addr[k_chunk]) ||
            !native_mx_scale_tmem_block_has_k128_span(tmem.k_scale_tmem_block_addr[k_chunk])) {
            return false;
        }
        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            if (tmem.q_scale_tmem_addr[k_chunk][k_atom] == 0 ||
                tmem.k_scale_tmem_addr[k_chunk][k_atom] == 0 ||
                !native_mx_host_tmem_addr_in_capacity(tmem.q_scale_tmem_addr[k_chunk][k_atom]) ||
                !native_mx_host_tmem_addr_in_capacity(tmem.k_scale_tmem_addr[k_chunk][k_atom]) ||
                !native_mx_host_scale_tmem_atom_addr_matches_block(
                    tmem.q_scale_tmem_block_addr[k_chunk],
                    tmem.q_scale_tmem_addr[k_chunk][k_atom]) ||
                !native_mx_host_scale_tmem_atom_addr_matches_block(
                    tmem.k_scale_tmem_block_addr[k_chunk],
                    tmem.k_scale_tmem_addr[k_chunk][k_atom]) ||
                tmem.q_scale_tmem_addr[k_chunk][k_atom] !=
                    native_mx_scale_tmem_atom_addr_from_block(
                        tmem.q_scale_tmem_block_addr[k_chunk],
                        k_atom) ||
                tmem.k_scale_tmem_addr[k_chunk][k_atom] !=
                    native_mx_scale_tmem_atom_addr_from_block(
                        tmem.k_scale_tmem_block_addr[k_chunk],
                        k_atom)) {
                return false;
            }
        }
    }

    if (tmem.s_scale_tmem_block_addr == 0 ||
        !native_mx_scale_tmem_block_has_k128_span(tmem.s_scale_tmem_block_addr)) {
        return false;
    }
    for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
        if (tmem.s_scale_tmem_addr[k_atom] == 0 ||
            !native_mx_host_tmem_addr_in_capacity(tmem.s_scale_tmem_addr[k_atom]) ||
            !native_mx_host_scale_tmem_atom_addr_matches_block(
                tmem.s_scale_tmem_block_addr,
                tmem.s_scale_tmem_addr[k_atom]) ||
            tmem.s_scale_tmem_addr[k_atom] !=
                native_mx_scale_tmem_atom_addr_from_block(
                    tmem.s_scale_tmem_block_addr,
                    k_atom)) {
            return false;
        }
    }
    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        if (tmem.v_scale_tmem_block_addr[n_chunk] == 0 ||
            tmem.v_scale_tmem_block_addr[n_chunk] ==
                tmem.s_scale_tmem_block_addr ||
            !native_mx_scale_tmem_block_has_k128_span(tmem.v_scale_tmem_block_addr[n_chunk])) {
            return false;
        }
        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            if (tmem.v_scale_tmem_addr[n_chunk][k_atom] == 0 ||
                !native_mx_host_tmem_addr_in_capacity(tmem.v_scale_tmem_addr[n_chunk][k_atom]) ||
                !native_mx_host_scale_tmem_atom_addr_matches_block(
                    tmem.v_scale_tmem_block_addr[n_chunk],
                    tmem.v_scale_tmem_addr[n_chunk][k_atom]) ||
                tmem.v_scale_tmem_addr[n_chunk][k_atom] !=
                    native_mx_scale_tmem_atom_addr_from_block(
                        tmem.v_scale_tmem_block_addr[n_chunk],
                        k_atom)) {
                return false;
            }
        }
    }

    return true;
}

inline bool native_mx_host_split_stage_material_looks_bound(
    const NativeMxSplitStageDescriptorMaterial& material) {
    const int qk_count = native_mx_host_stage_count(material.qk_material_stages);
    const int sv_count = native_mx_host_stage_count(material.sv_material_stages);
    if (!((qk_count == 1 || qk_count == PIPE_STAGES) &&
          (sv_count == 1 || sv_count == PIPE_STAGES))) {
        return false;
    }
    if (!native_mx_host_tmem_accumulator_addr_in_capacity(material.tmem.tmem_base) ||
        !native_mx_host_tmem_accumulator_addr_in_capacity(material.tmem.o_tmem_d)) {
        return false;
    }
    for (int stage = 0; stage < PIPE_STAGES; ++stage) {
        if (!native_mx_host_tmem_accumulator_addr_in_capacity(material.tmem.qk_tmem_d[stage])) {
            return false;
        }
    }
    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        if (!native_mx_host_tmem_accumulator_addr_in_capacity(material.tmem.sv_tmem_d[n_chunk])) {
            return false;
        }
        for (int stage = 0; stage < PIPE_STAGES; ++stage) {
            if (material.tmem.sv_tmem_d[n_chunk] == material.tmem.qk_tmem_d[stage]) {
                return false;
            }
        }
    }

    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        if (material.tmem.q_scale_tmem_block_addr[k_chunk] == 0 ||
            material.tmem.k_scale_tmem_block_addr[k_chunk] == 0 ||
            material.tmem.q_scale_tmem_block_addr[k_chunk] ==
                material.tmem.k_scale_tmem_block_addr[k_chunk] ||
            !native_mx_scale_tmem_block_has_k128_span(material.tmem.q_scale_tmem_block_addr[k_chunk]) ||
            !native_mx_scale_tmem_block_has_k128_span(material.tmem.k_scale_tmem_block_addr[k_chunk])) {
            return false;
        }
        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            if (material.tmem.q_scale_tmem_addr[k_chunk][k_atom] == 0 ||
                material.tmem.k_scale_tmem_addr[k_chunk][k_atom] == 0 ||
                !native_mx_host_tmem_addr_in_capacity(material.tmem.q_scale_tmem_addr[k_chunk][k_atom]) ||
                !native_mx_host_tmem_addr_in_capacity(material.tmem.k_scale_tmem_addr[k_chunk][k_atom]) ||
                !native_mx_host_scale_tmem_atom_addr_matches_block(
                    material.tmem.q_scale_tmem_block_addr[k_chunk],
                    material.tmem.q_scale_tmem_addr[k_chunk][k_atom]) ||
                !native_mx_host_scale_tmem_atom_addr_matches_block(
                    material.tmem.k_scale_tmem_block_addr[k_chunk],
                    material.tmem.k_scale_tmem_addr[k_chunk][k_atom]) ||
                material.tmem.q_scale_tmem_addr[k_chunk][k_atom] !=
                    native_mx_scale_tmem_atom_addr_from_block(
                        material.tmem.q_scale_tmem_block_addr[k_chunk],
                        k_atom) ||
                material.tmem.k_scale_tmem_addr[k_chunk][k_atom] !=
                    native_mx_scale_tmem_atom_addr_from_block(
                        material.tmem.k_scale_tmem_block_addr[k_chunk],
                        k_atom)) {
                return false;
            }
        }
    }

    if (material.tmem.s_scale_tmem_block_addr == 0 ||
        !native_mx_scale_tmem_block_has_k128_span(material.tmem.s_scale_tmem_block_addr)) {
        return false;
    }
    for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
        if (material.tmem.s_scale_tmem_addr[k_atom] == 0 ||
            !native_mx_host_tmem_addr_in_capacity(material.tmem.s_scale_tmem_addr[k_atom]) ||
            !native_mx_host_scale_tmem_atom_addr_matches_block(
                material.tmem.s_scale_tmem_block_addr,
                material.tmem.s_scale_tmem_addr[k_atom]) ||
            material.tmem.s_scale_tmem_addr[k_atom] !=
                native_mx_scale_tmem_atom_addr_from_block(
                    material.tmem.s_scale_tmem_block_addr,
                    k_atom)) {
            return false;
        }
    }
    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        if (material.tmem.v_scale_tmem_block_addr[n_chunk] == 0 ||
            material.tmem.v_scale_tmem_block_addr[n_chunk] ==
                material.tmem.s_scale_tmem_block_addr ||
            !native_mx_scale_tmem_block_has_k128_span(material.tmem.v_scale_tmem_block_addr[n_chunk])) {
            return false;
        }
        for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
            if (material.tmem.v_scale_tmem_addr[n_chunk][k_atom] == 0 ||
                !native_mx_host_tmem_addr_in_capacity(material.tmem.v_scale_tmem_addr[n_chunk][k_atom]) ||
                !native_mx_host_scale_tmem_atom_addr_matches_block(
                    material.tmem.v_scale_tmem_block_addr[n_chunk],
                    material.tmem.v_scale_tmem_addr[n_chunk][k_atom]) ||
                material.tmem.v_scale_tmem_addr[n_chunk][k_atom] !=
                    native_mx_scale_tmem_atom_addr_from_block(
                        material.tmem.v_scale_tmem_block_addr[n_chunk],
                        k_atom)) {
                return false;
            }
        }
    }

    for (int stage = 0; stage < qk_count; ++stage) {
        const NativeMxQkSmemDescriptorMaterial& qk = material.qk_smem_by_stage[stage];
        for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
            if (qk.q_scale_smem_desc[k_chunk] == 0 ||
                qk.k_scale_smem_desc[k_chunk] == 0) {
                return false;
            }
            for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
                if (qk.q_desc[k_chunk][k_atom] == 0 ||
                    qk.k_desc[k_chunk][k_atom] == 0) {
                    return false;
                }
            }
        }
    }

    for (int stage = 0; stage < sv_count; ++stage) {
        for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
            const NativeMxCompactSvSmemDescriptorMaterial& sv =
                material.sv_smem_by_stage[stage][n_chunk];
            if (sv.n_chunk != n_chunk ||
                sv.s_scale_smem_desc == 0 ||
                sv.v_scale_smem_desc == 0) {
                return false;
            }
            for (int k_atom = 0; k_atom < MX_SF_PER_MMA_K; ++k_atom) {
                if (sv.s_desc[k_atom] == 0 || sv.v_desc[k_atom] == 0) {
                    return false;
                }
            }
        }
    }

    return true;
}

inline bool native_mx_host_single_buffer_material_looks_bound(
    const NativeMxSplitStageDescriptorMaterial& material) {
    const int qk_material_stages =
        material.qk_material_stages > 0 ? material.qk_material_stages : PIPE_STAGES;
    const int sv_material_stages =
        material.sv_material_stages > 0 ? material.sv_material_stages : PIPE_STAGES;
    return qk_material_stages == 1 &&
           sv_material_stages == 1 &&
           native_mx_host_split_stage_material_looks_bound(material);
}

inline bool native_mx_host_single_buffer_scale_smem_material_looks_bound(
    const NativeMxSplitStageDescriptorMaterial& material) {
    const NativeMxQkSmemDescriptorMaterial& qk = material.qk_smem_by_stage[0];
    for (int k_chunk = 0; k_chunk < MX_QK_K_CHUNKS; ++k_chunk) {
        if (qk.q_scale_smem_desc[k_chunk] == 0 ||
            qk.k_scale_smem_desc[k_chunk] == 0) {
            return false;
        }
    }

    for (int n_chunk = 0; n_chunk < MX_SV_N_CHUNKS; ++n_chunk) {
        const NativeMxCompactSvSmemDescriptorMaterial& sv =
            material.sv_smem_by_stage[0][n_chunk];
        if (sv.n_chunk != n_chunk ||
            sv.s_scale_smem_desc == 0 ||
            sv.v_scale_smem_desc == 0) {
            return false;
        }
    }

    return true;
}

inline bool native_mx_host_single_buffer_launch_material_looks_bound(
    const NativeMxSplitStageDescriptorMaterial& material) {
    const int qk_material_stages =
        material.qk_material_stages > 0 ? material.qk_material_stages : PIPE_STAGES;
    const int sv_material_stages =
        material.sv_material_stages > 0 ? material.sv_material_stages : PIPE_STAGES;
    if (qk_material_stages != 1 || sv_material_stages != 1) {
        return false;
    }
    if (!native_mx_host_tmem_address_plan_looks_bound(material.tmem)) {
        return false;
    }

#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_REBUILD_SMEM_DESCS)
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_REBUILD_SCALE_SMEM_DESCS)
    return true;
#else
    return native_mx_host_single_buffer_scale_smem_material_looks_bound(material);
#endif
#else
    return native_mx_host_single_buffer_material_looks_bound(material);
#endif
}

void run_v32_native_mxf8_prefill_kernel_experimental(const SparseAttnFwdParams& params) {
    /*
    This is a research entry point only.  It is not build-wired because this
    machine has no CUDA installation and because the final MX scale descriptor
    layout must be validated on B300.

    Requirements:

      q:       [s_q, 64, 576], bf16 for the B300 target
      kv:      [num_blocks, page_block_size, 1, 656], V3.2 cache
      indices: [s_q, 1, topk], int32
      topk:    multiple of 128 for the production path

    Main cost we are deliberately accepting:

      V3.2 token-local scales are not MX-compatible, so K and V must be
      transcoded to MX block-scaled tiles before native fp8 MMA.

      For QK, NoPE is close enough to convert locally but is not passed through:
      the default path dequantizes the V3.2 dim128-scaled fp8 values and creates
      a fresh MX ue8m0 scale for each 32-wide GEMM-K group.  RoPE and padding
      are newly quantized.  For S@V, V scales must be rebuilt over 32-token
      reduction groups for every output dimension.

      For heads=64 this still uses a 128-row MX tile and zeros rows 64..127,
      because the documented block-scaled mx_float8 tile shapes start at 128
      rows.  That garbage work is part of the native-MX cost model.
    */
    KU_ASSERT(params.h_q == ACTIVE_H_B300);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.page_block_size > 0);
    KU_ASSERT(params.stride_kv_block >= params.page_block_size * V32_BYTES_PER_TOKEN);
    KU_ASSERT(params.s_kv == params.num_blocks * params.page_block_size);
}

void run_v32_native_mxf8_prefill_single_buffer_kernel_experimental(
    const SparseAttnFwdParams& params,
    const NativeMxSplitStageDescriptorMaterial& material) {
    /*
    Experimental B300 bring-up launcher for the native-MX path.

    This intentionally takes descriptor material explicitly.  The material is
    not derivable from the raw V3.2 cache: it must describe the staged shared
    memory operands, the 512B scale staging descriptors, the 1SM TMEM accumulator
    columns, and the SFA/SFB K128 TMEM scale-block bases.  The four K32 atom
    addresses are completed from those bases before validation and launch.
    During single-buffer bring-up,
    FLASHMLA_ENABLE_THEORETICAL_MXF8_SINGLE_BUFFER_DEFAULT_TMEM_PLAN can derive
    the TMEM layout inside the kernel from Allocator1Sm(), but this still leaves
    SMEM and scale-source descriptors as explicit material unless their rebuild
    macros are also enabled.  With DEFAULT_TMEM_PLAN + REBUILD_SMEM_DESCS +
    REBUILD_SCALE_SMEM_DESCS, the launcher can accept zero descriptor material
    and let the kernel derive the bring-up descriptors from its dynamic storage.

    Do not route production dispatch here until the CuTe SMEM/UTCCP descriptor
    generation is implemented and verified on B300.
    */
    KU_ASSERT(params.h_q == ACTIVE_H_B300);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.page_block_size > 0);
    KU_ASSERT(params.stride_kv_block >= params.page_block_size * V32_BYTES_PER_TOKEN);
    KU_ASSERT(params.s_kv == params.num_blocks * params.page_block_size);

    NativeMxSplitStageDescriptorMaterial completed_material = material;
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_SINGLE_BUFFER_DEFAULT_TMEM_PLAN)
    completed_material.tmem =
        native_mx_make_single_buffer_tmem_address_plan_theory(0);
#else
    completed_material.tmem =
        native_mx_complete_scale_atom_tmem_addresses(completed_material.tmem);
#endif
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_REBUILD_SMEM_DESCS)
    completed_material.qk_material_stages = 1;
    completed_material.sv_material_stages = 1;
#endif
    KU_ASSERT(native_mx_host_single_buffer_launch_material_looks_bound(completed_material));

    auto kernel = &sparse_attn_v32_native_mxf8_prefill_single_buffer_kernel_theory;
    constexpr size_t smem_size = NATIVE_MX_SINGLE_BUFFER_DYNAMIC_SMEM_BYTES;
    KU_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    kernel<<<dim3(params.s_q, MX_SV_N_CHUNKS, 1), NUM_THREADS, smem_size, params.stream>>>(
        params,
        completed_material);
    KU_CHECK_KERNEL_LAUNCH();
}

void run_v32_native_mxf8_prefill_single_buffer_kernel_experimental(
    const SparseAttnFwdParams& params,
    const NativeMxSingleBufferDescriptorMaterialInputs& inputs) {
    NativeMxSplitStageDescriptorMaterial material =
        make_native_mx_single_buffer_descriptor_material(inputs);
    run_v32_native_mxf8_prefill_single_buffer_kernel_experimental(params, material);
}

void run_v32_native_mxf8_prefill_single_buffer_kernel_experimental(
    const SparseAttnFwdParams& params) {
    /*
    Minimal native-MX bring-up entry point.

    This is the path that actually exercises the "build MX descriptors from the
    V3.2 cache" experiment without asking the caller to manufacture descriptor
    material on the host.  It is only valid when the kernel is compiled to:

      - derive the single-buffer TMEM layout from Allocator1Sm(),
      - rebuild Q/K/S/V payload SMEM descriptors from dynamic shared memory,
      - rebuild the 512B scale-stage SMEM descriptors from the same storage via
        the CUTLASS-style UTCCP source partition path.

    Otherwise a caller still has to use the explicit-material overload above.
    */
#if defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_SINGLE_BUFFER_DEFAULT_TMEM_PLAN) && \
    defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_REBUILD_SMEM_DESCS) && \
    defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_REBUILD_SCALE_SMEM_DESCS) && \
    defined(FLASHMLA_ENABLE_THEORETICAL_MXF8_CUTE_UTCCP_SCALE_DESCS)
    NativeMxSplitStageDescriptorMaterial material{};
    material.qk_material_stages = 1;
    material.sv_material_stages = 1;
    run_v32_native_mxf8_prefill_single_buffer_kernel_experimental(params, material);
#else
    (void)params;
    KU_ASSERT(
        false,
        "no-material native MX single-buffer launch requires "
        "FLASHMLA_ENABLE_THEORETICAL_MXF8_SINGLE_BUFFER_DEFAULT_TMEM_PLAN, "
        "FLASHMLA_ENABLE_THEORETICAL_MXF8_REBUILD_SMEM_DESCS, "
        "FLASHMLA_ENABLE_THEORETICAL_MXF8_REBUILD_SCALE_SMEM_DESCS, and "
        "FLASHMLA_ENABLE_THEORETICAL_MXF8_CUTE_UTCCP_SCALE_DESCS");
#endif
}

}  // namespace sm100::fwd_v32_native_fp8_prefill
