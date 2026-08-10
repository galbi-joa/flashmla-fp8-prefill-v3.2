# FP8 Sparse Prefill on SM100 (B200 / B300): Design Notes

This document explains the design of the `SparseAttnFwdMode::PrefillFp8` mode added to
`csrc/sm100/prefill/sparse/fwd_for_small_topk/head128`, which lets the sparse prefill kernel
consume the **MODEL1-layout fp8 KV cache natively** — i.e., raw fp8 rows and e8m0 scales are
gathered by TMA and dequantized on the fly in shared memory, without ever materializing a bf16
copy of the cache in global memory.

## 0. The problem

The support matrix before this change:

| Kernel | KV cache format on SM100 |
| :---: | :---: |
| Sparse Decoding (`fwd_for_small_topk`, decode mode) | **FP8** (MODEL1: 448B e4m3 NoPE + 7×e8m0 scales + 64×bf16 RoPE = 584B/token) |
| Sparse Prefill (`fwd`, `fwd_for_small_topk` prefill mode) | **BF16 only** |

When serving a model with an fp8 KV cache, every prefill therefore required dequantizing the
cache into a bf16 tensor in HBM first (extra memory, extra bandwidth, extra kernel launch), even
though decoding already reads the fp8 cache directly. This mode closes that gap: prefill and
decode now share one fp8 cache.

## 1. Why the MMA stays in bf16 (the "native MX-FP8" analysis)

The obvious-sounding alternative is to feed e4m3 directly to the tensor cores via
`tcgen05.mma.kind::mxf8f6f4` (the cache's e8m0 per-64-dim scales even match the MX scale format,
modulo a 2× replication to per-32 blocks). We analyzed this carefully and rejected it for this
kernel shape. The short version:

1. **The existing bf16 "dual gemm" is already 100% MAC-efficient.** Each CTA of the 2-CTA cluster
   holds only *half of the head dimensions* (256 of 512) of every gathered token — this halves
   per-CTA TMA traffic and smem. The 2-SM UMMA (`M=128, N=B_TOPK*2=128, K=256`) then computes the
   two half-dim partial products as the two lane-halves of the accumulator, and the softmax
   warps sum them during the TMEM→register pass (`retrieve_mask_and_reduce_p`). Per 64-token
   block this is exactly `128 heads × 64 tokens × 512 dims` useful MACs — no waste.

2. **MX MMA shapes cannot express that trick.** `tcgen05.mma` MX kinds require `M == 128` for
   `cta_group::1` and `M == 256` for `cta_group::2` (see CUTLASS `SM100_MMA_MXF8F6F4_SS` /
   `SM100_MMA_MXF8F6F4_2x1SM_SS` static asserts). With the dim-split layout, every legal MX
   configuration either computes 50% garbage blocks (M=256 2-SM, or M=128 1-SM with packed
   `[head × dim-half]` rows) — which at 2× fp8 rate lands exactly back at bf16 wall-clock — or
   requires each CTA to hold all 512 dims, doubling gather traffic and/or overflowing TMEM
   (a full-head O accumulator is 128 lanes × 512 columns = the entire 256KB TMEM).

3. **S·V cannot be native fp8 with this cache format at all**: V's e8m0 scales vary per
   *token* (the reduction dimension of S·V) per 64-dim group, while MX scale factors must be
   shared by 32 consecutive reduction elements. Folding the scales into P would require 8
   separately-quantized P matrices (one per dim-group), i.e. 8× the softmax-side quantization
   work and 4× the S smem.

4. **Accuracy**: dequant-to-bf16 reproduces the quantization recipe exactly (e4m3 → bf16 is
   exact; multiplying by a power-of-two e8m0 scale is exact in bf16), so the kernel's numerics
   match `tests/quant.py`'s dequantization bit-for-bit. A native-MX QK would additionally
   quantize Q to e4m3. This mirrors the decoding kernel's published design choice ("uses an FP8
   KV cache while performing the matrix multiplication in bfloat16").

So on SM100's MMA shape constraints, "native fp8" buys nothing at the tensor core for this
kernel; what *is* worth taking is the fp8 cache's 2.8× smaller footprint and the fused in-smem
dequantization — which is what this mode implements.

## 2. What the new mode reuses (and from where)

`PrefillFp8` is a third instantiation of the same `KernelTemplate`, combining:

**From the decode mode (the KV-side data path), unchanged:**
- Index-transform warp (warp 9): converts topk indices into TMA gather coordinates, loads the
  per-token e8m0 scales (CTA0 takes scales s0..s3 for dims 0..255, CTA1 takes s4..s6 for dims
  256..447), and builds the validity bitmask. Only the addressing differs (no batch dim, no
  extra-KV, prefill validity rule `0 <= idx < s_kv`).
- Raw NoPE TMA producer: `tma_gather4` of 256B/CTA per token from the paged cache
  (`tensor_map_kv_nope`, 576B token stride) into the fp8 staging buffers `K_raw[2]`.
- RoPE TMA producer (warp 8, CTA1): gathers the 64×bf16 RoPE slice straight into CTA1's
  bf16 K buffer (RoPE lives in dims 448..511, i.e. CTA1's local tile 3) with the transaction
  counted on CTA0's `bar_KV_full`.
- Dequant warpgroup (WG1, 128 threads/CTA): e4m3×2 → bf16×2 with the e8m0 scale
  (`fp8x2_to_bf16x2_with_scale`), writing UMMA-canonical SW128 K-major bf16 tiles.
- Buffer counts: 3 bf16 K buffers + 2 raw fp8 buffers (vs 4 K buffers in bf16 prefill).

**From the bf16 prefill mode (control flow & epilogue), unchanged:**
- Persistent scheduling via CLC (CLC producer stays on warp 10; the NoPE TMA therefore runs on
  warp 11 alone, instead of warps 10+11 as in decode).
- Q path (TMA → smem → UTCCP to TMEM, dual-gemm layout), MMA issue loop, softmax warpgroup,
  full-bf16 output epilogue, and the `max_logits`/`lse` outputs.

The only genuinely new code is the fp8-prefill branch of the index-transform warp (~50 lines)
and the mode wiring (TmaParams variant, barrier arrival counts, host-side tensor maps).

### Barrier arrival counts (the part most worth double-checking on hardware)

| Barrier | bf16 prefill | decode | fp8 prefill |
| :--- | :---: | :---: | :---: |
| `bar_KV_full[i]` | 1 (TMA tx only) | 9 = 8 dequant-warp arrives + 1 RoPE expect_tx | 9 (same as decode) |
| `bar_valid_coord_scales_full[i]` | 8 (B_TOPK/8 lanes) | 32 | 32 |
| `bar_valid_coord_scales_empty[i]` | 128 | 128 + (cta==1) + 2 + 128 | 128 + (cta==1) + **1** + 128 |
| `bar_clc_empty` | 539 | – | **838** = (128+128+32+1+1+128)·2 + 2 |

`bar_clc_empty` counts every thread that participates in the persistent outer loop:
WG0 (128) + WG1 dequant (128) + warp 9 (32) + warp 10 CLC (1) + warp 11 NoPE TMA (1) +
WG3 softmax (128), times two CTAs, plus the MMA thread (CTA0) and the RoPE TMA thread (CTA1).

## 3. Cache layout addressing (MODEL1, from `tests/quant.py`)

Per page block (`stride(0)` must be a multiple of 576 — guaranteed by `quantize_k_cache`'s padding):

```
[ page_block_size × (448B e4m3 NoPE ++ 128B bf16 RoPE) ]   <- token rows, 576B each
[ page_block_size × (7B e8m0 scales ++ 1B padding) ]       <- scale rows, 8B each
[ padding up to a multiple of 576B ]
```

- TMA NoPE/RoPE coordinates: `block_idx * (stride_kv_block / 576) + idx_in_block`.
- Scale address: `kv_base + page_block_size*576 + block_idx*stride_kv_block + idx_in_block*8
  + (cta_idx == 1 ? 4 : 0)`.
- Invalid tokens get TMA coordinate `-1` (OOB ⇒ zero-fill) and are masked to −inf in softmax via
  the validity bitmask, identical to decode.

## 4. API

`flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v, attn_sink, topk_length)` dispatches on the
dtype of `kv`:
- bf16 ⇒ existing paths (sm90, sm100 `fwd`, sm100 `fwd_for_small_topk`).
- `float8_e4m3fn` / `int8` / `uint8` ⇒ `PrefillFp8`. Requirements: SM100f, `h_q == 128`,
  `d_qk == d_v == 512`, `topk % 64 == 0`, `kv` shaped `[num_blocks, page_block_size, 1, 584]`,
  indices encoded as `block_idx * page_block_size + offset` (decode convention),
  `s_kv := num_blocks * page_block_size`.

Returns `(out, max_logits, lse)` with the same semantics as the bf16 prefill kernel.

## 5. What about DeepSeek V3.2 (d_qk = 576, 656B fp32-scale cache)?

The `PrefillFp8` kernel itself is MODEL1-only (d_qk = 512): the small-topk template hardwires
D_Q = 512 (TMEM column map, dual-gemm tile split — 576 dims would split 288/288, which is not
64-tile-aligned), and the V3.2 cache's scales are fp32 per-128-dims, not e8m0. Extending the
kernel to V3.2 is a separate project.

However, a cost model suggests V3.2 fp8 prefill may be practical even before a new in-kernel
producer exists.
At the realistic claim shape (s_q = 10K new tokens, s_kv = 100K pool, topk = 2048):

| cost | bytes | time @ ~6 TB/s |
| :--- | ---: | ---: |
| dequantize the whole pool once per layer | 181 MB (read 656B + write 1152B per token) | ~30 µs |
| attention KV reads, bf16 kernel | 23.6 GB | (compute-bound: ~4 ms of MMA at 1450 TFlops) |
| attention KV reads, hypothetical fp8 kernel | 13.4 GB | saves ≤ 1.7 ms of reads, largely hidden under MMA |

The dequant pass is **0.77%** of the attention's own KV traffic in this byte model. At topk =
2048 the bf16 `fwd` kernel is expected to be compute-bound, so much of the bf16-vs-fp8
read-traffic difference may hide under the MMA. This is a model, not a substitute for B300
measurement; use the V3.2 test's dequant-only and fused GPU e2e prints for the actual call.
Therefore the supported V3.2 path is a fused flow in `flash_mla_sparse_fwd`:

1. `dequantize_v32_kv_cache(kv, out=workspace)` — one bf16 pass over the pool, exactly matching
   `tests/quant.py`'s reference dequantization (fp32 multiply, bf16 store). Pass a persistent
   `dequant_workspace` to reuse the buffer across layers / chunked-prefill steps.
2. The existing bf16 `fwd` prefill kernel (d_qk = 576, topk % 128 == 0) runs on the dequantized
   pool. Accuracy is identical to dequantizing offline.

The remaining gap to trtllm-gen's fp8 prefill is its **fp8 tensor-core rate** (its cache is plain
e4m3 *without* per-block scales, enabling plain `kind::f8f6f4` MMA at 2× bf16 throughput — a
different accuracy regime). Closing that on the FlashMLA side would require an fp8-MMA variant
of the `fwd` streaming kernel and a per-tensor-scale cache format; see §1 for why the
scale-preserving formats cannot use MX MMA here.

## 6. Other limitations / future work

- For MODEL1, `topk` beyond ~1280 would be better served by a streaming-pipeline variant
  (analogous to the bf16 `fwd` kernel); `PrefillFp8` handles any topk correctly but is tuned for
  the small-topk regime (MODEL1's production topk is 512–1024).
- No extra-KV support in prefill mode (decode-only feature).
- If a future cache format adopted per-256-dim (per CTA half) scales, the QK MMA could become a
  true `kind::f8f6f4` fp8 MMA with the scale folded post-hoc per N-column — halving QK MMA time.
  With the current per-64-dim format this is not possible (see §1).

## 7. Which Hopper scheduling ideas transfer to Blackwell?

The Hopper/H800 dense MLA decode kernel is useful reference material, but it should not be copied
literally into SM100 sparse prefill.  Hopper's "seesaw" schedule solves a dense-decode register
pressure problem: only one 64x512 output accumulator fits in registers, so two warpgroups split
the output into left/right halves and interleave two consecutive KV blocks.

SM100 sparse prefill has a different bottleneck.  It already uses TMEM, 2-CTA/2-SM cooperation,
sparse topk gather, and separate producer/softmax/MMA roles.  For fp8 prefill, the risk is not
"two output matrices do not fit"; the risk is that fp8 dequantization or MX transcode serializes
in front of QK/SV and starves Tensor Cores.

There are also two distinct SM100 prefill kernels in play:

- `fwd_for_small_topk/head128` is the MODEL1 `PrefillFp8` implementation described in this doc.
  It has `B_TOPK = 64`, a raw-fp8 NoPE buffer, fp8 dequant warps, a RoPE TMA warp, and one
  bf16 `K` shared-memory buffer that is interpreted as K for QK and as V for S@V.  Its readiness
  is therefore expressed through `bar_KV_full` / `bar_KV_empty`, plus the raw-KV and
  coord/scale barriers.  A literal K-ready/V-ready split would require a real shared-memory and
  mbarrier refactor, not just renaming a barrier.
- `fwd/head128` is the regular streaming sparse-prefill kernel used by the practical V3.2 d=576
  plan.  It has `B_TOPK = 128`, separate logical K and V layouts, and the finer
  `bar_k_part0_ready`, `bar_k_part1_ready`, `bar_v_part0_ready`, `bar_v_part1_ready` contract.
  This is the right base for replacing only the K/V producers with a V3.2 fp8 -> bf16 producer.

| Hopper idea | Transfer to SM100 sparse fp8 prefill? | Blackwell interpretation |
| :--- | :--- | :--- |
| Seesaw online-softmax math over two dense KV blocks | No direct transfer | The sparse prefill loop already has a QK -> softmax -> S@V dependency per topk tile, and QK/S@V both consume Tensor Cores. Splitting O into left/right register halves does not address the fp8 producer bottleneck. |
| Overlap CUDA-core work with Tensor-Core work | Yes | Move fp8 dequant/MX transcode into producer warpgroups and keep it off the MMA warpgroup's critical path. |
| Fine-grained TMA-copy -> GEMM pipelining | Partially | The regular V3.2 base exposes this as `bar_k_part0_ready`, `bar_k_part1_ready`, `bar_v_part0_ready`, and `bar_v_part1_ready`. The existing MODEL1 small-topk path instead uses a combined `bar_KV_full` because K/V share the same half-tile buffer. |
| Launch next memory work as soon as current data is no longer needed | Yes | In the V3.2 regular path, K for block `b+1` can be prepared while block `b` is in QK/softmax/SV, and V for block `b` only has to be ready before S@V. In the existing MODEL1 small-topk path, this principle is already approximated by separate raw-fp8, coord/scale, dequant, and MMA stages. |
| One combined KV-ready barrier | No for new V3.2 work | QK only needs K, while V is needed later by S@V. A new V3.2 producer should not introduce a single `kv_ready`. The existing MODEL1 small-topk `bar_KV_full` is a consequence of its shared K/V smem layout, so changing it is a larger kernel redesign. |
| Cache hints | Measure on SM100 | Q loads already use `EVICT_FIRST`; sparse KV gather currently uses `EVICT_LAST` in the bf16 prefill path. Hopper's result is not enough to change SM100 gather hints blindly because topk gather locality is different. |
| Programmatic Dependent Launch | Mostly not relevant to single-kernel prefill | Hopper decode uses PDL to overlap `splitkv_mla` with `combine`. Sparse prefill normally produces final `out/max_logits/lse` in one kernel. PDL may matter only if a future V3.2 path keeps a separate pool-dequant kernel and can safely overlap a consumer prologue. |
| Tile scheduler | Already present in small-topk/persistent form | For MODEL1 fp8 prefill, the CLC persistent scheduler is the relevant load-balancing mechanism. For regular large-topk V3.2 streaming prefill, the existing `fwd` grid is the starting point. |

This leads to the current Blackwell plan:

1. Leave the current MODEL1 small-topk `PrefillFp8` path as the reference for an in-kernel
   fp8-cache consumer, but do not assume its combined `bar_KV_full` layout is the best shape for
   V3.2 d=576.
2. For the practical V3.2 fp8 -> bf16 path, only replace the K/V producer. Keep the existing bf16
   Q path, softmax, QK UMMA, S@V UMMA, epilogue, and part-level barriers unchanged.
3. For the native MX research path, split the producer role into `KProducer` and `VProducer`.
   `KProducer` signals `k_ready` as soon as QK can start; `VProducer` signals `v_ready` only for
   S@V.  This is the SM100 equivalent of Hopper's "do not let non-Tensor-Core work sit in front
   of Tensor-Core work" rule.
4. Do not introduce a Hopper-style seesaw unless profiling shows the output accumulator or
   softmax rescale work, not fp8 producer work, has become the bottleneck.

## 8. Current implementation status and B300 gates

What is production-wired today:

- MODEL1 small-topk `PrefillFp8` is compiled through
  `csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_prefill_fp8_k512.cu`
  and dispatched by `Fwd_Sm100_Head128_Fp8_Impl`.
- V3.2 fp8 prefill is not an in-kernel fp8 producer today.  Python detects the 656B cache,
  calls `dequantize_v32_kv_cache`, reshapes the result to bf16 `[s_kv, 1, 576]`, then calls the
  regular bf16 sparse-prefill path.
- `csrc/sm100/prefill/sparse/fwd_v32_fp8_prefill.cu` and
  `csrc/sm100/prefill/sparse/fwd_v32_native_fp8_prefill.cu` are research shells.  They are not in
  `setup.py` and should not be described as production kernels.

The B300 performance gates should be:

1. Measure the current V3.2 fused path as at least two numbers: `dequantize_v32_kv_cache` time
   and `sparse_attn_fwd` time.  The V3.2 test now prints dequant-only, fused GPU e2e, and
   Kineto-extracted attention-kernel-only time.  Use the fused GPU e2e number for comparison
   against trtllm / flashmla_kv numbers.
2. For the V3.2 fp8 -> bf16 in-kernel producer experiment, first wire only the K/V producers to
   the existing d=576 regular prefill smem layouts.  The pass condition is: no global bf16 KV
   workspace, no numerical drift beyond the current bf16 route, and no new QK wait on V work.
3. For the native MX fp8 experiment, report transcode time separately from QK/S@V time.  The
   native route is only interesting if MX tensor-core savings exceed MX tile/scale rebuild cost.
4. Collect at least these counters when hardware is available: Tensor-Core active percentage,
   mbarrier wait stalls around K/V readiness, L2 hit rate for sparse gathers, TMA throughput, and
   dequant/transcode instruction time.  Cache hints should be changed only after this data.

## 9. Verification status

This implementation was written and reviewed offline (no SM100 hardware in the loop). Before
relying on it:

1. Build with CUDA ≥ 12.9 (`-gencode arch=compute_100f,code=sm_100f` is already in setup.py).
2. Run `python tests/test_flash_mla_sparse_prefill_fp8.py` on a B200/B300 for the MODEL1
   small-topk in-kernel fp8 path.
3. Run `python tests/test_flash_mla_sparse_prefill_fp8_v32.py` on a B200/B300 for the V3.2
   fused path.  Compare with external baselines using `V3.2 fused GPU e2e`, not
   `V3.2 attn kernel only`; the latter intentionally excludes the pool-dequant pass.
   The performance set includes a B300-table-like case: `h_q=64`, `s_q=875`, `s_kv=70000`,
   `topk=2048`.
4. If a MODEL1 hang is observed, audit the barrier arrival counts in §2 first (in particular
   `bar_clc_empty` = 838 and `bar_valid_coord_scales_empty` = `128 + (cta==1) + 1 + 128`),
   then the warp-role assignment (warp 10 = CLC, warp 11 = NoPE TMA).
5. Run `tests/test_flash_mla_sparse_prefill.py` and `tests/test_flash_mla_sparse_decoding.py`
   to confirm the bf16-prefill and decode modes are unaffected (they share the template).
