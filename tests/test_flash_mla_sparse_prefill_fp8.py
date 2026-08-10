"""
Test for the fp8 sparse prefill kernel (SM100 only).

The kernel consumes the MODEL1-layout fp8 KV cache (the same format used by the sparse decoding
kernel; see quant.py, FP8KVCacheLayout.MODEL1_FP8Sparse) directly, dequantizing it on the fly in
shared memory. The reference therefore runs bf16 attention on the *dequantized* cache: the kernel's
in-smem dequantization (e4m3 -> bf16, times a power-of-two e8m0 scale) is exact in bf16, so the
remaining error budget is the same as the bf16 prefill kernel's.
"""
import dataclasses
import math
import random
import sys
import time
from typing import Optional, Tuple

import torch
import kernelkit as kk

import quant

from flash_mla import flash_mla_sparse_fwd

_counter = kk.Counter()


@dataclasses.dataclass
class TestParam:
    s_q: int
    s_kv: int   # Number of addressable tokens. Will be rounded up to a multiple of block_size
    topk: int   # Must be a multiple of 64
    block_size: int = 64
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = 512
    d_v: int = 512
    seed: int = -1  # -1: to be filled automatically
    check_correctness: bool = True
    is_all_indices_invalid: bool = False
    num_runs: int = 10
    have_attn_sink: bool = False
    have_topk_length: bool = False


@dataclasses.dataclass
class Testcase:
    q: torch.Tensor                     # [s_q, h_q, d_qk], bf16
    quant_k_cache: torch.Tensor         # [num_blocks, block_size, 1, 584], fp8_e4m3fn
    kv_dequant: torch.Tensor            # [s_kv_padded, d_qk], bf16; what the kernel "sees"
    indices: torch.Tensor               # [s_q, 1, topk], int32
    sm_scale: float
    attn_sink: Optional[torch.Tensor]   # [h_q], fp32
    topk_length: Optional[torch.Tensor] # [s_q], int32


def generate_testcase(p: TestParam) -> Testcase:
    torch.manual_seed(p.seed)
    random.seed(p.seed)

    num_blocks = (p.s_kv + p.block_size - 1) // p.block_size
    s_kv_padded = num_blocks * p.block_size

    # Match the data distribution the upstream bf16 prefill test calibrates its
    # tolerances against (randn/10, clamped to +-1): the kernel's bf16 rounding
    # error scales with the output magnitude, so hotter inputs would blow the
    # calibrated abs_tol without indicating any kernel bug. On top of the base
    # distribution, scale each 64-dim NoPE group DOWN by a random power of two:
    # this keeps the magnitude within the calibrated range while still giving
    # different groups different e8m0 scales (power-of-two scaling is exact in
    # bf16, so it adds no quantization error of its own).
    kv = (torch.randn((num_blocks, p.block_size, p.h_kv, p.d_qk), dtype=torch.bfloat16) / 10).clamp_(-1, 1)
    group_exp = torch.randint(0, 4, (num_blocks, p.block_size, p.h_kv, 7, 1), dtype=torch.float32)
    kv[..., :448] = (
        kv[..., :448].view(num_blocks, p.block_size, p.h_kv, 7, 64) * (2.0 ** -group_exp)
    ).view(num_blocks, p.block_size, p.h_kv, 448).to(torch.bfloat16)
    q = (torch.randn((p.s_q, p.h_q, p.d_qk), dtype=torch.bfloat16) / 10).clamp_(-1, 1)

    quant_k_cache = quant.quantize_k_cache(kv, quant.FP8KVCacheLayout.MODEL1_FP8Sparse)
    kv_dequant = quant.dequantize_k_cache(quant_k_cache, quant.FP8KVCacheLayout.MODEL1_FP8Sparse)
    kv_dequant = kv_dequant.view(s_kv_padded, p.d_qk)

    if p.is_all_indices_invalid:
        indices = torch.full((p.s_q, p.h_kv, p.topk), 2147483647, dtype=torch.int32)
    else:
        indices = torch.randint(0, s_kv_padded, (p.s_q, p.h_kv, p.topk), dtype=torch.int32)
        # Sprinkle in some invalid entries of both kinds (-1 and >= s_kv)
        invalid_mask = torch.rand((p.s_q, p.h_kv, p.topk)) < 0.05
        indices[invalid_mask & (torch.rand_like(indices, dtype=torch.float32) < 0.5)] = -1
        indices[invalid_mask & (indices != -1) & (torch.rand_like(indices, dtype=torch.float32) < 0.5)] = s_kv_padded + 12345

    attn_sink = torch.randn((p.h_q,), dtype=torch.float32) if p.have_attn_sink else None
    topk_length = torch.randint(0, p.topk + 1, (p.s_q,), dtype=torch.int32) if p.have_topk_length else None

    sm_scale = p.d_qk ** (-0.5)
    return Testcase(q, quant_k_cache, kv_dequant, indices, sm_scale, attn_sink, topk_length)


def ref_sparse_attn_fwd_fp8(p: TestParam, t: Testcase) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mirrors ref.ref_sparse_attn_fwd, but gathers from the dequantized fp8 cache.
    Returns (out_fp32, max_logits, lse).
    """
    s_kv_padded = t.kv_dequant.size(0)
    indices = t.indices.clone().squeeze(1)  # [s_q, topk]
    if t.topk_length is not None:
        mask = torch.arange(p.topk, device=indices.device).unsqueeze(0).broadcast_to(p.s_q, p.topk) >= t.topk_length.unsqueeze(1)
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= s_kv_padded)
    indices[invalid_mask] = 0

    q = t.q.float()
    gathered_kv = t.kv_dequant.index_select(dim=0, index=indices.flatten().long()).reshape(p.s_q, p.topk, p.d_qk).float()
    P = (q @ gathered_kv.transpose(1, 2)) * t.sm_scale    # [s_q, h_q, topk]
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)   # [s_q, h_q]
    max_logits = P.max(dim=-1).values       # [s_q, h_q]

    lse_for_o = orig_lse
    if t.attn_sink is not None:
        lse_for_o = torch.logsumexp(
            torch.stack([orig_lse, t.attn_sink.broadcast_to(p.s_q, p.h_q)], dim=0), dim=0
        )
    lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float("+inf")  # So that corresponding O will be 0
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., :p.d_v]    # [s_q, h_q, d_v]

    lonely_q_mask = orig_lse == float("-inf")
    orig_lse[lonely_q_mask] = float("+inf")
    return out, max_logits, orig_lse


@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()

    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = generate_testcase(p)
    torch.cuda.synchronize()

    def run_prefill():
        return flash_mla_sparse_fwd(
            t.q, t.quant_k_cache, t.indices, t.sm_scale, p.d_v, t.attn_sink, t.topk_length
        )

    ans_out, ans_max_logits, ans_lse = run_prefill()
    torch.cuda.synchronize()

    if p.num_runs > 0:
        num_valid_topk = min(p.topk, p.s_kv)
        flop = 2 * p.s_q * p.h_q * num_valid_topk * (p.d_qk + p.d_v)
        mem_vol = p.s_q * p.h_q * p.d_qk * 2 + p.s_q * num_valid_topk * 584 + p.s_q * p.h_q * p.d_v * 2
        ans_time = kk.bench_kineto(run_prefill, num_tests=p.num_runs).get_kernel_time("sparse_attn_fwd")
        print(f"Prefill:  {ans_time*1e6:4.0f} us, {flop/ans_time/1e12:6.1f} TFlops, {mem_vol/ans_time/1e12:4.2f} TBps")

    if p.check_correctness:
        ref_out, ref_max_logits, ref_lse = ref_sparse_attn_fwd_fp8(p, t)
        ref_lse[ref_lse == float("-inf")] = float("+inf")
        torch.cuda.synchronize()

        is_correct = True
        is_correct &= kk.check_is_allclose("out", ans_out.float(), ref_out, abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
        is_correct &= kk.check_is_allclose("max_logits", ans_max_logits, ref_max_logits, abs_tol=1e-6, rel_tol=2.01/65536)
        is_correct &= kk.check_is_allclose("lse", ans_lse, ref_lse, abs_tol=1e-6, rel_tol=2.01/65536)
        return is_correct
    else:
        return True


if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    if torch.cuda.get_device_properties(device).major != 10:
        print("FP8 sparse prefill requires an SM100f GPU (B200 / B300). Skipping.")
        sys.exit(0)

    correctness_cases = [
        TestParam(s_q, s_kv, topk, num_runs=0)
        for s_kv, topk in [
            # Regular shapes
            (128, 128),
            (256, 256),
            (512, 512),

            # Irregular shapes
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),

            # Irregular shapes with OOB TopK
            (95, 128),
            (153, 256),
            (114, 384),
        ]
        for s_q in [1, 62, 213]
    ]

    correctness_cases_with_features = [
        TestParam(s_q, s_kv, topk, num_runs=0, have_attn_sink=have_attn_sink, have_topk_length=have_topk_length)
        for s_kv, topk in [
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),
            (95, 128),
            (153, 256),
        ]
        for s_q in [62, 213]
        for have_attn_sink in [False, True]
        for have_topk_length in [False, True]
    ]

    corner_cases = [
        TestParam(s_q, s_kv, topk, is_all_indices_invalid=True, num_runs=0, have_attn_sink=True, have_topk_length=True)
        for s_q, s_kv, topk in [
            (1, 128, 128),
            (1234, 4321, 4096),
        ]
    ] + [
        # Some blocks may not have any valid topk indices
        TestParam(s_q, s_kv, topk, num_runs=0, have_attn_sink=True, have_topk_length=True)
        for s_kv, topk in [
            (32, 2048),
            (64, 8192),
        ]
        for s_q in [1, 1024]
    ]

    performance_cases = [
        # MODEL1 CONFIG2-like shapes
        TestParam(4096, s_kv, 1024, have_attn_sink=True)
        for s_kv in [8192, 32768, 49152, 65536]
    ]

    testcases = correctness_cases + correctness_cases_with_features + corner_cases + performance_cases

    failed_cases = []
    for test in testcases:
        if test != testcases[0] and test.num_runs > 0:
            time.sleep(0.3)
        if not run_test(test):
            failed_cases.append(test)

    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {len(testcases)} cases failed:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
        sys.exit(1)
    else:
        print(f"\033[32m\033[1mAll {len(testcases)} cases passed!\033[0m")
