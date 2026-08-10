"""
Test for sparse prefill with the DeepSeek V3.2 fp8 KV cache (d_qk = 576).

This path is a fused flow rather than a new kernel: `flash_mla_sparse_fwd` detects the
656B-per-token V3.2 cache, dequantizes the whole pool to bf16, and runs the existing bf16
prefill kernel. The dequant pass is expected to be small from the byte model, but this test
prints it separately so B300 runs can verify the actual cost. Correctness decomposes into two
independently-checked pieces:

  1. `dequantize_v32_kv_cache` must match tests/quant.py's reference dequantization exactly.
  2. The end-to-end result must match bf16 attention over the dequantized cache, with the
     same tolerances as the bf16 prefill test.

The benchmark print separates:

  1. V3.2 pool dequantization time.
  2. CUDA-event end-to-end V3.2 fused path time (dequantization + sparse attention).
  3. Sparse-attention kernel time extracted by Kineto.

Do not compare only item 3 against trtllm / flashmla_kv numbers.
"""
import dataclasses
import sys
import time
from typing import Optional, Tuple

import torch
import kernelkit as kk

import quant

from flash_mla import flash_mla_sparse_fwd
from flash_mla.flash_mla_interface import dequantize_v32_kv_cache

_counter = kk.Counter()

D_QK = 576
D_V = 512
BLOCK_SIZE = 64


@dataclasses.dataclass
class TestParam:
    s_q: int
    s_kv: int
    topk: int   # Must be a multiple of 128 (the d_qk=576 prefill kernel's block size)
    h_q: int = 128
    seed: int = -1
    check_correctness: bool = True
    num_runs: int = 10
    have_attn_sink: bool = False
    have_topk_length: bool = False
    reuse_workspace: bool = False


@dataclasses.dataclass
class Testcase:
    q: torch.Tensor
    quant_k_cache: torch.Tensor         # [num_blocks, BLOCK_SIZE, 1, 656]
    kv_dequant: torch.Tensor            # [s_kv_padded, D_QK], bf16
    indices: torch.Tensor
    sm_scale: float
    attn_sink: Optional[torch.Tensor]
    topk_length: Optional[torch.Tensor]
    workspace: Optional[torch.Tensor]


def generate_testcase(p: TestParam) -> Testcase:
    torch.manual_seed(p.seed)
    num_blocks = (p.s_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    s_kv_padded = num_blocks * BLOCK_SIZE

    # Same distribution as the upstream bf16 prefill test (randn/10, clamp +-1):
    # the tolerances below are calibrated for it. Each 128-dim NoPE group is
    # additionally scaled DOWN by a random power of two so that the four fp32
    # scale factors per token differ from each other without inflating the
    # output magnitude (power-of-two scaling is exact in bf16).
    kv = (torch.randn((num_blocks, BLOCK_SIZE, 1, D_QK), dtype=torch.bfloat16) / 10).clamp_(-1, 1)
    group_exp = torch.randint(0, 4, (num_blocks, BLOCK_SIZE, 1, 4, 1), dtype=torch.float32)
    kv[..., :512] = (
        kv[..., :512].view(num_blocks, BLOCK_SIZE, 1, 4, 128) * (2.0 ** -group_exp)
    ).view(num_blocks, BLOCK_SIZE, 1, 512).to(torch.bfloat16)
    q = (torch.randn((p.s_q, p.h_q, D_QK), dtype=torch.bfloat16) / 10).clamp_(-1, 1)

    quant_k_cache = quant.quantize_k_cache(kv, quant.FP8KVCacheLayout.V32_FP8Sparse)

    # Check piece 1: our dequant must match the reference dequant exactly
    ref_dequant = quant.dequantize_k_cache(quant_k_cache, quant.FP8KVCacheLayout.V32_FP8Sparse)
    our_dequant = dequantize_v32_kv_cache(quant_k_cache)
    assert torch.equal(our_dequant.view(ref_dequant.shape), ref_dequant), \
        "dequantize_v32_kv_cache does not match quant.dequantize_k_cache"

    indices = torch.randint(0, s_kv_padded, (p.s_q, 1, p.topk), dtype=torch.int32)
    invalid_mask = torch.rand((p.s_q, 1, p.topk)) < 0.05
    indices[invalid_mask] = -1

    attn_sink = torch.randn((p.h_q,), dtype=torch.float32) if p.have_attn_sink else None
    topk_length = torch.randint(0, p.topk + 1, (p.s_q,), dtype=torch.int32) if p.have_topk_length else None
    workspace = torch.empty(num_blocks, BLOCK_SIZE, 1, D_QK, dtype=torch.bfloat16) if p.reuse_workspace else None

    return Testcase(
        q, quant_k_cache, our_dequant.view(s_kv_padded, D_QK), indices,
        D_QK ** (-0.5), attn_sink, topk_length, workspace,
    )


def ref_sparse_attn_fwd(p: TestParam, t: Testcase) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_kv_padded = t.kv_dequant.size(0)
    indices = t.indices.clone().squeeze(1)
    if t.topk_length is not None:
        mask = torch.arange(p.topk, device=indices.device).unsqueeze(0).broadcast_to(p.s_q, p.topk) >= t.topk_length.unsqueeze(1)
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= s_kv_padded)
    indices[invalid_mask] = 0

    q = t.q.float()
    gathered_kv = t.kv_dequant.index_select(dim=0, index=indices.flatten().long()).reshape(p.s_q, p.topk, D_QK).float()
    P = (q @ gathered_kv.transpose(1, 2)) * t.sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)
    max_logits = P.max(dim=-1).values

    lse_for_o = orig_lse
    if t.attn_sink is not None:
        lse_for_o = torch.logsumexp(
            torch.stack([orig_lse, t.attn_sink.broadcast_to(p.s_q, p.h_q)], dim=0), dim=0
        )
    lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float("+inf")
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., :D_V]

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
    if p.num_runs > 0 and t.workspace is None:
        t.workspace = torch.empty(
            t.quant_k_cache.shape[0], t.quant_k_cache.shape[1], 1, D_QK,
            dtype=torch.bfloat16,
            device=t.quant_k_cache.device,
        )
    torch.cuda.synchronize()

    def run_prefill():
        return flash_mla_sparse_fwd(
            t.q, t.quant_k_cache, t.indices, t.sm_scale, D_V, t.attn_sink, t.topk_length,
            dequant_workspace=t.workspace,
        )

    def run_dequant():
        return dequantize_v32_kv_cache(t.quant_k_cache, out=t.workspace)

    ans_out, ans_max_logits, ans_lse = run_prefill()
    torch.cuda.synchronize()

    if p.num_runs > 0:
        num_valid_topk = min(p.topk, p.s_kv)
        flop = 2 * p.s_q * p.h_q * num_valid_topk * (D_QK + D_V)
        num_cache_tokens = t.quant_k_cache.shape[0] * t.quant_k_cache.shape[1]
        dequant_bytes = num_cache_tokens * (656 + D_QK * 2)
        # bench_by_cuda_events reports one time per callable. The second callable
        # is run_prefill(), which includes both pool dequantization and sparse attention.
        dequant_time, fused_gpu_e2e_time = kk.bench_by_cuda_events(
            [run_dequant, run_prefill],
            num_warmups_each=3,
            num_runs_each=p.num_runs,
        )
        kineto_result = kk.bench_kineto(run_prefill, num_tests=p.num_runs)
        attn_time = kineto_result.get_kernel_time("sparse_attn_fwd")

        print(f"V3.2 dequant only:         {dequant_time*1e6:4.0f} us, {dequant_bytes/dequant_time/1e9:6.1f} GB/s")
        print(f"V3.2 fused GPU e2e:       {fused_gpu_e2e_time*1e6:4.0f} us, {flop/fused_gpu_e2e_time/1e12:6.1f} TFlops")
        print(f"V3.2 attn kernel only:    {attn_time*1e6:4.0f} us, {flop/attn_time/1e12:6.1f} TFlops")

    if p.check_correctness:
        ref_out, ref_max_logits, ref_lse = ref_sparse_attn_fwd(p, t)
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

    correctness_cases = [
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0,
                  have_attn_sink=have_attn_sink, have_topk_length=have_topk_length,
                  reuse_workspace=reuse_workspace)
        for h_q in [128, 64]
        for s_kv, topk in [
            (128, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),
            (95, 128),
            (33000, 2048),   # V3.2-realistic: DSA topk
        ]
        for s_q in [1, 62, 213]
        for have_attn_sink in [False, True]
        for have_topk_length in [False]
        for reuse_workspace in [s_q == 62]   # exercise the workspace path on one s_q
    ] + [
        TestParam(62, 1840, 256, num_runs=0, have_topk_length=True),
    ]

    performance_cases = [
        # B300 table-like sparse prefill: heads=64, NEW=875, KV-TOT=70000, topk=2048.
        TestParam(875, 70000, 2048, h_q=64, have_attn_sink=True, reuse_workspace=True),

        # Larger s_q/s_kv sweep for the byte-model claim; not a direct match to the B300 table.
        *[
            TestParam(4096, s_kv, 2048, h_q=128, have_attn_sink=True, reuse_workspace=True)
            for s_kv in [32768, 65536, 98304]
        ],
    ]

    testcases = correctness_cases + performance_cases

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
