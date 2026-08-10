"""Adapter from the standalone benchmark to the native v8 extension symbol."""

import flash_mla.cuda as flash_mla_cuda


SUPPORTED_HEADS = {64, 128}


def make_runner(case, tensors, constants):
    if case.layout != "v32":
        raise RuntimeError("v8 only supports layout=v32")
    if case.heads not in SUPPORTED_HEADS:
        raise RuntimeError(
            f"v8 only supports H64/H128 execution buckets, got H{case.heads}"
        )
    if case.effective_topk % 64 != 0:
        raise RuntimeError(
            f"v8 requires topk % 64 == 0, got {case.effective_topk}"
        )

    # Preserve the full H128 tensor.  The bridge allocates one H128 result and
    # launches the H64 kernel twice into disjoint head slices; no cat/copy is
    # part of the measured path.
    q = tensors.q_bf16_flat

    def run():
        out, _, _ = flash_mla_cuda.v32_kernelv8_fp8_prefill_fwd(
            q,
            tensors.kv_flashmla_fp8,
            tensors.indices_flat,
            case.sm_scale,
            constants["HEAD_DIM_V"],
            None,
            None,
        )
        return out

    return run
