#!/usr/bin/env python3
"""Focused H64/H128 correctness and launch checks for native V3.2 FP8 v8.

The H128 entry point deliberately launches the same H64 kernel twice into
disjoint slices of one preallocated result.  This single test program covers:

* H64 correctness, including the instrumented H64 trace entry point;
* the full H128 result matches two explicit H64 calls bit-for-bit;
* output, max-logit, and LSE match the dequantized V3.2 reference;
* non-contiguous H64 Q slices, H128 row strides, attention sinks, topk lengths,
  padded page strides, and a non-default stream; and
* an optional CUPTI audit proving H64 launches once and H128 launches twice,
  with no concatenate/copy/combine CUDA work.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import torch

import flash_mla.cuda as flash_mla_cuda
import test_flash_mla_sparse_prefill_fp8_v32 as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--heads",
        choices=("all", "64", "128"),
        default="all",
        help="head-count contract to test",
    )
    parser.add_argument(
        "--case",
        choices=(
            "all",
            "edge",
            "lengths",
            "early",
            "page",
            "stride",
            "poison",
            "extreme",
            "sink-inf",
            "canonical",
            "ring",
            "persistent",
        ),
        default="all",
    )
    parser.add_argument(
        "--launch-only",
        action="store_true",
        help="launch and synchronize without reading outputs",
    )
    parser.add_argument(
        "--audit-launches",
        action="store_true",
        help="after correctness, use Torch/CUPTI to audit the 1/2-launch topology",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--audit-warmup", type=int, default=2)
    parser.add_argument(
        "--trace-json",
        default="",
        help="optional Chrome trace path for the audited H128 call",
    )
    parser.add_argument(
        "--skip-h64-control",
        action="store_true",
        help="during launch audit, profile only H128",
    )
    return parser.parse_args()


def set_h64_edge_inputs(testcase: base.Testcase, param: base.TestParam) -> None:
    assert testcase.topk_length is not None
    requested = torch.tensor(
        [-7, param.topk + 17, 0, 1, 63, 64, 65, param.topk - 1,
         param.topk, 2 * param.topk],
        dtype=torch.int32,
        device=testcase.topk_length.device,
    )
    testcase.topk_length.copy_(requested[:param.s_q])
    testcase.indices[1, 0, :8] = torch.tensor(
        [63, 64, 65, -7, 1 << 30, 0, 0, 62],
        dtype=torch.int32,
        device=testcase.indices.device,
    )


def tensors_bitwise_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return (
        lhs.shape == rhs.shape
        and lhs.dtype == rhs.dtype
        and torch.equal(
            lhs.contiguous().view(torch.uint8),
            rhs.contiguous().view(torch.uint8),
        )
    )


def set_edge_inputs(testcase: base.Testcase, param: base.TestParam) -> None:
    assert testcase.topk_length is not None
    testcase.topk_length.copy_(
        torch.tensor(
            [-9, param.topk + 19, 0, 1, 63, 64, 65, 127, 128, 129],
            dtype=torch.int32,
            device=testcase.topk_length.device,
        )
    )
    testcase.indices[1, 0, :10] = torch.tensor(
        [62, 63, 64, 65, 126, 127, 128, 129, -7, 1 << 30],
        dtype=torch.int32,
        device=testcase.indices.device,
    )

    testcase.q[:, 64:, :].copy_(testcase.q[:, :64, :])

    # Exercise a real non-unit head stride, not just the larger row stride of a
    # contiguous H128 tensor.  Poisoned odd slots make an incorrect descriptor
    # or head-base calculation fail loudly.
    q_backing = torch.full(
        (param.s_q, 2 * param.h_q, base.D_QK),
        float("nan"),
        dtype=testcase.q.dtype,
        device=testcase.q.device,
    )
    q_backing[:, ::2, :].copy_(testcase.q)
    testcase.q = q_backing[:, ::2, :]
    assert testcase.q.stride(1) == 2 * base.D_QK

    # The upper half must use its own attention-sink pointer.  +inf forces its
    # output to zero while the lower half remains ordinary attention.
    assert testcase.attn_sink is not None
    testcase.attn_sink[:64] = float("-inf")
    testcase.attn_sink[64:] = float("inf")


def set_page_inputs(testcase: base.Testcase, _: base.TestParam) -> None:
    testcase.indices[0, 0, :13] = torch.tensor(
        [62, 63, 64, 65, 126, 127, 128, 129, 130, 191, 192, -3, 1 << 30],
        dtype=torch.int32,
        device=testcase.indices.device,
    )


def set_length_inputs(testcase: base.Testcase, param: base.TestParam) -> None:
    assert testcase.topk_length is not None
    testcase.topk_length.copy_(
        torch.tensor(
            [
                -7,
                param.topk + 17,
                0,
                1,
                63,
                64,
                65,
                127,
                128,
                129,
                param.topk - 1,
                param.topk,
                2 * param.topk,
            ],
            dtype=torch.int32,
            device=testcase.topk_length.device,
        )
    )


def set_invalid_poison(testcase: base.Testcase, _: base.TestParam) -> None:
    testcase.indices.fill_(-1)
    testcase.quant_k_cache.view(torch.uint8).fill_(0x7F)


def set_extreme_fp8_inputs(
    testcase: base.Testcase, param: base.TestParam
) -> None:
    raw = testcase.quant_k_cache[..., :512].view(torch.uint8)
    pattern = torch.tensor(
        [0x7E, 0xFE, 0x01, 0x81, 0x00, 0x38, 0xB8, 0x20],
        dtype=torch.uint8,
        device=raw.device,
    )
    raw.copy_(pattern.repeat(raw.numel() // pattern.numel()).view_as(raw))
    scales = testcase.quant_k_cache[..., 512:528].view(torch.float32)
    scale_pattern = torch.tensor(
        [
            0.73 * 2.0**-20,
            0.89 * 2.0**-12,
            1.07 * 2.0**-8,
            1.31 * 2.0**-4,
        ],
        dtype=torch.float32,
        device=scales.device,
    )
    scales.copy_(scale_pattern.expand_as(scales))
    testcase.kv_dequant = base.dequantize_v32_kv_cache(
        testcase.quant_k_cache
    ).view(-1, base.D_QK)
    testcase.indices.copy_(
        torch.arange(
            testcase.indices.numel(),
            dtype=torch.int32,
            device=testcase.indices.device,
        )
        .remainder_(testcase.kv_dequant.size(0))
        .view_as(testcase.indices)
    )
    original_q = testcase.q.clone()
    testcase.q.zero_()
    active_heads = tuple(head for head in (
        15,
        16,
        31,
        32,
        47,
        48,
        63,
        64,
        79,
        80,
        95,
        96,
        111,
        112,
        127,
    ) if head < param.h_q)
    testcase.q[:, active_heads] = original_q[:, active_heads]


def set_infinite_sink(testcase: base.Testcase, _: base.TestParam) -> None:
    assert testcase.attn_sink is not None
    testcase.attn_sink.fill_(float("inf"))


def set_single_block_padding(
    testcase: base.Testcase, _: base.TestParam
) -> None:
    page_size = testcase.quant_k_cache.size(1)
    bytes_per_token = testcase.quant_k_cache.size(3)
    storage = torch.empty(
        (1, page_size + 7, 1, bytes_per_token),
        dtype=testcase.quant_k_cache.dtype,
        device=testcase.quant_k_cache.device,
    )
    padded_view = storage[:, :page_size]
    padded_view.copy_(testcase.quant_k_cache)
    assert padded_view.stride(0) > page_size * bytes_per_token
    testcase.quant_k_cache = padded_view


def call_v8(
    q: torch.Tensor,
    testcase: base.Testcase,
    attn_sink: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return flash_mla_cuda.v32_kernelv8_fp8_prefill_fwd(
        q,
        testcase.quant_k_cache,
        testcase.indices,
        testcase.sm_scale,
        base.D_V,
        attn_sink,
        testcase.topk_length,
    )


@torch.inference_mode()
def run_h64_case(
    name: str,
    param: base.TestParam,
    mutate: Callable[[base.Testcase, base.TestParam], None] | None,
    launch_only: bool,
) -> bool:
    print(f"[v8-h64] {name}: {param}")
    testcase = base.generate_testcase(param)
    if mutate is not None:
        mutate(testcase, param)

    check_trace = name in {"edge", "early"} and not launch_only
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        answer = call_v8(testcase.q, testcase, testcase.attn_sink)
        traced_answer = (
            flash_mla_cuda.v32_kernelv8_fp8_prefill_trace(
                testcase.q,
                testcase.quant_k_cache,
                testcase.indices,
                testcase.sm_scale,
                base.D_V,
                0,
                testcase.attn_sink,
                testcase.topk_length,
            )
            if check_trace
            else None
        )
    stream.synchronize()

    if launch_only:
        return True

    if traced_answer is not None:
        traced_out, traced_max, traced_lse, timestamps, schema = traced_answer
        if int(schema.item()) != 5:
            raise AssertionError(f"trace schema: expected 5, got {int(schema.item())}")
        expected_shape = (16, (param.topk + 63) // 64 + 1, 16)
        if tuple(timestamps.shape) != expected_shape:
            raise AssertionError(
                f"trace timestamp shape: expected {expected_shape}, "
                f"got {tuple(timestamps.shape)}"
            )
        if int(timestamps[12, 0, 9].item()) == 0:
            raise AssertionError("trace BF16-wait point 9 was not recorded")
        for track in range(8, 12):
            if int(timestamps[track, 0, 6].item()) == 0:
                raise AssertionError(
                    f"trace DQ1 track {track} publish point 6 was not recorded"
                )
        for value_name, traced_value, untraced_value in (
            ("out", traced_out, answer[0]),
            ("max_logits", traced_max, answer[1]),
            ("lse", traced_lse, answer[2]),
        ):
            if not tensors_bitwise_equal(traced_value, untraced_value):
                raise AssertionError(
                    f"v8-h64.{name}: traced {value_name} is not bitwise equal"
                )

    ref_out, ref_max_logits, ref_lse = base.ref_sparse_attn_fwd(param, testcase)
    out, max_logits, lse = answer
    correct = True
    correct &= base.kk.check_is_allclose(
        f"h64.{name}.out",
        out.float(),
        ref_out,
        abs_tol=8e-4,
        rel_tol=3.01 / 128,
        cos_diff_tol=7e-6,
    )
    correct &= base.kk.check_is_allclose(
        f"h64.{name}.max_logits",
        max_logits,
        ref_max_logits,
        abs_tol=1e-6,
        rel_tol=2.01 / 65536,
    )
    correct &= base.kk.check_is_allclose(
        f"h64.{name}.lse",
        lse,
        ref_lse,
        abs_tol=1e-6,
        rel_tol=2.01 / 65536,
    )
    return bool(correct)


def explicit_h64x2(
    testcase: base.Testcase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    halves = []
    for head_base in (0, 64):
        # Keep the non-contiguous row stride from the original H128 tensor.  The
        # native bridge promises to consume arbitrary last-dimension-contiguous
        # Q strides, and the production H128 path relies on that same contract.
        q_half = testcase.q[:, head_base : head_base + 64, :]
        assert q_half.stride(0) == testcase.q.stride(0)
        assert q_half.stride(-1) == 1
        sink_half = (
            None
            if testcase.attn_sink is None
            else testcase.attn_sink[head_base : head_base + 64]
        )
        halves.append(call_v8(q_half, testcase, sink_half))

    return tuple(
        torch.cat((halves[0][idx], halves[1][idx]), dim=1)
        for idx in range(3)
    )


@torch.inference_mode()
def run_h128_case(
    name: str,
    param: base.TestParam,
    mutate: Callable[[base.Testcase, base.TestParam], None] | None,
    launch_only: bool,
) -> bool:
    print(f"[v8-h128] {name}: {param}")
    testcase = base.generate_testcase(param)
    if mutate is not None:
        mutate(testcase, param)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        answer = call_v8(testcase.q, testcase, testcase.attn_sink)
        explicit = None if launch_only else explicit_h64x2(testcase)
        repeat = (
            call_v8(testcase.q, testcase, testcase.attn_sink)
            if name == "edge" and not launch_only
            else None
        )
    stream.synchronize()

    if launch_only:
        return True

    assert explicit is not None
    out, max_logits, lse = answer
    if tuple(out.shape) != (param.s_q, 128, base.D_V):
        raise AssertionError(f"unexpected H128 output shape: {tuple(out.shape)}")
    if tuple(max_logits.shape) != (param.s_q, 128):
        raise AssertionError(
            f"unexpected H128 max-logit shape: {tuple(max_logits.shape)}"
        )
    if tuple(lse.shape) != (param.s_q, 128):
        raise AssertionError(f"unexpected H128 LSE shape: {tuple(lse.shape)}")
    if out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise AssertionError("H128 output must be contiguous bfloat16")
    if max_logits.dtype != torch.float32 or not max_logits.is_contiguous():
        raise AssertionError("H128 max_logits must be contiguous float32")
    if lse.dtype != torch.float32 or not lse.is_contiguous():
        raise AssertionError("H128 LSE must be contiguous float32")

    for value_name, h128_value, h64x2_value in zip(
        ("out", "max_logits", "lse"), answer, explicit
    ):
        if not tensors_bitwise_equal(h128_value, h64x2_value):
            raise AssertionError(
                f"{name}.{value_name}: H128 result is not bitwise equal "
                "to two explicit H64 calls"
            )

    if repeat is not None:
        for value_name, value, repeat_value in zip(
            ("out", "max_logits", "lse"), answer, repeat
        ):
            if not tensors_bitwise_equal(value, repeat_value):
                raise AssertionError(f"edge.{value_name}: H128 is not deterministic")

        if not tensors_bitwise_equal(max_logits[:, :64], max_logits[:, 64:]):
            raise AssertionError("sink-offset: equal Q halves have different maxima")
        if not tensors_bitwise_equal(lse[:, :64], lse[:, 64:]):
            raise AssertionError("sink-offset: equal Q halves have different LSE")
        if torch.count_nonzero(out[:, 64:]).item() != 0:
            raise AssertionError("sink-offset: +inf upper sink must produce zero output")

        rejection_calls = (
            (
                "H96",
                lambda: call_v8(
                    testcase.q[:, :96, :],
                    testcase,
                    testcase.attn_sink[:96],
                ),
                "expects h_q == 64 or 128",
            ),
            (
                "H128 sink H64",
                lambda: call_v8(
                    testcase.q,
                    testcase,
                    testcase.attn_sink[:64],
                ),
                "attn_sink",
            ),
            (
                "H128 trace",
                lambda: flash_mla_cuda.v32_kernelv8_fp8_prefill_trace(
                    testcase.q,
                    testcase.quant_k_cache,
                    testcase.indices,
                    testcase.sm_scale,
                    base.D_V,
                    0,
                    testcase.attn_sink,
                    testcase.topk_length,
                ),
                "trace currently supports h_q == 64 only",
            ),
        )
        for rejection_name, rejection_call, expected_text in rejection_calls:
            try:
                rejection_call()
            except RuntimeError as exc:
                if expected_text not in str(exc):
                    raise AssertionError(
                        f"{rejection_name}: unexpected error: {exc}"
                    ) from exc
            else:
                raise AssertionError(f"{rejection_name}: expected explicit rejection")

    ref_out, ref_max_logits, ref_lse = base.ref_sparse_attn_fwd(
        param, testcase
    )
    correct = True
    correct &= base.kk.check_is_allclose(
        f"{name}.out",
        out.float(),
        ref_out,
        abs_tol=8e-4,
        rel_tol=3.01 / 128,
        cos_diff_tol=7e-6,
    )
    correct &= base.kk.check_is_allclose(
        f"{name}.max_logits",
        max_logits,
        ref_max_logits,
        abs_tol=1e-6,
        rel_tol=2.01 / 65536,
    )
    correct &= base.kk.check_is_allclose(
        f"{name}.lse",
        lse,
        ref_lse,
        abs_tol=1e-6,
        rel_tol=2.01 / 65536,
    )

    return bool(correct)


AUDIT_MARKER_PREFIX = "v8_launch_audit::"


def build_audit_inputs(device: torch.device):
    generator = torch.Generator(device=device).manual_seed(12864)
    q128 = (
        torch.randn(
            (3, 128, base.D_QK),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    ).clamp_(-1, 1)
    kv = torch.zeros(
        (2, base.BLOCK_SIZE, 1, 656), dtype=torch.uint8, device=device
    )
    kv[..., 512:528].view(torch.float32).fill_(1.0)
    rope = torch.randn(
        (2, base.BLOCK_SIZE, 1, 64),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    kv[..., 528:656].view(torch.bfloat16).copy_(rope)
    indices = torch.randint(
        0,
        2 * base.BLOCK_SIZE,
        (3, 1, 64),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    indices[0, 0, :8] = torch.tensor(
        [63, 64, 127, -1, 1 << 30, 0, 0, 65],
        dtype=torch.int32,
        device=device,
    )
    topk_length = torch.tensor([-7, 1, 81], dtype=torch.int32, device=device)
    sink128 = torch.linspace(-2.0, 2.0, 128, dtype=torch.float32, device=device)
    return q128, kv, indices, base.D_QK**-0.5, sink128, topk_length


def is_cuda_activity(event) -> bool:
    return event.device_type == torch.autograd.DeviceType.CUDA


def profile_one_launch_contract(
    label: str,
    expected_launches: int,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    sink: torch.Tensor,
    topk_length: torch.Tensor,
    warmup: int,
    trace_json: str = "",
) -> None:
    for _ in range(warmup):
        flash_mla_cuda.v32_kernelv8_fp8_prefill_fwd(
            q, kv, indices, sm_scale, base.D_V, sink, topk_length
        )
    torch.cuda.synchronize(q.device)

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with torch.profiler.record_function(f"{AUDIT_MARKER_PREFIX}{label}"):
            result = flash_mla_cuda.v32_kernelv8_fp8_prefill_fwd(
                q, kv, indices, sm_scale, base.D_V, sink, topk_length
            )
        torch.cuda.synchronize(q.device)

    if tuple(result[0].shape) != (3, q.shape[1], base.D_V):
        raise AssertionError(f"{label}: unexpected output shape {tuple(result[0].shape)}")

    events = list(profiler.events())
    cuda_events = [event for event in events if is_cuda_activity(event)]
    marker_events = [
        event
        for event in cuda_events
        if event.name.startswith(AUDIT_MARKER_PREFIX)
    ]
    device_events = [
        event
        for event in cuda_events
        if not event.name.startswith(AUDIT_MARKER_PREFIX)
    ]
    names = [event.name for event in device_events]
    v8_events = [event for event in device_events if "v8_kernel" in event.name]

    print(
        f"[{label}] audited CUDA activities ({len(device_events)} total; "
        f"{len(marker_events)} marker event(s) excluded):"
    )
    for name, count in Counter(names).items():
        print(f"  {count} x {name}")

    if len(v8_events) != expected_launches:
        raise AssertionError(
            f"{label}: expected {expected_launches} v8_kernel launches, "
            f"observed {len(v8_events)}"
        )
    if len(device_events) != expected_launches:
        unexpected = [name for name in names if "v8_kernel" not in name]
        raise AssertionError(
            f"{label}: unexpected CUDA activity (combine/copy/memset): {unexpected}"
        )

    cpu_names = [event.name for event in events if not is_cuda_activity(event)]
    forbidden = [
        name
        for name in cpu_names
        if name in {"aten::cat", "aten::concat", "aten::concatenate"}
        or name.startswith("aten::cat.")
    ]
    if forbidden:
        raise AssertionError(f"{label}: forbidden tensor concatenation: {forbidden}")

    if trace_json:
        trace_path = Path(trace_json)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))
        print(f"[{label}] trace: {trace_path}")

    print(
        f"[PASS] {label}: exactly {expected_launches} launch(es), "
        "no CUDA combine/copy activity, no aten::cat"
    )


def run_launch_audit(args: argparse.Namespace) -> None:
    device = torch.device("cuda", args.device)
    q128, kv, indices, sm_scale, sink128, topk_length = build_audit_inputs(device)
    if not args.skip_h64_control:
        profile_one_launch_contract(
            "H64-control",
            1,
            q128[:, :64, :],
            kv,
            indices,
            sm_scale,
            sink128[:64],
            topk_length,
            args.audit_warmup,
        )
    profile_one_launch_contract(
        "H128",
        2,
        q128,
        kv,
        indices,
        sm_scale,
        sink128,
        topk_length,
        args.audit_warmup,
        args.trace_json,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.set_device(args.device)
    major, minor = torch.cuda.get_device_capability(args.device)
    if major != 10:
        raise SystemExit(f"native v8 requires SM100f, got SM {major}.{minor}")
    for symbol in (
        "v32_kernelv8_fp8_prefill_fwd",
        "v32_kernelv8_fp8_prefill_trace",
    ):
        if not hasattr(flash_mla_cuda, symbol):
            raise SystemExit(f"missing {symbol}; rebuild the extension")

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(f"cuda:{args.device}")
    torch.set_float32_matmul_precision("high")
    num_sms = torch.cuda.get_device_properties(args.device).multi_processor_count

    h64_cases = {
        "edge": (
            base.TestParam(
                s_q=10, s_kv=64, topk=64, h_q=64, seed=701,
                num_runs=0, have_attn_sink=True, have_topk_length=True,
            ),
            set_h64_edge_inputs,
        ),
        "stride": (
            base.TestParam(
                s_q=2, s_kv=63, topk=64, h_q=64, seed=706,
                num_runs=0, have_attn_sink=True,
            ),
            set_single_block_padding,
        ),
        "page": (
            base.TestParam(
                s_q=3, s_kv=130, topk=128, h_q=64, seed=702,
                num_runs=0, have_attn_sink=True,
            ),
            set_page_inputs,
        ),
        "lengths": (
            base.TestParam(
                s_q=13, s_kv=257, topk=192, h_q=64, seed=707,
                num_runs=0, have_attn_sink=True, have_topk_length=True,
            ),
            set_length_inputs,
        ),
        "early": (
            base.TestParam(
                s_q=2, s_kv=257, topk=192, h_q=64, seed=704,
                num_runs=0, have_attn_sink=True,
            ),
            None,
        ),
        "poison": (
            base.TestParam(
                s_q=1, s_kv=64, topk=64, h_q=64, seed=708, num_runs=0,
            ),
            set_invalid_poison,
        ),
        "extreme": (
            base.TestParam(
                s_q=1, s_kv=64, topk=64, h_q=64, seed=709, num_runs=0,
            ),
            set_extreme_fp8_inputs,
        ),
        "sink-inf": (
            base.TestParam(
                s_q=1, s_kv=64, topk=64, h_q=64, seed=710,
                num_runs=0, have_attn_sink=True,
            ),
            set_infinite_sink,
        ),
        "canonical": (
            base.TestParam(
                s_q=1, s_kv=2113, topk=2048, h_q=64, seed=705,
                num_runs=0, have_attn_sink=True,
            ),
            None,
        ),
        "ring": (
            base.TestParam(
                s_q=1, s_kv=5000, topk=4096, h_q=64, seed=703, num_runs=0,
            ),
            None,
        ),
        "persistent": (
            base.TestParam(
                s_q=num_sms + 1, s_kv=64, topk=64, h_q=64, seed=711,
                num_runs=0, have_attn_sink=True, have_topk_length=True,
            ),
            None,
        ),
    }

    h128_cases = {
        "edge": (
            base.TestParam(
                s_q=10,
                s_kv=130,
                topk=128,
                h_q=128,
                seed=801,
                num_runs=0,
                have_attn_sink=True,
                have_topk_length=True,
            ),
            set_edge_inputs,
        ),
        "stride": (
            base.TestParam(
                s_q=3,
                s_kv=63,
                topk=64,
                h_q=128,
                seed=802,
                num_runs=0,
                have_attn_sink=True,
                have_topk_length=True,
            ),
            set_single_block_padding,
        ),
        "page": (
            base.TestParam(
                s_q=3,
                s_kv=130,
                topk=128,
                h_q=128,
                seed=804,
                num_runs=0,
                have_attn_sink=True,
            ),
            set_page_inputs,
        ),
        "lengths": (
            base.TestParam(
                s_q=13,
                s_kv=257,
                topk=192,
                h_q=128,
                seed=805,
                num_runs=0,
                have_attn_sink=True,
                have_topk_length=True,
            ),
            set_length_inputs,
        ),
        "early": (
            base.TestParam(
                s_q=2,
                s_kv=257,
                topk=192,
                h_q=128,
                seed=806,
                num_runs=0,
                have_attn_sink=True,
            ),
            None,
        ),
        "poison": (
            base.TestParam(
                s_q=1,
                s_kv=64,
                topk=64,
                h_q=128,
                seed=807,
                num_runs=0,
            ),
            set_invalid_poison,
        ),
        "extreme": (
            base.TestParam(
                s_q=1,
                s_kv=64,
                topk=64,
                h_q=128,
                seed=808,
                num_runs=0,
            ),
            set_extreme_fp8_inputs,
        ),
        "sink-inf": (
            base.TestParam(
                s_q=1,
                s_kv=64,
                topk=64,
                h_q=128,
                seed=809,
                num_runs=0,
                have_attn_sink=True,
            ),
            set_infinite_sink,
        ),
        "canonical": (
            base.TestParam(
                s_q=2,
                s_kv=2113,
                topk=2048,
                h_q=128,
                seed=803,
                num_runs=0,
                have_attn_sink=True,
            ),
            None,
        ),
        "ring": (
            base.TestParam(
                s_q=1,
                s_kv=5000,
                topk=4096,
                h_q=128,
                seed=810,
                num_runs=0,
            ),
            None,
        ),
        "persistent": (
            base.TestParam(
                s_q=num_sms + 1,
                s_kv=64,
                topk=64,
                h_q=128,
                seed=811,
                num_runs=0,
                have_attn_sink=True,
                have_topk_length=True,
            ),
            None,
        ),
    }

    failures = []
    requested_heads = (64, 128) if args.heads == "all" else (int(args.heads),)
    for heads in requested_heads:
        cases = h64_cases if heads == 64 else h128_cases
        selected = (
            cases.items()
            if args.case == "all"
            else ((args.case, cases[args.case]),)
        )
        for name, (param, mutate) in selected:
            runner = run_h64_case if heads == 64 else run_h128_case
            if not runner(name, param, mutate, args.launch_only):
                failures.append(f"h{heads}.{name}")
    if failures:
        raise SystemExit(f"v8 correctness failed: {', '.join(failures)}")
    print("All selected v8 correctness cases passed")

    if args.audit_launches:
        run_launch_audit(args)


if __name__ == "__main__":
    main()
