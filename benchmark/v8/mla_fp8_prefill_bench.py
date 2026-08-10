#!/usr/bin/env python3
"""Standalone benchmark for the SM100 V3.2 native-FP8 sparse-prefill v8 path.

The benchmark models SGLang's DP=1 attention hierarchy:

    total TP = attention TP * attention CP
    local query heads = global query heads / attention TP

Rank-local query heads are zero-padded into the H64 or H128 execution bucket.
The v8 bridge launches the H64 kernel once for H64 and twice for H128.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
from typing import Callable, Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER = "benchmark/v8/v32_kernelv8_fp8_prefill_fwd_adapter.py:make_runner"

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_DIM_V32 = KV_LORA_RANK + QK_ROPE_HEAD_DIM
HEAD_DIM_V = 512
PAGE_SIZE = 64
SOFTMAX_SCALE_V32 = (128 + QK_ROPE_HEAD_DIM) ** -0.5
FP8_DTYPE = torch.float8_e4m3fn


class InvalidParallelConfig(ValueError):
    """A TP/CP/head tuple that cannot represent an attention rank."""


@dataclass(frozen=True)
class PlanSkip:
    kind: str
    global_heads: int
    tp_size: int
    cp_size: int
    reason: str
    cp_rank: Optional[int] = None
    new_tokens: Optional[int] = None

    def describe(self) -> str:
        fields = [
            f"Hglobal={self.global_heads}",
            f"TP={self.tp_size}",
            f"CP={self.cp_size}",
        ]
        if self.cp_rank is not None:
            fields.append(f"rank={self.cp_rank}")
        if self.new_tokens is not None:
            fields.append(f"new={self.new_tokens}")
        return f"[{self.kind}] " + " ".join(fields) + f": {self.reason}"


@dataclass(frozen=True)
class Case:
    name: str
    layout: str
    batch: int
    new_tokens: int
    cached_len: int
    cp_size: int
    heads: int
    topk: int
    seed: int = 0
    global_heads: int = 128
    tp_size: int = 0
    cp_rank: int = 0
    heads_local: Optional[int] = None

    def __post_init__(self) -> None:
        if self.layout != "v32":
            raise ValueError(f"v8 supports only layout=v32, got {self.layout}")
        if self.batch <= 0 or self.new_tokens <= 0:
            raise ValueError("batch and new_tokens must be positive")
        if self.cached_len < 0:
            raise ValueError("cached_len must be non-negative")
        if self.cp_size <= 0:
            raise ValueError("cp_size must be positive")
        if self.cp_rank < 0 or self.cp_rank >= self.cp_size:
            raise ValueError(
                f"cp_rank must be in [0, {self.cp_size}), got {self.cp_rank}"
            )
        if self.q_per_rank <= 0:
            raise ValueError(
                f"CP rank {self.cp_rank} receives no query tokens for "
                f"new_tokens={self.new_tokens}, cp_size={self.cp_size}"
            )
        if self.global_heads <= 0:
            raise ValueError("global_heads must be positive")
        if self.heads not in {64, 128}:
            raise ValueError(f"execution heads must be H64/H128, got H{self.heads}")
        if self.logical_heads <= 0 or self.logical_heads > self.heads:
            raise ValueError(
                f"local heads H{self.logical_heads} do not fit H{self.heads}"
            )
        if self.tp_size:
            if self.tp_size % self.cp_size != 0:
                raise ValueError(
                    f"total TP {self.tp_size} must be divisible by CP {self.cp_size}"
                )
            if self.global_heads % self.attention_tp_size != 0:
                raise ValueError(
                    f"global heads H{self.global_heads} must be divisible by "
                    f"attention TP {self.attention_tp_size}"
                )
            expected = self.global_heads // self.attention_tp_size
            if expected != self.logical_heads:
                raise ValueError(
                    f"parallel config implies Hlocal={expected}, got {self.logical_heads}"
                )

    @property
    def logical_heads(self) -> int:
        return self.heads if self.heads_local is None else self.heads_local

    @property
    def attention_tp_size(self) -> int:
        if self.tp_size:
            return self.tp_size // self.cp_size
        if self.global_heads % self.logical_heads == 0:
            return self.global_heads // self.logical_heads
        return 0

    @property
    def q_per_rank(self) -> int:
        quotient, remainder = divmod(self.new_tokens, self.cp_size)
        return quotient + int(self.cp_rank < remainder)

    @property
    def cp_remainder(self) -> int:
        return self.new_tokens % self.cp_size

    @property
    def kv_len(self) -> int:
        return self.cached_len + self.new_tokens

    @property
    def rows(self) -> int:
        return self.batch * self.q_per_rank

    @property
    def sm_scale(self) -> float:
        return SOFTMAX_SCALE_V32

    @property
    def effective_topk(self) -> int:
        topk = min(self.topk, self.kv_len)
        if topk < 64:
            return topk
        return max(64, topk // 64 * 64)


@dataclass
class BenchTensors:
    q_bf16_flat: torch.Tensor
    kv_flashmla_fp8: torch.Tensor
    indices_flat: torch.Tensor


@dataclass
class TimeResult:
    backend: str
    ok: bool
    ms: float = float("nan")
    note: str = ""


def execution_heads_for_local(local_heads: int) -> int:
    if local_heads <= 0:
        raise ValueError(f"local heads must be positive, got {local_heads}")
    if local_heads <= 64:
        return 64
    if local_heads <= 128:
        return 128
    raise ValueError(
        "v8 supports rank-local heads up to H128, "
        f"got H{local_heads}"
    )


def derive_parallel_shape(
    tp_size: int, cp_size: int, global_heads: int
) -> tuple[int, int, int]:
    if tp_size <= 0:
        raise InvalidParallelConfig(f"total TP must be positive, got {tp_size}")
    if cp_size <= 0:
        raise InvalidParallelConfig(f"CP must be positive, got {cp_size}")
    if global_heads <= 0:
        raise InvalidParallelConfig(
            f"global heads must be positive, got H{global_heads}"
        )
    if tp_size % cp_size != 0:
        raise InvalidParallelConfig(
            f"invalid TP/CP pair: total TP {tp_size} is not divisible by "
            f"CP {cp_size}; SGLang requires TP % CP == 0"
        )
    attention_tp = tp_size // cp_size
    if global_heads % attention_tp != 0:
        raise InvalidParallelConfig(
            f"global heads H{global_heads} is not divisible by attention "
            f"TP={attention_tp} (total TP={tp_size}, CP={cp_size})"
        )
    local_heads = global_heads // attention_tp
    try:
        execution_heads = execution_heads_for_local(local_heads)
    except ValueError as exc:
        raise InvalidParallelConfig(str(exc)) from exc
    return attention_tp, local_heads, execution_heads


def make_parallel_case(
    name: str,
    layout: str,
    batch: int,
    new_tokens: int,
    cached_len: int,
    tp_size: int,
    cp_size: int,
    global_heads: int,
    topk: int,
    seed: int = 0,
    cp_rank: int = 0,
) -> Case:
    _, local_heads, execution_heads = derive_parallel_shape(
        tp_size, cp_size, global_heads
    )
    return Case(
        name=name,
        layout=layout,
        batch=batch,
        new_tokens=new_tokens,
        cached_len=cached_len,
        cp_size=cp_size,
        heads=execution_heads,
        topk=topk,
        seed=seed,
        global_heads=global_heads,
        tp_size=tp_size,
        cp_rank=cp_rank,
        heads_local=local_heads,
    )


def build_cases(args) -> list[Case]:
    cases: list[Case] = []
    seen: set[tuple] = set()
    skip_map: OrderedDict[tuple, PlanSkip] = OrderedDict()
    valid_parallel_configs: set[tuple[int, int, int]] = set()

    def record_skip(skip: PlanSkip) -> None:
        key = (
            skip.kind,
            skip.global_heads,
            skip.tp_size,
            skip.cp_size,
            skip.cp_rank,
            skip.new_tokens,
        )
        skip_map.setdefault(key, skip)

    def emit(
        name: str,
        layout: str,
        batch: int,
        new: int,
        cached: int,
        cp: int,
        topk: int,
    ) -> None:
        for global_heads in args.global_heads:
            if args.heads is None:
                tp_values = args.tp or [8]
            else:
                tp_values = []
                for local_heads in args.heads:
                    if global_heads % local_heads != 0:
                        raise ValueError(
                            f"legacy --heads H{local_heads} does not divide "
                            f"--global-heads H{global_heads}"
                        )
                    tp_values.append(cp * global_heads // local_heads)
            for tp_size in tp_values:
                try:
                    derive_parallel_shape(tp_size, cp, global_heads)
                except InvalidParallelConfig as exc:
                    record_skip(
                        PlanSkip(
                            "tp_cp", global_heads, tp_size, cp, str(exc)
                        )
                    )
                    continue
                valid_parallel_configs.add((global_heads, tp_size, cp))
                for cp_rank in args.cp_rank:
                    if cp_rank < 0 or cp_rank >= cp:
                        record_skip(
                            PlanSkip(
                                "cp_rank",
                                global_heads,
                                tp_size,
                                cp,
                                f"cp_rank must be in [0, {cp}), got {cp_rank}",
                                cp_rank,
                                new,
                            )
                        )
                        continue
                    quotient, remainder = divmod(new, cp)
                    q_per_rank = quotient + int(cp_rank < remainder)
                    if q_per_rank == 0:
                        record_skip(
                            PlanSkip(
                                "empty_rank",
                                global_heads,
                                tp_size,
                                cp,
                                "this CP rank receives zero query tokens",
                                cp_rank,
                                new,
                            )
                        )
                        continue
                    case = make_parallel_case(
                        name,
                        layout,
                        batch,
                        new,
                        cached,
                        tp_size,
                        cp,
                        global_heads,
                        topk,
                        seed=args.seed,
                        cp_rank=cp_rank,
                    )
                    key = (
                        case.layout,
                        case.batch,
                        case.new_tokens,
                        case.cached_len,
                        case.tp_size,
                        case.cp_size,
                        case.cp_rank,
                        case.global_heads,
                        case.logical_heads,
                        case.topk,
                        case.name,
                    )
                    if case.kv_len >= 64 and key not in seen:
                        seen.add(key)
                        cases.append(case)

    for layout in args.layouts:
        for batch in args.batch:
            for new in args.new:
                for cached in args.cached:
                    for cp in args.cp:
                        for topk in args.topk:
                            emit(args.suite, layout, batch, new, cached, cp, topk)

    skips = tuple(skip_map.values())
    args.skipped_configs = skips
    args.parallel_skips = tuple(skip for skip in skips if skip.kind == "tp_cp")
    args.valid_parallel_configs = tuple(sorted(valid_parallel_configs))
    if not cases:
        detail = "; ".join(skip.describe() for skip in skips)
        suffix = f": {detail}" if detail else ""
        raise ValueError(
            "no runnable benchmark cases remain after filtering invalid "
            f"matrix configurations{suffix}"
        )
    return cases[: args.limit or None]


def quantize_k_cache_v32(input_k_cache: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, h_k, d = input_k_cache.shape
    if h_k != 1 or d != HEAD_DIM_V32:
        raise ValueError(f"expected KV [blocks, {PAGE_SIZE}, 1, {HEAD_DIM_V32}]")
    x = input_k_cache.squeeze(2)
    out = torch.empty(
        (num_blocks, block_size, 656), dtype=FP8_DTYPE, device=x.device
    )
    nope = out[..., :KV_LORA_RANK]
    scales = out[..., KV_LORA_RANK:528].view(torch.float32)
    rope = out[..., 528:].view(torch.bfloat16)
    rope.copy_(x[..., KV_LORA_RANK:])
    for tile in range(4):
        sl = slice(tile * 128, (tile + 1) * 128)
        inv = (x[..., sl].abs().amax(dim=-1).float() / 448.0).clamp_min(1e-12)
        scales[..., tile] = inv
        nope[..., sl] = (x[..., sl].float() / inv.unsqueeze(-1)).to(FP8_DTYPE)
    return out.view(num_blocks, block_size, 1, 656)


def build_tensors(case: Case) -> BenchTensors:
    torch.manual_seed(case.seed)
    num_blocks = (case.kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    q = torch.zeros(
        case.rows,
        case.heads,
        HEAD_DIM_V32,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q[:, : case.logical_heads].copy_(
        (
            torch.randn(
                case.rows,
                case.logical_heads,
                HEAD_DIM_V32,
                dtype=torch.bfloat16,
                device="cuda",
            )
            / 10
        ).clamp_(-1, 1)
    )
    kv = (
        torch.randn(
            num_blocks,
            PAGE_SIZE,
            1,
            HEAD_DIM_V32,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10
    ).clamp_(-1, 1)
    if case.effective_topk < 64 or case.effective_topk % 64:
        raise ValueError(
            f"v8 requires effective topk to be a positive multiple of 64, "
            f"got {case.effective_topk}"
        )
    indices = torch.randint(
        0,
        case.kv_len,
        (case.rows, 1, case.effective_topk),
        dtype=torch.int32,
        device="cuda",
    )
    return BenchTensors(q, quantize_k_cache_v32(kv), indices)


def repo_path(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def load_native_factory(spec: str):
    module_ref, sep, func_name = spec.partition(":")
    if not sep:
        raise ValueError("adapter must look like module:function or path.py:function")
    path = repo_path(module_ref)
    if path.exists():
        module_name = f"flashmla_v8_bench_{abs(hash(str(path)))}"
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load adapter from {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    return getattr(module, func_name)


def make_runner(case: Case, tensors: BenchTensors, adapter: str):
    factory = load_native_factory(adapter)
    constants = {
        "HEAD_DIM_V32": HEAD_DIM_V32,
        "HEAD_DIM_V": HEAD_DIM_V,
        "PAGE_SIZE": PAGE_SIZE,
        "FP8_DTYPE": FP8_DTYPE,
    }
    runner = factory(case=case, tensors=tensors, constants=constants)
    if not callable(runner):
        raise TypeError("v8 adapter did not return a callable")
    return runner


def short_err(exc: Exception) -> str:
    lines = str(exc).strip().splitlines()
    return (lines[0] if lines else type(exc).__name__)[:240]


def time_kernel(
    backend: str,
    fn: Callable[[], torch.Tensor],
    iters: int,
    warmup: int,
    flush_mb: int,
) -> TimeResult:
    try:
        use_nvtx = os.getenv("MLA_PREFILL_BENCH_NVTX", "").lower() in {
            "1", "true", "yes", "on"
        }
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        flush_buf = torch.empty(
            flush_mb * 1024 * 1024, dtype=torch.int8, device="cuda"
        )
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for index in range(iters):
            flush_buf.zero_()
            if use_nvtx:
                torch.cuda.nvtx.range_push("bench::native_kernelv8::measure")
            starts[index].record()
            fn()
            ends[index].record()
            if use_nvtx:
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        samples = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
        return TimeResult(backend, True, samples[len(samples) // 2])
    except Exception as exc:  # noqa: BLE001
        return TimeResult(backend, False, note=short_err(exc))


def result_row(
    case: Case,
    result: TimeResult,
    backend_heads_actual: Optional[int] = None,
) -> dict[str, str]:
    return {
        "case": case.name,
        "layout": case.layout,
        "backend": result.backend,
        "ok": "1" if result.ok else "0",
        "ms": f"{result.ms:.6f}" if result.ok else "",
        "note": result.note,
        "batch": str(case.batch),
        "new_tokens": str(case.new_tokens),
        "cached_len": str(case.cached_len),
        "kv_len": str(case.kv_len),
        "tp_size": str(case.tp_size),
        "attention_tp_size": str(case.attention_tp_size),
        "cp_size": str(case.cp_size),
        "cp_rank": str(case.cp_rank),
        "cp_remainder": str(case.cp_remainder),
        "q_per_rank": str(case.q_per_rank),
        "rows": str(case.rows),
        "extend_tokens_gpu": str(case.rows),
        "global_heads": str(case.global_heads),
        "local_heads": str(case.logical_heads),
        "heads": str(case.logical_heads),
        "heads_actual": str(case.heads),
        "execution_heads": str(case.heads),
        "backend_heads_actual": str(backend_heads_actual or case.heads),
        "heads_padded": str(case.heads),
        "topk_requested": str(case.topk),
        "topk_effective": str(case.effective_topk),
        "seed": str(case.seed),
    }


def print_result_matrix(
    rows: list[dict[str, str]],
    backends: set[str],
    flashmla_bf16_heads: Optional[int] = None,
) -> None:
    del backends, flashmla_bf16_heads
    if not rows:
        return
    headers = (
        "case layout BS rows seqlen TP attnTP CP rank top_k "
        "Hglobal Hlocal Hexec median_ms"
    )
    print(headers)
    for row in rows:
        ms = f"{float(row['ms']):.4f}" if row["ok"] == "1" else "n/a"
        print(
            f"{row['case']} {row['layout']} {row['batch']} {row['rows']} "
            f"{row['kv_len']} {row['tp_size']} {row['attention_tp_size']} "
            f"{row['cp_size']} {row['cp_rank']} {row['topk_effective']} "
            f"{row['global_heads']} {row['local_heads']} "
            f"{row['execution_heads']} {ms}"
        )


def write_csv(path: str, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] wrote {len(rows)} rows -> {output}")


def print_plan(cases: list[Case], args) -> None:
    for skip in args.skipped_configs:
        print(f"[skip] {skip.describe()}")
    print("case layout BS new cached TP attnTP CP rank Hglobal Hlocal Hexec rows topk")
    for case in cases:
        print(
            f"{case.name} {case.layout} {case.batch} {case.new_tokens} "
            f"{case.cached_len} {case.tp_size} {case.attention_tp_size} "
            f"{case.cp_size} {case.cp_rank} {case.global_heads} "
            f"{case.logical_heads} {case.heads} {case.rows} {case.effective_topk}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--suite", choices=("smoke", "cp", "wide", "custom"), default="custom"
    )
    parser.add_argument("--layouts", nargs="+", choices=("v32",), default=["v32"])
    parser.add_argument("--test", nargs="+", choices=("v8",), default=["v8"])
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--new", nargs="+", type=int, default=[875])
    parser.add_argument("--cached", nargs="+", type=int, default=[69125])
    parser.add_argument("--global-heads", nargs="+", type=int, default=[128])
    parallel = parser.add_mutually_exclusive_group()
    parallel.add_argument("--tp", "--tp-size", dest="tp", nargs="+", type=int)
    parallel.add_argument(
        "--heads",
        nargs="+",
        type=int,
        help="legacy local-head alias; prefer --global-heads with --tp",
    )
    parser.add_argument("--cp", nargs="+", type=int, default=[1])
    parser.add_argument("--cp-rank", nargs="+", type=int, default=[0])
    parser.add_argument("--topk", nargs="+", type=int, default=[2048])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--flush-mb", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--csv", default="benchmark/v8/results.csv", help="empty string disables CSV"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--verbose-cases", action="store_true")
    parser.add_argument(
        "--native-kernelv8-module", default=DEFAULT_ADAPTER, help="v8 adapter"
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="accepted for compatibility; this v8-only tool never installs packages",
    )
    return parser


def parse_args(argv: Optional[list[str]] = None):
    return build_arg_parser().parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        cases = build_cases(args)
    except ValueError as exc:
        raise SystemExit(f"invalid benchmark configuration: {exc}") from exc

    print(f"v8 adapter: {args.native_kernelv8_module}")
    print(f"planned cases: {len(cases)}")
    print_plan(cases, args)
    if args.dry_run:
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run on an SM100 B200/B300 host")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise SystemExit(f"native v8 requires SM100f, got SM {major}.{minor}")

    all_rows: list[dict[str, str]] = []
    for index, case in enumerate(cases, 1):
        if not args.quiet_progress:
            print(
                f"[{index}/{len(cases)}] B={case.batch} rows={case.rows} "
                f"TP={case.tp_size} CP={case.cp_size}/rank{case.cp_rank} "
                f"H={case.global_heads}->{case.logical_heads}->{case.heads} "
                f"topk={case.effective_topk}",
                flush=True,
            )
        try:
            tensors = build_tensors(case)
            runner = make_runner(case, tensors, args.native_kernelv8_module)
            result = time_kernel(
                "native_kernelv8", runner, args.iters, args.warmup, args.flush_mb
            )
        except Exception as exc:  # noqa: BLE001
            result = TimeResult("native_kernelv8", False, note=short_err(exc))
        row = result_row(case, result, case.heads)
        all_rows.append(row)
        if not args.quiet_progress:
            outcome = f"{result.ms:.4f} ms" if result.ok else f"FAILED: {result.note}"
            print(f"  v8={outcome}")

    print_result_matrix(all_rows, {"native_kernelv8"})
    if args.csv:
        write_csv(args.csv, all_rows)
    if any(row["ok"] != "1" for row in all_rows):
        raise SystemExit("one or more v8 benchmark cases failed")


if __name__ == "__main__":
    main()
