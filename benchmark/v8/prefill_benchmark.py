#!/usr/bin/env python3
"""Standalone SM100 benchmark for DeepSeek V3.2 FP8 sparse prefill.

The benchmark models SGLang's DP=1 attention hierarchy:

    total TP = attention TP * attention CP
    local query heads = global query heads / attention TP

With one visible GPU, no arguments measure H64 and H128 over batch sizes
1, 2, 4, 8, 16, and 32. With two or more visible GPUs, the launcher uses every
GPU and measures physical pure-TP and DSA-CP endpoints with real NCCL traffic.
The default comparison is FlashMLA's FP8 sparse decode path versus the native
FP8 prefill kernel.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
from dataclasses import dataclass
from datetime import timedelta
import importlib
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER = "benchmark/v8/v32_kernelv8_fp8_prefill_fwd_adapter.py:make_runner"
DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32]
DEFAULT_BACKENDS = ("decode", "v8")
TP_SWEEP = [1, 2, 4, 8]
CP_SWEEP = [1, 2, 4, 8]

BACKEND_LABELS = {
    "decode": "decode",
    "v8": "v8",
    "bf16": "bf16",
    "fp8_dequant": "fp8-dequant",
    "trtllm": "trtllm-gen",
    "decode_chunked": "decode-chunked",
}

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_DIM_V32 = KV_LORA_RANK + QK_ROPE_HEAD_DIM
HEAD_DIM_V = 512
HIDDEN_SIZE = 7168
LATENT_KV_ELEMENTS = HEAD_DIM_V32
PAGE_SIZE = 64
SOFTMAX_SCALE_V32 = (128 + QK_ROPE_HEAD_DIM) ** -0.5
FP8_DTYPE = torch.float8_e4m3fn
TORCHRUN_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)
DISTRIBUTED_TIMEOUT = timedelta(seconds=120)
DEFAULT_LAUNCHER_WATCHDOG_SECONDS = 2 * 60 * 60


class InvalidParallelConfig(ValueError):
    """A TP/CP/head tuple that cannot represent an attention rank."""


@dataclass(frozen=True)
class DistributedContext:
    """Process identity for a physical multi-GPU benchmark job."""

    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    control_group: Any = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class RankTopology:
    """SGLang-compatible `(CP, attention-TP)` rank coordinates."""

    world_size: int
    total_tp: int
    cp_size: int
    global_rank: int
    attention_tp_size: int
    attention_tp_rank: int
    cp_rank: int
    attention_tp_ranks: tuple[int, ...]
    cp_ranks: tuple[int, ...]
    head_start: int
    head_stop: int


@dataclass
class RankGroups:
    """Process groups and metadata used by one TP/CP factorization."""

    topology: RankTopology
    attention_tp_group: Any = None
    cp_group: Any = None


@dataclass
class CommunicationBuffers:
    """Production-shaped NCCL payloads for one benchmark case."""

    groups: RankGroups
    cp_local_kv: Optional[torch.Tensor] = None
    cp_gathered_kv: Optional[torch.Tensor] = None
    cp_reordered_kv: Optional[torch.Tensor] = None
    tp_projected_output: Optional[torch.Tensor] = None

    @property
    def has_cp(self) -> bool:
        return self.groups.topology.cp_size > 1

    @property
    def has_tp(self) -> bool:
        return self.groups.topology.attention_tp_size > 1

    @property
    def cp_payload_bytes(self) -> int:
        return 0 if self.cp_local_kv is None else self.cp_local_kv.numel() * 2

    @property
    def tp_payload_bytes(self) -> int:
        if self.tp_projected_output is None:
            return 0
        return self.tp_projected_output.numel() * 2

    def run_cp(self) -> None:
        if not self.has_cp:
            return
        assert self.cp_local_kv is not None
        assert self.cp_gathered_kv is not None
        dist.all_gather_into_tensor(
            self.cp_gathered_kv,
            self.cp_local_kv,
            group=self.groups.cp_group,
        )
        self.cp_reordered_kv = rerange_cp_rank_major(
            self.cp_gathered_kv, self.groups.topology.cp_size
        )

    def run_tp(self) -> None:
        if not self.has_tp:
            return
        assert self.tp_projected_output is not None
        dist.all_reduce(
            self.tp_projected_output,
            op=dist.ReduceOp.SUM,
            group=self.groups.attention_tp_group,
        )

    def run_cp_envelope(self, kernel: Callable[[], torch.Tensor]) -> None:
        self.run_cp()
        kernel()


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
    physical: bool = False
    global_rank: int = 0
    world_size: int = 1
    attention_tp_rank: int = 0

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
        if (self.logical_rows if self.physical else self.q_per_rank) <= 0:
            raise ValueError(
                f"CP rank {self.cp_rank} receives no query tokens for "
                f"batch={self.batch}, new_tokens={self.new_tokens}, "
                f"cp_size={self.cp_size}"
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
        if self.physical:
            topology = derive_rank_topology(
                world_size=self.world_size,
                total_tp=self.tp_size,
                cp_size=self.cp_size,
                global_heads=self.global_heads,
                global_rank=self.global_rank,
            )
            if topology.cp_rank != self.cp_rank:
                raise ValueError(
                    f"rank {self.global_rank} implies CP rank {topology.cp_rank}, "
                    f"got {self.cp_rank}"
                )
            if topology.attention_tp_rank != self.attention_tp_rank:
                raise ValueError(
                    f"rank {self.global_rank} implies attention-TP rank "
                    f"{topology.attention_tp_rank}, got {self.attention_tp_rank}"
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
    def logical_rows(self) -> int:
        if not self.physical:
            return self.batch * self.q_per_rank
        quotient, remainder = divmod(self.batch * self.new_tokens, self.cp_size)
        return quotient + int(self.cp_rank < remainder)

    @property
    def logical_rows_min(self) -> int:
        if not self.physical:
            return self.logical_rows
        return self.batch * self.new_tokens // self.cp_size

    @property
    def logical_rows_max(self) -> int:
        if not self.physical:
            return self.logical_rows
        return ceil_align(self.batch * self.new_tokens, self.cp_size) // self.cp_size

    @property
    def padded_total_rows(self) -> int:
        logical = self.batch * self.new_tokens
        if not self.physical:
            return logical
        logical = ceil_align(logical, self.attention_tp_size)
        return ceil_align(logical, 2 * self.cp_size)

    @property
    def padding_rows(self) -> int:
        return self.rows - self.logical_rows

    @property
    def padding_rows_min(self) -> int:
        return self.rows - self.logical_rows_max

    @property
    def padding_rows_max(self) -> int:
        return self.rows - self.logical_rows_min

    @property
    def cp_remainder(self) -> int:
        tokens = self.batch * self.new_tokens if self.physical else self.new_tokens
        return tokens % self.cp_size

    @property
    def kv_len(self) -> int:
        return self.cached_len + self.new_tokens

    @property
    def rows(self) -> int:
        if self.physical:
            return self.padded_total_rows // self.cp_size
        return self.logical_rows

    @property
    def topology_kind(self) -> str:
        if not self.physical:
            return "simulated-rank-local"
        if self.cp_size == 1:
            return "production-pure-tp"
        if self.cp_size == self.tp_size:
            return (
                "sglang-dsa-cp-endpoint"
                if self.tp_size <= 8
                else "experimental-dsa-cp-over-8"
            )
        return "experimental-tp-cp"

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
    kv_bf16_flat: torch.Tensor
    kv_flashmla_fp8: torch.Tensor
    indices_flat: torch.Tensor
    q_fp8_batched: Optional[torch.Tensor] = None
    kv_trtllm_fp8: Optional[torch.Tensor] = None
    indices_batched: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None
    dequant_workspace: Optional[torch.Tensor] = None


@dataclass
class TimeResult:
    backend: str
    ok: bool
    ms: float = float("nan")
    note: str = ""
    kernel_ms: float = float("nan")
    cp_comm_ms: float = 0.0
    tp_comm_ms: float = 0.0
    envelope_ms: float = float("nan")


def ceil_align(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return (value + alignment - 1) // alignment * alignment


def divisors(value: int) -> list[int]:
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    return [candidate for candidate in range(1, value + 1) if value % candidate == 0]


def critical_path_median(samples_by_rank: Any) -> float:
    """Compute p50 of the synchronous per-iteration rank maximum."""

    samples = torch.as_tensor(samples_by_rank, dtype=torch.float64, device="cpu")
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("latency samples must have shape [ranks, iterations]")
    if not torch.isfinite(samples).all():
        raise ValueError("latency samples must all be finite")
    critical = samples.max(dim=0).values.sort().values
    return float(critical[len(critical) // 2].item())


def rerange_cp_rank_major(
    gathered: torch.Tensor, cp_size: int
) -> torch.Tensor:
    """Convert NCCL rank-major rows to SGLang's token-major CP order."""

    if gathered.ndim < 1 or cp_size <= 0 or gathered.shape[0] % cp_size:
        raise ValueError(
            "gathered CP tensor must have a leading dimension divisible by CP"
        )
    rows = gathered.shape[0] // cp_size
    tail = tuple(gathered.shape[1:])
    return (
        gathered.view(cp_size, rows, *tail)
        .transpose(0, 1)
        .reshape(rows * cp_size, *tail)
    )


def derive_rank_topology(
    world_size: int,
    total_tp: int,
    cp_size: int,
    global_heads: int,
    global_rank: int,
) -> RankTopology:
    """Map a global rank to SGLang's fastest-changing attention-TP axis."""

    if world_size != total_tp:
        raise InvalidParallelConfig(
            "physical mode requires WORLD_SIZE == total TP; "
            f"got WORLD_SIZE={world_size}, TP={total_tp}"
        )
    if global_rank < 0 or global_rank >= world_size:
        raise InvalidParallelConfig(
            f"global rank must be in [0, {world_size}), got {global_rank}"
        )
    attention_tp, local_heads, _ = derive_parallel_shape(
        total_tp, cp_size, global_heads
    )
    attention_tp_rank = global_rank % attention_tp
    cp_rank = global_rank // attention_tp
    attention_tp_ranks = tuple(
        cp_rank * attention_tp + rank for rank in range(attention_tp)
    )
    cp_ranks = tuple(
        attention_tp_rank + rank * attention_tp for rank in range(cp_size)
    )
    head_start = attention_tp_rank * local_heads
    return RankTopology(
        world_size=world_size,
        total_tp=total_tp,
        cp_size=cp_size,
        global_rank=global_rank,
        attention_tp_size=attention_tp,
        attention_tp_rank=attention_tp_rank,
        cp_rank=cp_rank,
        attention_tp_ranks=attention_tp_ranks,
        cp_ranks=cp_ranks,
        head_start=head_start,
        head_stop=head_start + local_heads,
    )


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
    physical: bool = False,
    global_rank: int = 0,
    world_size: int = 1,
    attention_tp_rank: int = 0,
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
        physical=physical,
        global_rank=global_rank,
        world_size=world_size,
        attention_tp_rank=attention_tp_rank,
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
                    if args.physical_mode:
                        derive_rank_topology(
                            args.distributed_world_size,
                            tp_size,
                            cp,
                            global_heads,
                            args.distributed_rank,
                        )
                except InvalidParallelConfig as exc:
                    record_skip(
                        PlanSkip(
                            "tp_cp", global_heads, tp_size, cp, str(exc)
                        )
                    )
                    continue
                if args.physical_mode and batch * new < cp:
                    record_skip(
                        PlanSkip(
                            "empty_rank",
                            global_heads,
                            tp_size,
                            cp,
                            "physical CP requires total query tokens >= CP on every rank",
                            None,
                            new,
                        )
                    )
                    continue
                valid_parallel_configs.add((global_heads, tp_size, cp))
                topology = (
                    derive_rank_topology(
                        args.distributed_world_size,
                        tp_size,
                        cp,
                        global_heads,
                        args.distributed_rank,
                    )
                    if args.physical_mode
                    else None
                )
                cp_ranks = [topology.cp_rank] if topology else args.cp_rank
                for cp_rank in cp_ranks:
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
                    logical_tokens = batch * new if args.physical_mode else new
                    quotient, remainder = divmod(logical_tokens, cp)
                    local_logical_rows = quotient + int(cp_rank < remainder)
                    if local_logical_rows == 0:
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
                        physical=args.physical_mode,
                        global_rank=args.distributed_rank,
                        world_size=args.distributed_world_size,
                        attention_tp_rank=(
                            topology.attention_tp_rank if topology else 0
                        ),
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


def build_tensors(case: Case, backends: tuple[str, ...] = DEFAULT_BACKENDS) -> BenchTensors:
    if case.physical:
        q_generator = torch.Generator(device="cuda").manual_seed(
            case.seed
            + 10_000 * case.cp_rank
            + 100 * case.attention_tp_rank
            + 1
        )
        kv_generator = torch.Generator(device="cuda").manual_seed(case.seed + 2)
        indices_generator = torch.Generator(device="cuda").manual_seed(
            case.seed + 10_000 * case.cp_rank + 3
        )
    else:
        # Preserve the historical one-GPU tensor stream exactly.
        torch.manual_seed(case.seed)
        q_generator = kv_generator = indices_generator = None
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
                generator=q_generator,
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
            generator=kv_generator,
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
        generator=indices_generator,
    )
    if case.padding_rows:
        q[case.logical_rows :].zero_()
    runner_batch = 1 if case.physical else case.batch
    runner_q = case.rows if case.physical else case.q_per_rank
    q_batched = q.view(runner_batch, runner_q, case.heads, HEAD_DIM_V32)
    indices_batched = indices.view(
        runner_batch, runner_q, case.effective_topk
    )
    need_trtllm = "trtllm" in backends
    need_dequant = "fp8_dequant" in backends
    return BenchTensors(
        q_bf16_flat=q,
        kv_bf16_flat=kv.view(num_blocks * PAGE_SIZE, 1, HEAD_DIM_V32),
        kv_flashmla_fp8=quantize_k_cache_v32(kv),
        indices_flat=indices,
        q_fp8_batched=q_batched.to(FP8_DTYPE) if need_trtllm else None,
        kv_trtllm_fp8=(
            kv.permute(0, 2, 1, 3).contiguous().to(FP8_DTYPE)
            if need_trtllm
            else None
        ),
        indices_batched=indices_batched if need_trtllm else None,
        seq_lens=(
            torch.full(
                (runner_batch,), case.kv_len, dtype=torch.int32, device="cuda"
            )
            if need_trtllm
            else None
        ),
        dequant_workspace=(
            torch.empty_like(kv) if need_dequant else None
        ),
    )


def create_rank_groups(
    cases: list[Case], context: DistributedContext
) -> dict[tuple[int, int, int], RankGroups]:
    """Create every process group in one deterministic global order."""

    if not context.enabled or context.world_size == 1:
        return {}
    configs = sorted(
        {(case.tp_size, case.cp_size, case.global_heads) for case in cases}
    )
    result: dict[tuple[int, int, int], RankGroups] = {}
    for total_tp, cp_size, global_heads in configs:
        topology = derive_rank_topology(
            context.world_size,
            total_tp,
            cp_size,
            global_heads,
            context.rank,
        )
        attention_tp_size = topology.attention_tp_size
        local_tp_group = None
        if attention_tp_size == context.world_size:
            local_tp_group = dist.group.WORLD
        elif attention_tp_size > 1:
            for cp_rank in range(cp_size):
                ranks = list(
                    range(
                        cp_rank * attention_tp_size,
                        (cp_rank + 1) * attention_tp_size,
                    )
                )
                group = dist.new_group(
                    ranks=ranks, timeout=DISTRIBUTED_TIMEOUT
                )
                if context.rank in ranks:
                    local_tp_group = group

        local_cp_group = None
        if cp_size == context.world_size:
            local_cp_group = dist.group.WORLD
        elif cp_size > 1:
            for attention_tp_rank in range(attention_tp_size):
                ranks = [
                    attention_tp_rank + cp_rank * attention_tp_size
                    for cp_rank in range(cp_size)
                ]
                group = dist.new_group(
                    ranks=ranks, timeout=DISTRIBUTED_TIMEOUT
                )
                if context.rank in ranks:
                    local_cp_group = group

        result[(total_tp, cp_size, global_heads)] = RankGroups(
            topology=topology,
            attention_tp_group=local_tp_group,
            cp_group=local_cp_group,
        )
    return result


def build_communication_buffers(
    case: Case,
    groups: RankGroups,
) -> CommunicationBuffers:
    """Allocate real NCCL payloads outside every timed interval.

    CP mirrors the per-layer BF16 latent-KV all-gather and token-major rerange.
    TP is a communication-only probe at the mathematically valid post-o_proj
    `[rows, hidden_size]` boundary; projection GEMMs are intentionally absent.
    """

    buffers = CommunicationBuffers(groups=groups)
    if groups.topology.cp_size > 1:
        buffers.cp_local_kv = torch.empty(
            (case.rows, LATENT_KV_ELEMENTS),
            dtype=torch.bfloat16,
            device="cuda",
        )
        buffers.cp_local_kv.fill_(float(case.cp_rank + 1))
        if case.padding_rows:
            buffers.cp_local_kv[case.logical_rows :].zero_()
        buffers.cp_gathered_kv = torch.empty(
            (case.rows * case.cp_size, LATENT_KV_ELEMENTS),
            dtype=torch.bfloat16,
            device="cuda",
        )
    if groups.topology.attention_tp_size > 1:
        buffers.tp_projected_output = torch.zeros(
            (case.rows, HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device="cuda",
        )
    return buffers


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


def max_dynamic_smem_bytes() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    value = getattr(props, "shared_memory_per_block_optin", None)
    if value:
        return int(value)
    value = getattr(props, "shared_memory_per_block", None)
    return int(value) if value else None


def decode_metadata_smem_bytes(rows: int) -> int:
    return 4 * (rows * 5 + 1)


def load_flashmla_api():
    from flash_mla import (
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    return flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata


def make_decode_runner(case: Case, tensors: BenchTensors):
    required_smem = decode_metadata_smem_bytes(case.rows)
    max_smem = max_dynamic_smem_bytes()
    if max_smem is not None and required_smem > max_smem:
        raise RuntimeError(
            "decode metadata smem "
            f"{required_smem // 1024} KiB exceeds device limit "
            f"{max_smem // 1024} KiB"
        )

    _, flash_mla_with_kvcache, get_mla_metadata = load_flashmla_api()
    q = tensors.q_bf16_flat.view(
        case.rows, 1, case.heads, HEAD_DIM_V32
    )
    cache_seqlens = torch.full(
        (case.rows,), case.kv_len, dtype=torch.int32, device="cuda"
    )
    block_table = torch.empty((case.rows, 0), dtype=torch.int32, device="cuda")
    sched_meta, num_splits = get_mla_metadata()

    def run():
        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=tensors.kv_flashmla_fp8,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=HEAD_DIM_V,
            tile_scheduler_metadata=sched_meta,
            num_splits=num_splits,
            softmax_scale=case.sm_scale,
            is_fp8_kvcache=True,
            indices=tensors.indices_flat,
        )
        return out

    return run


def make_decode_chunked_runner(case: Case, tensors: BenchTensors):
    max_smem = max_dynamic_smem_bytes()
    if max_smem is None:
        raise RuntimeError("device shared-memory limit is unavailable")
    max_rows = (max_smem // 4 - 1) // 5
    if max_rows <= 0:
        raise RuntimeError("device cannot hold decode scheduler metadata")

    _, flash_mla_with_kvcache, get_mla_metadata = load_flashmla_api()
    chunks = []
    for begin in range(0, case.rows, max_rows):
        end = min(begin + max_rows, case.rows)
        rows = end - begin
        q = tensors.q_bf16_flat[begin:end].view(
            rows, 1, case.heads, HEAD_DIM_V32
        )
        indices = tensors.indices_flat[begin:end]
        cache_seqlens = torch.full(
            (rows,), case.kv_len, dtype=torch.int32, device="cuda"
        )
        block_table = torch.empty((rows, 0), dtype=torch.int32, device="cuda")
        sched_meta, num_splits = get_mla_metadata()
        chunks.append(
            (q, indices, cache_seqlens, block_table, sched_meta, num_splits)
        )

    def run():
        outputs = []
        for q, indices, cache_seqlens, block_table, sched_meta, num_splits in chunks:
            out, _ = flash_mla_with_kvcache(
                q=q,
                k_cache=tensors.kv_flashmla_fp8,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                head_dim_v=HEAD_DIM_V,
                tile_scheduler_metadata=sched_meta,
                num_splits=num_splits,
                softmax_scale=case.sm_scale,
                is_fp8_kvcache=True,
                indices=indices,
            )
            outputs.append(out)
        return tuple(outputs)

    return run


def make_bf16_runner(case: Case, tensors: BenchTensors):
    flash_mla_sparse_fwd, _, _ = load_flashmla_api()

    def run():
        out, _, _ = flash_mla_sparse_fwd(
            tensors.q_bf16_flat,
            tensors.kv_bf16_flat,
            tensors.indices_flat,
            case.sm_scale,
            HEAD_DIM_V,
        )
        return out

    return run


def make_fp8_dequant_runner(case: Case, tensors: BenchTensors):
    flash_mla_sparse_fwd, _, _ = load_flashmla_api()

    def run():
        out, _, _ = flash_mla_sparse_fwd(
            tensors.q_bf16_flat,
            tensors.kv_flashmla_fp8,
            tensors.indices_flat,
            case.sm_scale,
            HEAD_DIM_V,
            dequant_workspace=tensors.dequant_workspace,
        )
        return out

    return run


def make_trtllm_runner(case: Case, tensors: BenchTensors):
    import flashinfer

    if (
        tensors.q_fp8_batched is None
        or tensors.kv_trtllm_fp8 is None
        or tensors.indices_batched is None
        or tensors.seq_lens is None
    ):
        raise RuntimeError("TRTLLM tensors were not prepared")
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")

    def run():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=tensors.q_fp8_batched,
            kv_cache=tensors.kv_trtllm_fp8,
            workspace_buffer=workspace,
            qk_nope_head_dim=128,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=tensors.indices_batched,
            seq_lens=tensors.seq_lens,
            max_seq_len=case.kv_len,
            sparse_mla_top_k=case.effective_topk,
            bmm1_scale=case.sm_scale,
            backend="trtllm-gen",
        )

    return run


def make_backend_runner(
    backend: str, case: Case, tensors: BenchTensors, args
):
    if backend == "v8":
        return make_runner(case, tensors, args.native_kernelv8_module)
    if backend == "decode":
        return make_decode_runner(case, tensors)
    if backend == "decode_chunked":
        return make_decode_chunked_runner(case, tensors)
    if backend == "bf16":
        return make_bf16_runner(case, tensors)
    if backend == "fp8_dequant":
        return make_fp8_dequant_runner(case, tensors)
    if backend == "trtllm":
        return make_trtllm_runner(case, tensors)
    raise ValueError(f"unknown benchmark backend: {backend}")


def short_err(exc: Exception) -> str:
    lines = str(exc).strip().splitlines()
    return (lines[0] if lines else type(exc).__name__)[:240]


def is_fatal_cuda_exception(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cuda error",
            "illegal memory access",
            "device-side assert",
            "misaligned address",
            "launch failure",
            "out of memory",
        )
    )


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
                torch.cuda.nvtx.range_push(f"bench::{backend}::measure")
            starts[index].record()
            fn()
            ends[index].record()
            if use_nvtx:
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        samples = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
        median = samples[len(samples) // 2]
        return TimeResult(
            backend,
            True,
            median,
            kernel_ms=median,
        )
    except Exception as exc:  # noqa: BLE001
        return TimeResult(backend, False, note=short_err(exc))


def time_distributed_operation(
    label: str,
    fn: Callable[[], Any],
    iters: int,
    warmup: int,
    flush_mb: int,
    context: DistributedContext,
) -> float:
    """Return `median(iteration-wise max rank CUDA latency)`.

    Alignment barriers, L2 flushing, and the MAX reduction are instrumentation
    and remain outside the timed CUDA-event interval.
    """

    use_nvtx = os.getenv("MLA_PREFILL_BENCH_NVTX", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not context.enabled or context.world_size <= 1:
        raise ValueError("distributed timing requires a multi-rank context")
    if context.control_group is None:
        raise ValueError("distributed timing requires a Gloo control group")

    flush_buf = None
    local_error = None
    try:
        flush_buf = torch.empty(
            flush_mb * 1024 * 1024, dtype=torch.int8, device="cuda"
        )
    except Exception as exc:  # noqa: BLE001
        local_error = short_err(exc)
    shared_error = sync_setup_error(local_error, context)
    if shared_error:
        raise RuntimeError(f"distributed {label} allocation failed: {shared_error}")
    assert flush_buf is not None

    for warmup_index in range(warmup):
        dist.barrier(group=context.control_group)
        local_error = None
        try:
            fn()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            local_error = short_err(exc)
        shared_error = sync_setup_error(local_error, context)
        if shared_error:
            raise RuntimeError(
                f"distributed {label} warmup {warmup_index} failed: "
                f"{shared_error}"
            )

    local_samples: list[float] = []
    for iteration in range(iters):
        local_error = None
        try:
            flush_buf.zero_()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            local_error = short_err(exc)
        shared_error = sync_setup_error(local_error, context)
        if shared_error:
            raise RuntimeError(
                f"distributed {label} pre-iteration {iteration} failed: "
                f"{shared_error}"
            )

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        local_ms = float("nan")
        local_error = None
        nvtx_active = False
        try:
            if use_nvtx:
                torch.cuda.nvtx.range_push(f"bench::{label}::measure")
                nvtx_active = True
            start.record()
            fn()
            end.record()
            end.synchronize()
            local_ms = start.elapsed_time(end)
        except Exception as exc:  # noqa: BLE001
            local_error = short_err(exc)
        finally:
            if nvtx_active:
                torch.cuda.nvtx.range_pop()
        shared_error = sync_setup_error(local_error, context)
        if shared_error:
            raise RuntimeError(
                f"distributed {label} iteration {iteration} failed: "
                f"{shared_error}"
            )
        local_samples.append(local_ms)

    local_tensor = torch.tensor(local_samples, dtype=torch.float64, device="cpu")
    gathered_samples = [
        torch.empty_like(local_tensor) for _ in range(context.world_size)
    ]
    dist.all_gather(
        gathered_samples, local_tensor, group=context.control_group
    )
    return critical_path_median(torch.stack(gathered_samples))


def sync_setup_error(
    local_error: Optional[str], context: DistributedContext
) -> Optional[str]:
    """Make every rank take the same setup or runtime error path."""

    if not context.enabled or context.world_size == 1:
        return local_error
    errors: list[Optional[str]] = [None] * context.world_size
    dist.all_gather_object(errors, local_error, group=context.control_group)
    for rank, error in enumerate(errors):
        if error:
            return f"rank {rank}: {error}"
    return None


def result_row(
    case: Case,
    result: TimeResult,
    backend_heads_actual: Optional[int] = None,
    communication: Optional[CommunicationBuffers] = None,
) -> dict[str, str]:
    topology = communication.groups.topology if communication else None
    kernel_ms = (
        result.kernel_ms
        if result.kernel_ms == result.kernel_ms
        else result.ms
    )
    return {
        "case": case.name,
        "layout": case.layout,
        "backend": result.backend,
        "ok": "1" if result.ok else "0",
        "ms": f"{result.ms:.6f}" if result.ok else "",
        "kernel_ms": f"{kernel_ms:.6f}" if result.ok else "",
        "cp_latent_allgather_probe_ms": (
            f"{result.cp_comm_ms:.6f}" if result.cp_comm_ms else ""
        ),
        "tp_o_proj_allreduce_probe_ms": (
            f"{result.tp_comm_ms:.6f}" if result.tp_comm_ms else ""
        ),
        "cp_allgather_probe_plus_kernel_ms": (
            f"{result.envelope_ms:.6f}"
            if result.ok and result.envelope_ms == result.envelope_ms
            else ""
        ),
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
        "q_per_rank": str(case.logical_rows if case.physical else case.q_per_rank),
        "logical_rows": str(case.logical_rows),
        "logical_rows_min": str(case.logical_rows_min),
        "logical_rows_max": str(case.logical_rows_max),
        "padded_total_rows": str(case.padded_total_rows),
        "padding_rows": str(case.padding_rows),
        "padding_rows_min": str(case.padding_rows_min),
        "padding_rows_max": str(case.padding_rows_max),
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
        "execution_mode": (
            "physical-nccl" if case.physical else "single-gpu-shape-simulation"
        ),
        "topology_status": case.topology_kind,
        "world_size": str(case.world_size),
        "global_rank": str(case.global_rank),
        "rank_scope": "all-ranks" if case.physical else "selected-rank",
        "attention_tp_rank": str(case.attention_tp_rank),
        "attention_tp_group": (
            ":".join(map(str, topology.attention_tp_ranks)) if topology else ""
        ),
        "cp_group": ":".join(map(str, topology.cp_ranks)) if topology else "",
        "latency_reduction": (
            "median(iteration-wise-rank-max)" if case.physical else "local-median"
        ),
        "cp_payload_bytes_per_rank": (
            str(communication.cp_payload_bytes) if communication else "0"
        ),
        "tp_payload_bytes_per_rank": (
            str(communication.tp_payload_bytes) if communication else "0"
        ),
        "cp_metric_scope": (
            "bf16-latent-kv-allgather-rerange-probe;output-not-packed-into-kernel-cache"
            if communication and communication.has_cp
            else ""
        ),
        "tp_metric_scope": (
            "post-o_proj-shaped-allreduce-probe;projection-gemms-excluded"
            if communication and communication.has_tp
            else ""
        ),
    }


def print_result_matrix(
    rows: list[dict[str, str]],
    backends: set[str],
    flashmla_bf16_heads: Optional[int] = None,
) -> None:
    del flashmla_bf16_heads
    if not rows:
        return
    physical = any(row.get("execution_mode") == "physical-nccl" for row in rows)
    ordered_backends = [name for name in BACKEND_LABELS if name in backends]
    grouped: OrderedDict[tuple[str, ...], dict[str, dict[str, str]]] = OrderedDict()
    for row in rows:
        key = (
            row["case"],
            row["layout"],
            row["batch"],
            row["rows"],
            row["kv_len"],
            row["tp_size"],
            row["attention_tp_size"],
            row["cp_size"],
            (
                "all"
                if row.get("execution_mode") == "physical-nccl"
                else row["cp_rank"]
            ),
            row["global_heads"],
            row["local_heads"],
            row["execution_heads"],
            row["topk_effective"],
            row.get("topology_status", ""),
        )
        grouped.setdefault(key, {})[row["backend"]] = row

    headers = [
        "case",
        "layout",
        "BS",
        "rows",
        "seqlen",
        "TP",
        "attnTP",
        "CP",
        "rank",
        "Hglobal",
        "Hlocal",
        "Hexec",
        "top_k",
        "topology",
    ]
    if physical:
        headers.extend(["CP latent AG probe", "TP o_proj AR probe"])
        for name in ordered_backends:
            headers.extend(
                [
                    f"{BACKEND_LABELS[name]} kernel",
                    f"{BACKEND_LABELS[name]} CPprobe+kernel",
                ]
            )
    else:
        headers.extend(BACKEND_LABELS[name] for name in ordered_backends)
    show_delta = "decode" in ordered_backends and "v8" in ordered_backends
    if show_delta:
        headers.append("v8 vs decode kernel" if physical else "v8 vs decode")

    table_rows: list[list[str]] = []
    for key, backend_rows in grouped.items():
        values = list(key)
        if physical:
            representative = next(
                (
                    row
                    for row in backend_rows.values()
                    if row.get("cp_latent_allgather_probe_ms")
                    or row.get("tp_o_proj_allreduce_probe_ms")
                ),
                next(iter(backend_rows.values())),
            )
            values.extend(
                [
                    (
                        f"{float(representative['cp_latent_allgather_probe_ms']):.4f}"
                        if representative.get("cp_latent_allgather_probe_ms")
                        else "-"
                    ),
                    (
                        f"{float(representative['tp_o_proj_allreduce_probe_ms']):.4f}"
                        if representative.get("tp_o_proj_allreduce_probe_ms")
                        else "-"
                    ),
                ]
            )
            for backend in ordered_backends:
                row = backend_rows.get(backend)
                if row is None or row["ok"] != "1":
                    values.extend(["n/a", "n/a"])
                else:
                    values.extend(
                        [
                            f"{float(row['kernel_ms']):.4f}",
                            (
                                f"{float(row['cp_allgather_probe_plus_kernel_ms']):.4f}"
                                if row.get("cp_allgather_probe_plus_kernel_ms")
                                else "-"
                            ),
                        ]
                    )
        else:
            for backend in ordered_backends:
                row = backend_rows.get(backend)
                values.append(
                    f"{float(row['ms']):.4f}"
                    if row is not None and row["ok"] == "1"
                    else "n/a"
                )
        if show_delta:
            decode_row = backend_rows.get("decode")
            v8_row = backend_rows.get("v8")
            if (
                decode_row is None
                or v8_row is None
                or decode_row["ok"] != "1"
                or v8_row["ok"] != "1"
            ):
                values.append("n/a")
            else:
                metric = "kernel_ms" if physical else "ms"
                decode_ms = float(decode_row[metric])
                v8_ms = float(v8_row[metric])
                faster = (decode_ms - v8_ms) / decode_ms * 100.0
                direction = "faster" if faster >= 0 else "slower"
                values.append(f"{abs(faster):.2f}% {direction}")
        table_rows.append(values)

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in table_rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def write_csv(path: str, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[csv] wrote {len(rows)} rows -> {output}")


def print_plan(cases: list[Case], args) -> None:
    for skip in args.skipped_configs:
        print(f"[skip] {skip.describe()}")
    print(
        "case layout BS new cached TP attnTP CP rank Hglobal Hlocal Hexec "
        "logical_rows logical[min,max] physical_rows pad[min,max] topk topology"
    )
    for case in cases:
        print(
            f"{case.name} {case.layout} {case.batch} {case.new_tokens} "
            f"{case.cached_len} {case.tp_size} {case.attention_tp_size} "
            f"{case.cp_size} {case.cp_rank} {case.global_heads} "
            f"{case.logical_heads} {case.heads} {case.logical_rows} "
            f"[{case.logical_rows_min},{case.logical_rows_max}] {case.rows} "
            f"[{case.padding_rows_min},{case.padding_rows_max}] "
            f"{case.effective_topk} {case.topology_kind}"
        )


def resolve_parallel_presets(args) -> None:
    tp_present = args.tp is not None
    cp_present = args.cp is not None

    if args.physical_mode:
        if args.heads is not None:
            raise ValueError(
                "--heads is a rank-local simulation option; use --tp/--cp "
                "for a physical job"
            )
        if args.cp_rank is not None or args.all_ranks:
            raise ValueError(
                "physical mode derives CP rank from RANK; remove "
                "--cp-rank/--all-ranks"
            )
        world_size = args.distributed_world_size
        if not tp_present and not cp_present:
            # Pure TP and the CP=TP topology forced by current DP=1 DSA CP.
            args.tp = [world_size]
            args.cp = [1, world_size]
            mode = f"physical-{world_size}gpu-sglang-endpoints"
        elif tp_present and not cp_present:
            args.tp = args.tp or [world_size]
            args.cp = [1]
            mode = f"physical-{world_size}gpu-tp"
        elif not tp_present and cp_present:
            args.tp = [world_size]
            # A bare --cp deliberately expands topology what-if divisors too.
            args.cp = args.cp or divisors(world_size)
            mode = f"physical-{world_size}gpu-cp-sweep"
        else:
            args.tp = args.tp or [world_size]
            args.cp = args.cp or divisors(world_size)
            mode = f"physical-{world_size}gpu-tp-cp"
        wrong_tp = [value for value in args.tp if value != world_size]
        if wrong_tp:
            raise ValueError(
                "one physical job requires every --tp value to equal WORLD_SIZE "
                f"({world_size}); got {wrong_tp}. Use --gpus N and --tp N in "
                "a separate command for another TP degree."
            )
        args.cp_rank = [0]
    elif args.heads is not None:
        args.cp = args.cp or [1]
        mode = "custom-heads"
    elif not tp_present and not cp_present:
        # Default: H64 first (TP=2), then H128 (TP=1), both at CP=1.
        args.tp = [2, 1]
        args.cp = [1]
        mode = "h64-h128"
    elif tp_present and not cp_present:
        args.tp = args.tp or list(TP_SWEEP)
        args.cp = [1]
        mode = "tp-sweep"
    elif not tp_present and cp_present:
        args.tp = [8]
        args.cp = args.cp or list(CP_SWEEP)
        mode = "cp-sweep"
    else:
        args.tp = args.tp or list(TP_SWEEP)
        args.cp = args.cp or list(CP_SWEEP)
        mode = "tp-cp-matrix"

    if not args.physical_mode and args.cp_rank is None:
        args.cp_rank = [0]
    if not args.physical_mode and args.all_ranks:
        args.cp_rank = list(range(max(args.cp)))
    args.suite = args.suite or mode


def selected_backends(args) -> tuple[str, ...]:
    selected = list(DEFAULT_BACKENDS)
    optional = (
        ("bf16", args.bf16),
        ("fp8_dequant", args.fp8_dequant),
        ("trtllm", args.trtllm),
        ("decode_chunked", args.decode_chunked),
    )
    selected.extend(name for name, enabled in optional if enabled)
    return tuple(selected)


def print_available_kernels() -> None:
    print("default comparison:")
    print("  decode          FlashMLA FP8 sparse decode used as prefill")
    print("  v8              native FP8 sparse-prefill kernel")
    print("optional comparisons:")
    print("  --bf16          FlashMLA BF16 sparse prefill")
    print("  --fp8-dequant   V3.2 FP8 cache dequantization + BF16 prefill")
    print("  --trtllm        TRTLLM-GEN FP8 through FlashInfer")
    print("  --decode-chunked  sequential decode chunks when direct decode is too large")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark DeepSeek V3.2 FP8 sparse prefill. With no arguments, "
            "runs BS 1,2,4,8,16,32 and compares decode vs v8. One visible GPU "
            "keeps the H64/H128 rank-shape microbenchmark; two or more visible "
            "GPUs auto-launch one process per GPU with real NCCL communication."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--suite", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("v32",),
        default=["v32"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--test", nargs="+", choices=("v8",), default=["v8"], help=argparse.SUPPRESS
    )
    parser.add_argument("--batch", nargs="+", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--new", nargs="+", type=int, default=[875])
    parser.add_argument("--cached", nargs="+", type=int, default=[69125])
    parser.add_argument(
        "--global-heads", nargs="+", type=int, default=[128], help=argparse.SUPPRESS
    )
    parallel = parser.add_mutually_exclusive_group()
    parallel.add_argument(
        "--tp",
        "--tp-size",
        dest="tp",
        nargs="*",
        type=int,
        help=(
            "one GPU: no values means TP 1,2,4,8 shape sweep; physical job: "
            "TP must equal the number of participating GPUs"
        ),
    )
    parallel.add_argument(
        "--heads",
        nargs="+",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cp",
        nargs="*",
        type=int,
        help=(
            "one GPU: no values means CP 1,2,4,8 at TP=8; physical job: no "
            "values means every CP divisor, while default uses SGLang endpoints"
        ),
    )
    rank_group = parser.add_mutually_exclusive_group()
    rank_group.add_argument("--cp-rank", nargs="+", type=int, default=None)
    rank_group.add_argument(
        "--all-ranks",
        action="store_true",
        help="benchmark every CP rank instead of rank 0 only",
    )
    parser.add_argument("--topk", nargs="+", type=int, default=[2048])
    parser.add_argument("--bf16", action="store_true", help="also compare BF16 prefill")
    parser.add_argument(
        "--fp8-dequant",
        action="store_true",
        help="also compare FP8-cache dequantization plus BF16 prefill",
    )
    parser.add_argument(
        "--trtllm",
        action="store_true",
        help="also compare TRTLLM-GEN FP8 through FlashInfer",
    )
    parser.add_argument(
        "--decode-chunked",
        action="store_true",
        help="also compare sequential decode chunks without output concatenation",
    )
    parser.add_argument(
        "--list-kernels",
        action="store_true",
        help="show default and optional comparison kernels, then exit",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke timing with 2 warmups and 5 measured iterations",
    )
    parser.add_argument("--flush-mb", type=int, default=256)
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help=(
            "number of visible GPUs to use; 0 auto-uses all visible GPUs, "
            "and --gpus 1 forces the single-GPU path"
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--csv",
        "--out",
        dest="csv",
        default="benchmark/v8/results.csv",
        help="CSV output path; empty string disables CSV",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--verbose-cases", action="store_true")
    parser.add_argument(
        "--native-kernelv8-module",
        default=DEFAULT_ADAPTER,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--distributed-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def parse_args(
    argv: Optional[list[str]] = None,
    *,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
):
    args = build_arg_parser().parse_args(argv)
    args.distributed_world_size = distributed_world_size
    args.distributed_rank = distributed_rank
    args.physical_mode = distributed_world_size > 1
    resolve_parallel_presets(args)
    if args.quick:
        args.warmup = 2
        args.iters = 5
    if args.iters <= 0:
        raise ValueError(f"--iters must be positive, got {args.iters}")
    if args.warmup < 0:
        raise ValueError(f"--warmup must be non-negative, got {args.warmup}")
    if args.flush_mb < 0:
        raise ValueError(f"--flush-mb must be non-negative, got {args.flush_mb}")
    if args.limit < 0:
        raise ValueError(f"--limit must be non-negative, got {args.limit}")
    return args


def _launcher_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-kernels", action="store_true")
    parser.add_argument("--distributed-worker", action="store_true")
    return parser.parse_known_args(argv)[0]


def has_complete_torchrun_env() -> bool:
    present = {name for name in TORCHRUN_ENV_KEYS if name in os.environ}
    if present and len(present) != len(TORCHRUN_ENV_KEYS):
        missing = sorted(set(TORCHRUN_ENV_KEYS) - present)
        raise ValueError(
            "partial torchrun environment; missing " + ", ".join(missing)
        )
    return bool(present)


def selected_gpu_count(argv: list[str], *, allow_unavailable: bool = False) -> int:
    launcher = _launcher_args(argv)
    if launcher.gpus < 0:
        raise ValueError(f"--gpus must be non-negative, got {launcher.gpus}")
    available = torch.cuda.device_count()
    selected = launcher.gpus or available
    if selected > available and not allow_unavailable:
        raise ValueError(
            f"requested {selected} GPUs, but only {available} are visible; "
            "check CUDA_VISIBLE_DEVICES or lower --gpus"
        )
    return max(selected, 1 if allow_unavailable else 0)


def launcher_watchdog_seconds() -> int:
    raw = os.getenv(
        "MLA_PREFILL_BENCH_WATCHDOG_SECONDS",
        str(DEFAULT_LAUNCHER_WATCHDOG_SECONDS),
    )
    try:
        seconds = int(raw)
    except ValueError as exc:
        raise ValueError(
            "MLA_PREFILL_BENCH_WATCHDOG_SECONDS must be an integer"
        ) from exc
    if seconds <= 0:
        raise ValueError(
            "MLA_PREFILL_BENCH_WATCHDOG_SECONDS must be positive"
        )
    return seconds


def terminate_process_group(process: subprocess.Popen) -> None:
    """Stop torchrun and all of its workers without leaving orphaned ranks."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def maybe_launch_distributed(argv: list[str]) -> Optional[int]:
    """Auto-launch torchrun once when more than one CUDA device is visible."""

    launcher = _launcher_args(argv)
    already_launched = launcher.distributed_worker or has_complete_torchrun_env()
    if already_launched or launcher.dry_run or launcher.list_kernels:
        return None
    if any(flag in argv for flag in ("-h", "--help")):
        return None
    gpu_count = selected_gpu_count(argv)
    if gpu_count <= 1:
        return None
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={gpu_count}",
        str(Path(__file__).resolve()),
        *argv,
        "--distributed-worker",
    ]
    environment = os.environ.copy()
    environment.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    environment.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    watchdog_seconds = launcher_watchdog_seconds()
    print(
        f"[launcher] {gpu_count} visible GPUs -> one NCCL worker per GPU",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        env=environment,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=watchdog_seconds)
    except subprocess.TimeoutExpired:
        print(
            "[launcher] distributed benchmark exceeded the "
            f"{watchdog_seconds}s watchdog; terminating every worker",
            file=sys.stderr,
            flush=True,
        )
        terminate_process_group(process)
        return 124
    except KeyboardInterrupt:
        terminate_process_group(process)
        raise


def initialize_distributed(args) -> DistributedContext:
    has_torchrun_env = has_complete_torchrun_env()
    if args.distributed_worker and not has_torchrun_env:
        raise RuntimeError(
            "--distributed-worker is internal and requires a complete torchrun "
            "environment"
        )
    launched = has_torchrun_env
    if not launched:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return DistributedContext()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size <= 0:
        raise RuntimeError(f"WORLD_SIZE must be positive, got {world_size}")
    if local_world_size <= 0:
        raise RuntimeError(
            f"LOCAL_WORLD_SIZE must be positive, got {local_world_size}"
        )
    if rank < 0 or rank >= world_size:
        raise RuntimeError(f"RANK must be in [0, {world_size}), got {rank}")
    if local_rank < 0 or local_rank >= local_world_size:
        raise RuntimeError(
            f"LOCAL_RANK must be in [0, {local_world_size}), got {local_rank}"
        )
    if local_world_size != world_size:
        raise RuntimeError(
            "this benchmark supports one physical node only; "
            f"WORLD_SIZE={world_size}, LOCAL_WORLD_SIZE={local_world_size}"
        )
    if world_size != args.distributed_world_size:
        raise RuntimeError(
            f"parsed WORLD_SIZE={args.distributed_world_size}, runtime has {world_size}"
        )
    if args.gpus and args.gpus != world_size:
        raise RuntimeError(
            f"--gpus={args.gpus} must match torchrun WORLD_SIZE={world_size}"
        )
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} has no visible CUDA device "
            f"(count={torch.cuda.device_count()})"
        )
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=DISTRIBUTED_TIMEOUT,
        device_id=torch.device("cuda", local_rank),
    )
    control_group = (
        dist.new_group(backend="gloo", timeout=DISTRIBUTED_TIMEOUT)
        if world_size > 1
        else None
    )
    return DistributedContext(True, rank, local_rank, world_size, control_group)


def main(argv: Optional[list[str]] = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        launched_code = maybe_launch_distributed(raw_argv)
    except ValueError as exc:
        raise SystemExit(f"invalid launcher configuration: {exc}") from exc
    if launched_code is not None:
        if launched_code:
            raise SystemExit(launched_code)
        return

    launcher = _launcher_args(raw_argv)
    try:
        torchrun_env = has_complete_torchrun_env()
    except ValueError as exc:
        raise SystemExit(f"invalid distributed environment: {exc}") from exc
    if torchrun_env or launcher.distributed_worker:
        planned_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        planned_rank = int(os.environ.get("RANK", "0"))
        if planned_world_size <= 0:
            raise SystemExit(
                f"invalid distributed environment: WORLD_SIZE must be positive, "
                f"got {planned_world_size}"
            )
        if planned_rank < 0 or planned_rank >= planned_world_size:
            raise SystemExit(
                "invalid distributed environment: RANK must be in "
                f"[0, {planned_world_size}), got {planned_rank}"
            )
    elif launcher.dry_run:
        try:
            planned_world_size = selected_gpu_count(
                raw_argv, allow_unavailable=True
            )
        except ValueError as exc:
            raise SystemExit(f"invalid GPU configuration: {exc}") from exc
        planned_rank = 0
    else:
        planned_world_size = 1
        planned_rank = 0
    try:
        args = parse_args(
            raw_argv,
            distributed_world_size=planned_world_size,
            distributed_rank=planned_rank,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid benchmark configuration: {exc}") from exc
    if args.list_kernels:
        print_available_kernels()
        return
    try:
        cases = build_cases(args)
    except ValueError as exc:
        raise SystemExit(f"invalid benchmark configuration: {exc}") from exc

    backends = selected_backends(args)
    if args.dry_run:
        print(f"scenario: {args.suite}")
        print("backends: " + ", ".join(BACKEND_LABELS[name] for name in backends))
        print(f"planned cases: {len(cases)}")
        print_plan(cases, args)
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run on an SM100 B200/B300 host")

    context = initialize_distributed(args)
    v8_failed = False
    try:
        major, minor = torch.cuda.get_device_capability()
        if major != 10:
            raise SystemExit(f"native v8 requires SM100f, got SM {major}.{minor}")

        if context.enabled and context.world_size > 1:
            manifest = [
                (
                    case.name,
                    case.layout,
                    case.batch,
                    case.new_tokens,
                    case.cached_len,
                    case.tp_size,
                    case.cp_size,
                    case.global_heads,
                    case.logical_heads,
                    case.heads,
                    case.topk,
                )
                for case in cases
            ]
            manifests: list[Any] = [None] * context.world_size
            dist.all_gather_object(
                manifests, manifest, group=context.control_group
            )
            if any(candidate != manifest for candidate in manifests):
                raise RuntimeError("rank case manifests differ before benchmarking")

        if context.is_main:
            print(f"scenario: {args.suite}")
            print(
                "execution: "
                + (
                    f"physical NCCL on {context.world_size} GPUs"
                    if args.physical_mode
                    else "one-GPU rank-local shape microbenchmark"
                )
            )
            if args.physical_mode:
                print(
                    "latency: kernel=rank-max p50; CP=BF16 latent-KV "
                    "all-gather+rerange; TP=post-o_proj payload all-reduce probe"
                )
            print("backends: " + ", ".join(BACKEND_LABELS[name] for name in backends))
            print(f"planned cases: {len(cases)}")
            print_plan(cases, args)

        rank_groups = create_rank_groups(cases, context)
        all_rows: list[dict[str, str]] = []
        for index, case in enumerate(cases, 1):
            if context.is_main and not args.quiet_progress:
                print(
                    f"[{index}/{len(cases)}] B={case.batch} rows={case.rows} "
                    f"(logical={case.logical_rows}, pad={case.padding_rows}) "
                    f"TP={case.tp_size} CP={case.cp_size}/rank{case.cp_rank} "
                    f"H={case.global_heads}->{case.logical_heads}->{case.heads} "
                    f"topk={case.effective_topk}",
                    flush=True,
                )

            tensors = None
            communication = None
            setup_error = None
            try:
                tensors = build_tensors(case, backends)
                if args.physical_mode:
                    groups = rank_groups[
                        (case.tp_size, case.cp_size, case.global_heads)
                    ]
                    communication = build_communication_buffers(case, groups)
            except Exception as exc:  # noqa: BLE001
                setup_error = short_err(exc)
            setup_error = sync_setup_error(setup_error, context)
            if setup_error:
                if args.physical_mode:
                    raise RuntimeError(
                        "distributed CUDA case setup failed; aborting the job "
                        f"instead of risking collective divergence: {setup_error}"
                    )
                if context.is_main:
                    for backend in backends:
                        all_rows.append(
                            result_row(
                                case,
                                TimeResult(backend, False, note=setup_error),
                                case.heads,
                                communication,
                            )
                        )
                v8_failed = True
                continue
            assert tensors is not None

            cp_ms = 0.0
            tp_ms = 0.0
            if communication is not None:
                if communication.has_cp:
                    cp_ms = time_distributed_operation(
                        "cp_latent_kv_allgather",
                        communication.run_cp,
                        args.iters,
                        args.warmup,
                        args.flush_mb,
                        context,
                    )
                if communication.has_tp:
                    tp_ms = time_distributed_operation(
                        "tp_o_proj_allreduce_probe",
                        communication.run_tp,
                        args.iters,
                        args.warmup,
                        args.flush_mb,
                        context,
                    )

            for backend in backends:
                runner = None
                runner_error = None
                runner_error_fatal = False
                try:
                    runner = make_backend_runner(backend, case, tensors, args)
                except Exception as exc:  # noqa: BLE001
                    runner_error = short_err(exc)
                    runner_error_fatal = is_fatal_cuda_exception(exc)
                    if runner_error_fatal:
                        runner_error = "fatal-cuda: " + runner_error
                runner_error = sync_setup_error(runner_error, context)
                if runner_error:
                    if args.physical_mode and "fatal-cuda:" in runner_error:
                        raise RuntimeError(
                            "distributed CUDA backend setup failed; aborting "
                            f"the job: {runner_error}"
                        )
                    result = TimeResult(
                        backend,
                        False,
                        note=runner_error,
                        cp_comm_ms=cp_ms,
                        tp_comm_ms=tp_ms,
                    )
                    if backend == "v8":
                        v8_failed = True
                elif args.physical_mode:
                    assert runner is not None and communication is not None
                    kernel_ms = time_distributed_operation(
                        f"{backend}_kernel",
                        runner,
                        args.iters,
                        args.warmup,
                        args.flush_mb,
                        context,
                    )
                    cp_envelope_ms = float("nan")
                    if communication.has_cp:
                        cp_envelope_ms = time_distributed_operation(
                            f"{backend}_cp_allgather_probe_plus_kernel",
                            lambda: communication.run_cp_envelope(runner),
                            args.iters,
                            args.warmup,
                            args.flush_mb,
                            context,
                        )
                    result = TimeResult(
                        backend,
                        True,
                        kernel_ms,
                        kernel_ms=kernel_ms,
                        cp_comm_ms=cp_ms,
                        tp_comm_ms=tp_ms,
                        envelope_ms=cp_envelope_ms,
                    )
                else:
                    assert runner is not None
                    result = time_kernel(
                        backend, runner, args.iters, args.warmup, args.flush_mb
                    )
                    if backend == "v8" and not result.ok:
                        v8_failed = True

                if context.is_main:
                    all_rows.append(
                        result_row(case, result, case.heads, communication)
                    )
                    if not args.quiet_progress:
                        if result.ok and args.physical_mode:
                            details = [f"kernel={result.kernel_ms:.4f} ms"]
                            if result.cp_comm_ms:
                                details.append(f"CP_AG={result.cp_comm_ms:.4f} ms")
                            if result.tp_comm_ms:
                                details.append(f"TP_AR_probe={result.tp_comm_ms:.4f} ms")
                            if result.envelope_ms == result.envelope_ms:
                                details.append(
                                    f"CPprobe+kernel={result.envelope_ms:.4f} ms"
                                )
                            outcome = ", ".join(details)
                        elif result.ok:
                            outcome = f"{result.ms:.4f} ms"
                        else:
                            outcome = f"n/a: {result.note}"
                        print(f"  {BACKEND_LABELS[backend]}={outcome}")

        if context.enabled:
            dist.barrier()
        if context.is_main:
            print_result_matrix(all_rows, set(backends))
            if args.csv:
                write_csv(args.csv, all_rows)
        if v8_failed:
            raise SystemExit("one or more v8 benchmark cases failed")
    finally:
        if context.enabled and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
