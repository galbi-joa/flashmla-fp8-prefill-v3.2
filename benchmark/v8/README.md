# FP8 sparse-prefill benchmark

This directory contains the V3.2 FP8 sparse-prefill benchmark, the native
prefill adapter, and CPU-only tests for TP/CP topology, SGLang-compatible query
padding, latency reduction, and CSV metadata.

- English build and benchmark guide: [v32_native_fp8_prefill_v8.md](../../docs/v32_native_fp8_prefill_v8.md)
- 中文编译与 benchmark 指南：[v32_native_fp8_prefill_v8_zh-CN.md](../../docs/v32_native_fp8_prefill_v8_zh-CN.md)

Run the planning tests without a GPU:

```bash
python -m unittest benchmark.v8.test_prefill_benchmark
```

The default command auto-uses one process on a one-GPU host or one NCCL worker
per visible GPU on a multi-GPU host:

```bash
PYTHONPATH="$PWD" python -B benchmark/v8/prefill_benchmark.py
```

One GPU retains the local H64/H128 shape benchmark. With `N >= 2` GPUs, the
physical default runs `TP=N,CP=1` and `TP=N,CP=N`; rank-local kernel latency,
BF16 latent-KV CP all-gather, and a correctly shaped post-o_proj TP all-reduce
probe are reported separately. NCCL uses the machine's real NVLink/PCIe
transport. These communication values are scoped probes, not a full-layer
end-to-end or distributed-output correctness claim.

Distributed runtime failures are coordinated through a Gloo control group;
NCCL groups use a 120-second failure timeout. The automatic launcher also has a
two-hour whole-job watchdog, configurable with
`MLA_PREFILL_BENCH_WATCHDOG_SECONDS`.

Preview an eight-GPU CP topology without using a GPU:

```bash
python -B benchmark/v8/prefill_benchmark.py --dry-run --gpus 8 --cp --csv ""
```

The equivalent short module command is `python -m benchmark.v8`.
