# FP8 sparse-prefill benchmark

This directory contains the V3.2 FP8 sparse-prefill benchmark, the native
prefill adapter, and CPU-only tests for TP/CP planning and CSV metadata.

- English build and benchmark guide: [v32_native_fp8_prefill_v8.md](../../docs/v32_native_fp8_prefill_v8.md)
- 中文编译与 benchmark 指南：[v32_native_fp8_prefill_v8_zh-CN.md](../../docs/v32_native_fp8_prefill_v8_zh-CN.md)

Run the planning tests without a GPU:

```bash
python -m unittest benchmark.v8.test_prefill_benchmark
```

Preview the default H64/H128, decode/v8 comparison without using a GPU:

```bash
python -B benchmark/v8/prefill_benchmark.py --dry-run
```

The equivalent short module command is `python -m benchmark.v8`.
