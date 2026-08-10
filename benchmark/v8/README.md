# v8 benchmark

This directory contains the v8-only V3.2 native-FP8 sparse-prefill benchmark,
its adapter, and CPU-only tests for TP/CP planning and CSV metadata.

- English build and benchmark guide: [v32_native_fp8_prefill_v8.md](../../docs/v32_native_fp8_prefill_v8.md)
- 中文编译与 benchmark 指南：[v32_native_fp8_prefill_v8_zh-CN.md](../../docs/v32_native_fp8_prefill_v8_zh-CN.md)

Run the planning tests without a GPU:

```bash
python -m unittest benchmark.v8.test_mla_fp8_prefill_bench
```

Preview a TP=1, CP=1 H128 batch sweep:

```bash
python -B benchmark/v8/mla_fp8_prefill_bench.py \
  --suite custom --batch 1 2 4 8 16 32 \
  --new 875 --cached 69125 --global-heads 128 \
  --tp 1 --cp 1 --cp-rank 0 --topk 2048 --test v8 --dry-run
```
