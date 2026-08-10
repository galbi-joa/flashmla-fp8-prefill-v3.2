# FlashMLA V3.2 Native FP8 Sparse Prefill

This repository contains the SM100 native-FP8 sparse-prefill path for the
DeepSeek V3.2 KV-cache layout:

## Tensor shapes that are actually measured

For one planned case, setup creates the following tensors before timing:

| Tensor | Shape | Type |
| --- | --- | --- |
| Query | `[rows, Hexec, 576]` | BF16 |
| V3.2 FP8 KV cache | `[ceil(kv_len / 64), 64, 1, 656]` | FP8 byte container |
| Sparse indices | `[rows, 1, topk_effective]` | INT32 |
| Output | `[rows, Hexec, 512]` | BF16 |
| Max logits | `[rows, Hexec]` | FP32 |
| LSE | `[rows, Hexec]` | FP32 |

`batch` is represented by flattening its rank-local query tokens into `rows`.
The microbenchmark uses one visible KV pool of length `cached + new`; it does
not allocate a separate KV pool per batch item. This isolates the kernel's
query-row workload and matches the flat native-FP8 API boundary.

The fixed layout and execution contract are:

- query/key dimension: `d_qk = 576`
- value/output dimension: `d_v = 512`
- KV-cache page size: 64 tokens
- KV-cache storage: 656 bytes per token
- supported execution head extents: H64 and H128

The CUDA kernel computes 64 query heads per launch. H64 therefore uses one
kernel launch. H128 allocates `out`, `max_logits`, and `lse` once at their full
H128 shapes, then launches the H64 kernel twice into disjoint head slices.
There is no output concatenation, combine kernel, D2D copy, or cross-head
reduction.

Detailed companion documentation is also available in
[English](docs/v32_native_fp8_prefill_v8.md) and
[简体中文](docs/v32_native_fp8_prefill_v8_zh-CN.md).

## 1. Compile the library

### Requirements

- NVIDIA B200 or B300 (`SM100f`)
- CUDA 12.9 or newer; the validated B300 environment used CUDA 13.0
- a CUDA-enabled PyTorch build compatible with the installed CUDA toolkit
- Python 3.10 or newer
- `setuptools`, `wheel`, and `ninja`
- `kernelkit` only when running the numerical correctness tests

### Clone and initialize CUTLASS

```bash
git clone https://github.com/galbi-joa/FlashMLA_v3.2_fp8_prefill.git
cd FlashMLA_v3.2_fp8_prefill
git submodule update --init --recursive
```

Because the repository is private, GitHub must authenticate the clone through
your configured HTTPS credential or SSH key.

### Prepare a Python environment

Use an environment that already has a suitable CUDA-enabled PyTorch build.
For example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja

# Install a CUDA-enabled PyTorch build appropriate for this machine.
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

### Build SM100 in place

Adjust `CUDA_HOME` if CUDA is installed elsewhere:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_MLA_DISABLE_SM90=1
export NVCC_THREADS=8
export MAX_JOBS=8

python setup.py build_ext --inplace -v
```

The in-place build creates `flash_mla/cuda*.so`. Always put this checkout
before site-packages when testing so an older installed extension is not loaded:

```bash
PYTHONPATH="$PWD" python - <<'PY'
import torch
import flash_mla.cuda as cuda

print("extension:", cuda.__file__)
print("GPU:", torch.cuda.get_device_name(0))
assert hasattr(cuda, "v32_kernelv8_fp8_prefill_fwd")
assert hasattr(cuda, "v32_kernelv8_fp8_prefill_trace")
PY
```

To install instead of using an in-place build, keep the same environment
variables and run:

```bash
python -m pip install -v --no-build-isolation .
```

### Correctness and launch validation

The repository uses one consolidated native-FP8 test file for H64, H128, edge
cases, trace validation, launch auditing, and Compute Sanitizer entry points.

```bash
# Full H64 and H128 numerical suite.
PYTHONPATH="$PWD" python -B \
  tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads all --case all

# CUPTI audit: H64 must launch once and H128 exactly twice.
PYTHONPATH="$PWD" python -B \
  tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads 128 --case edge --launch-only --audit-launches \
  --trace-json /tmp/v8_h128_launches.json

# Minimal memory-safety check.
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="$PWD" \
compute-sanitizer --tool memcheck --error-exitcode=1 \
  python -B tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads 128 --case edge --launch-only
```

The launch audit fails if H64 does not produce exactly one native kernel
activity, if H128 does not produce exactly two, or if it observes a CUDA
combine, copy, memset, or `aten::cat` inside the profiled API call.

## 2. Benchmark the native FP8 kernel

Run the standalone benchmark from the repository root:

```bash
PYTHONPATH="$PWD" python -B benchmark/v8/mla_fp8_prefill_bench.py --help
```

The benchmark imports neither SGLang nor FlashInfer and never installs
dependencies. It measures only `v32_kernelv8_fp8_prefill_fwd`.

### What one benchmark case means

The planner models an SGLang-style DP=1 attention group. Each requested list is
part of a Cartesian product, after invalid configurations are filtered.

| CLI value | Meaning |
| --- | --- |
| `--batch` | Number of requests, reported as BS. It multiplies rank-local query rows. |
| `--new` | New query tokens per request before context-parallel partitioning. |
| `--cached` | Cached tokens already visible to the rank. |
| `--tp` | Total tensor-parallel degree, including the CP subdivision. |
| `--cp` | Attention context-parallel degree. `TP % CP == 0` is required. |
| `--cp-rank` | The CP rank represented by this single-process microbenchmark. |
| `--global-heads` | Model query heads before attention-TP sharding. |
| `--topk` | Requested sparse KV selections per query row. |

For every valid case, the benchmark derives:

```text
attention_tp = TP / CP
Hlocal       = global_heads / attention_tp
Hexec        = 64  when 1 <= Hlocal <= 64
             = 128 when 65 <= Hlocal <= 128

quotient, remainder = divmod(new, CP)
q_per_rank           = quotient + int(cp_rank < remainder)
rows                 = batch * q_per_rank
kv_len               = cached + new
topk_effective       = floor(min(topk, kv_len) / 64) * 64
```

`global_heads` must be divisible by `attention_tp`, `Hexec` may not exceed 128,
and `topk_effective` must be at least 64. Smaller `Hlocal` values are zero-padded
to H64 or H128 before timing; the kernel still executes the full `Hexec` extent.

All random input generation, KV quantization, and sparse-index construction
happen outside the timed interval. Each measured iteration flushes the requested
L2-sized buffer before the start CUDA event. The flush itself is excluded from
elapsed time. Results report the median of CUDA-event samples, so CPU setup and
Python dispatch time are not included. For H128, both sequential H64 CUDA
launches are inside the measured interval.

### Example A: H64 and H128, multiple batch sizes

These two runs use the same query/KV/top-k workload and batch-size sweep. Only
the head mapping changes.

#### H64: TP=2, CP=1

```bash
RUN_TAG=$(date +%Y%m%d_%H%M%S)
PYTHONPATH="$PWD" python -B benchmark/v8/mla_fp8_prefill_bench.py \
  --suite custom \
  --layouts v32 \
  --batch 1 2 4 8 16 32 \
  --new 875 \
  --cached 69125 \
  --global-heads 128 \
  --tp 2 \
  --cp 1 \
  --cp-rank 0 \
  --topk 2048 \
  --test v8 \
  --warmup 50 \
  --iters 300 \
  --flush-mb 256 \
  --no-install \
  --csv "benchmark/v8/tp2_h64_bs_${RUN_TAG}.csv"
```

Here `attention_tp=2`, `Hlocal=64`, and `Hexec=64`. Every timed API call
contains one H64 kernel launch. The measured row counts are `875, 1750, 3500,
7000, 14000, 28000` for BS `1, 2, 4, 8, 16, 32`.

#### H128: TP=1, CP=1

```bash
RUN_TAG=$(date +%Y%m%d_%H%M%S)
PYTHONPATH="$PWD" python -B benchmark/v8/mla_fp8_prefill_bench.py \
  --suite custom \
  --layouts v32 \
  --batch 1 2 4 8 16 32 \
  --new 875 \
  --cached 69125 \
  --global-heads 128 \
  --tp 1 \
  --cp 1 \
  --cp-rank 0 \
  --topk 2048 \
  --test v8 \
  --warmup 50 \
  --iters 300 \
  --flush-mb 256 \
  --no-install \
  --csv "benchmark/v8/tp1_h128_bs_${RUN_TAG}.csv"
```

Here `attention_tp=1`, `Hlocal=128`, and `Hexec=128`. Every timed API call
contains two H64 kernel launches. The measured row counts are `875, 1750, 3500,
7000, 14000, 28000` for BS `1, 2, 4, 8, 16, 32`.

### Example B: fixed TP=8 with several CP values

```bash
RUN_TAG=$(date +%Y%m%d_%H%M%S)
PYTHONPATH="$PWD" python -B benchmark/v8/mla_fp8_prefill_bench.py \
  --suite custom \
  --layouts v32 \
  --batch 1 8 \
  --new 10000 \
  --cached 90000 \
  --global-heads 128 \
  --tp 8 \
  --cp 1 2 4 8 \
  --cp-rank 0 \
  --topk 512 1024 2048 \
  --test v8 \
  --warmup 50 \
  --iters 300 \
  --flush-mb 256 \
  --no-install \
  --csv "benchmark/v8/tp8_cp_bs_topk_${RUN_TAG}.csv"
```

The head mapping is exact:

| TP | CP | attention TP | Hlocal | Hexec | CUDA launches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 8 | 16 | 64 | 1 |
| 8 | 2 | 4 | 32 | 64 | 1 |
| 8 | 4 | 2 | 64 | 64 | 1 |
| 8 | 8 | 1 | 128 | 128 | 2 |

### Example C: TP × CP matrix

```bash
RUN_TAG=$(date +%Y%m%d_%H%M%S)
PYTHONPATH="$PWD" python -B benchmark/v8/mla_fp8_prefill_bench.py \
  --suite custom \
  --layouts v32 \
  --batch 1 4 16 \
  --new 875 \
  --cached 69125 \
  --global-heads 128 \
  --tp 1 2 4 8 \
  --cp 1 2 4 8 \
  --cp-rank 0 \
  --topk 2048 \
  --test v8 \
  --warmup 30 \
  --iters 100 \
  --flush-mb 256 \
  --no-install \
  --csv "benchmark/v8/tp_cp_matrix_${RUN_TAG}.csv"
```

The planner records but skips invalid pairs such as `TP=2, CP=4`. Valid pairs
are printed before allocation and stored with separate `tp_size`,
`attention_tp_size`, `cp_size`, `global_heads`, `local_heads`, and
`execution_heads` CSV columns.

### Example D: inspect every CP rank and a remainder split

```bash
PYTHONPATH="$PWD" python -B benchmark/v8/mla_fp8_prefill_bench.py \
  --suite custom \
  --layouts v32 \
  --batch 1 \
  --new 875 \
  --cached 69125 \
  --global-heads 128 \
  --tp 8 \
  --cp 8 \
  --cp-rank 0 1 2 3 4 5 6 7 \
  --topk 2048 \
  --test v8 \
  --warmup 30 \
  --iters 100 \
  --flush-mb 256 \
  --no-install \
  --csv benchmark/v8/cp_rank_remainder.csv
```

For `new=875` and `CP=8`, ranks 0-2 receive 110 query tokens and ranks 3-7
receive 109. Their sum remains 875; the benchmark never silently discards the
remainder. These ranks are benchmarked sequentially in one process, not as a
distributed multi-GPU job.

### Preview without using a GPU

Add `--dry-run` to any command to print every derived shape, valid TP/CP pair,
skipped pair, head mapping, rank-local row count, and effective top-k without
allocating CUDA tensors:

```bash
python -B benchmark/v8/mla_fp8_prefill_bench.py \
  --suite custom --batch 1 8 --new 10000 --cached 90000 \
  --global-heads 128 --tp 8 --cp 1 2 4 8 --cp-rank 0 \
  --topk 512 1024 2048 --test v8 --dry-run --csv ""
```

The CPU-only benchmark-planner tests are:

```bash
python -B -m unittest benchmark.v8.test_mla_fp8_prefill_bench
```

## 3. Files changed

The H64/H128 implementation added or modified the following files on top of the
pre-existing FP8 development branch:

| File | Purpose |
| --- | --- |
| `csrc/sm100/prefill/sparse/fwd_v32_kernelv8.cu` | SM100 sparse-prefill CUDA kernel and the H64 launch implementation. |
| `csrc/api/v32_kernelv8_fp8_prefill_fwd_bridge.h` | PyTorch bridge, validation, full-shape output allocation, and H64/H128 dispatch. |
| `csrc/sm100/prefill/sparse/v8_trace_schema.h` | Trace-buffer schema used by the diagnostic API. |
| `csrc/sm100/prefill/sparse/common_subroutine.h` | Shared sparse-prefill helper required by this kernel path. |
| `csrc/api/api.cpp` | Python extension registration for the forward and trace entry points. |
| `setup.py` | Build registration for the added SM100 CUDA source. |
| `tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py` | Consolidated H64/H128 correctness, bitwise split comparison, trace, launch-count, and sanitizer test entry points. |
| `benchmark/v8/mla_fp8_prefill_bench.py` | Standalone benchmark runner and TP/CP/head-shape planner. |
| `benchmark/v8/v32_kernelv8_fp8_prefill_fwd_adapter.py` | Adapter from benchmark tensors to the native extension API. |
| `benchmark/v8/test_mla_fp8_prefill_bench.py` | CPU-only tests for benchmark planning and validation. |
| `benchmark/v8/README.md`, `benchmark/v8/__init__.py` | Benchmark package notes and package marker. |
| `docs/v32_native_fp8_prefill_v8.md` | Detailed English implementation and operating notes. |
| `docs/v32_native_fp8_prefill_v8_zh-CN.md` | Detailed Simplified Chinese implementation and operating notes. |
| `README.md` | Build, validation, measured tensor shapes, and benchmark workflows. |
| `.gitignore` | Excludes generated artifacts and private working materials from publication. |

Generated binaries, Python caches, benchmark outputs, PDFs, and private working
notes are not part of the repository.
