# Native FP8 Sparse Prefill for DeepSeek V3.2

This repository adds an SM100 sparse-prefill path for the DeepSeek V3.2 FP8
KV-cache format to FlashMLA.

## Highlights

- Reads the V3.2 656-byte FP8 KV-cache format directly.
- Supports H64 and H128 execution on NVIDIA B200 and B300 GPUs.
- Includes numerical correctness, edge-case, launch-audit, and sanitizer tests.
- Includes standalone single-GPU and physical multi-GPU TP/CP benchmarks.

## Requirements

- NVIDIA B200 or B300 (`SM100f`)
- CUDA 12.9 or newer
- CUDA-enabled PyTorch compatible with the installed CUDA toolkit
- Python 3.10 or newer
- `setuptools`, `wheel`, and `ninja`
- `kernelkit` when running the numerical correctness tests

## Build

```bash
git clone https://github.com/galbi-joa/flashmla-fp8-prefill-v3.2.git
cd flashmla-fp8-prefill-v3.2
git submodule update --init --recursive

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_MLA_DISABLE_SM90=1
export NVCC_THREADS=8
export MAX_JOBS=8

python setup.py build_ext --inplace -v
```

Verify that Python loads the extension from this checkout:

```bash
PYTHONPATH="$PWD" python - <<'PY'
import flash_mla.cuda as cuda

print(cuda.__file__)
assert hasattr(cuda, "v32_kernelv8_fp8_prefill_fwd")
PY
```

To install the package instead, keep the same build environment and run:

```bash
python -m pip install -v --no-build-isolation .
```

## Validate

Run the complete H64/H128 numerical suite:

```bash
PYTHONPATH="$PWD" python -B \
  tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads all --case all
```

Run the CPU-only benchmark-planner tests:

```bash
python -B -m unittest benchmark.v8.test_prefill_benchmark
```

Additional launch-audit and Compute Sanitizer commands are in
[Appendix D](#appendix-d-advanced-validation).

## Benchmark

The default command automatically uses one process on a one-GPU host or one
NCCL worker per visible GPU on a multi-GPU host:

```bash
PYTHONPATH="$PWD" python -B benchmark/v8/prefill_benchmark.py
```

Useful examples:

```bash
# Show all options.
python -B benchmark/v8/prefill_benchmark.py --help

# Preview an eight-GPU CP topology without allocating CUDA tensors.
python -B benchmark/v8/prefill_benchmark.py \
  --dry-run --gpus 8 --cp --csv ""

# Run physical TP=8 with CP=1,2,4,8 on an eight-GPU host.
python -B benchmark/v8/prefill_benchmark.py --gpus 8 --cp

# Add optional comparison paths.
python -B benchmark/v8/prefill_benchmark.py \
  --bf16 --fp8-dequant --trtllm --decode-chunked
```

The equivalent short module command is:

```bash
python -m benchmark.v8
```

See the appendices below for tensor contracts, H128 launch details, distributed
topology, timing boundaries, and advanced benchmark options.

## Repository layout

| Path | Purpose |
| --- | --- |
| `csrc/sm100/prefill/sparse/fwd_v32_kernelv8.cu` | Native V3.2 FP8 sparse-prefill kernel. |
| `csrc/api/v32_kernelv8_fp8_prefill_fwd_bridge.h` | PyTorch bridge and H64/H128 dispatch. |
| `tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py` | Consolidated correctness and validation suite. |
| `benchmark/v8/prefill_benchmark.py` | Single- and multi-GPU benchmark runner. |
| `benchmark/v8/test_prefill_benchmark.py` | CPU-only benchmark-planner tests. |

---

## Appendix A: Kernel and tensor contract

For one benchmark case, setup creates the following tensors before timing:

| Tensor | Shape | Type |
| --- | --- | --- |
| Query | `[rows, Hexec, 576]` | BF16 |
| V3.2 FP8 KV cache | `[ceil(kv_len / 64), 64, 1, 656]` | FP8 byte container |
| Sparse indices | `[rows, 1, topk_effective]` | INT32 |
| Output | `[rows, Hexec, 512]` | BF16 |
| Max logits | `[rows, Hexec]` | FP32 |
| LSE | `[rows, Hexec]` | FP32 |

The fixed execution contract is:

- query/key dimension: `d_qk = 576`
- value/output dimension: `d_v = 512`
- KV-cache page size: 64 tokens
- KV-cache storage: 656 bytes per token
- supported execution head extents: H64 and H128

The CUDA kernel computes 64 query heads per launch. H64 uses one launch. H128
allocates full-shape `out`, `max_logits`, and `lse` tensors and launches the H64
kernel twice into disjoint head slices. It does not concatenate outputs or run
a combine kernel, D2D copy, or cross-head reduction.

`batch` is represented by flattening query tokens into `rows`. Each process
uses one visible KV pool of length `cached + new`; it does not allocate a
separate KV pool per batch item. Physical CP applies SGLang-compatible padding
and rank partitioning to the flattened query-token stream.

## Appendix B: Benchmark topology

### Automatic execution mode

- With one visible GPU, the benchmark runs local H64/H128 kernel shapes. TP/CP
  values are shape metadata rather than a distributed-performance claim.
- With `N >= 2` visible GPUs, the benchmark relaunches through
  `torch.distributed.run`, assigns one NCCL worker per GPU, and uses `TP=N`.
- The physical default runs pure attention TP `(TP=N, CP=1)` and the current
  DP=1 DSA-CP endpoint `(TP=N, CP=N)`.
- `--gpus N` selects the first `N` visible devices. `--gpus 1` forces the local
  one-GPU path.

The automatic launcher is single-node. All participating devices must be
SM100 B200/B300 GPUs, and a physical job requires `WORLD_SIZE == TP`.

### Rank layout

The physical rank mapping is:

```text
attention_tp      = TP / CP
Hlocal            = global_heads / attention_tp
attention_tp_rank = global_rank % attention_tp
cp_rank           = global_rank // attention_tp

Hexec = 64   when 1 <= Hlocal <= 64
        128  when 65 <= Hlocal <= 128
```

Physical CP uses the following flattened padding and partitioning rule:

```text
Tlogical  = batch * new
Tpadded   = ceil_align(ceil_align(Tlogical, attention_tp), 2 * CP)
rows      = Tpadded / CP
logical_r = Tlogical // CP + int(r < Tlogical % CP)
padding_r = rows - logical_r

kv_len         = cached + new
topk_effective = floor(min(topk, kv_len) / 64) * 64
```

For example, `batch=1`, `new=875`, and `TP=CP=8` pad the stream to
880 tokens. Every rank executes 110 rows; ranks 0-2 have 110 logical rows,
while ranks 3-7 have 109 logical rows and one dummy row.

Smaller local-head counts are padded to the supported H64 or H128 execution
bucket. The CSV records logical and execution head counts separately.

### Topology status

- `CP=1` is the pure-TP path.
- Current DP=1 DSA CP uses the `CP=TP` endpoint.
- Intermediate factorizations such as `TP=8, CP=2/4` are real distributed
  measurements but are marked `experimental-tp-cp`.
- Endpoints beyond the current single-node DSA limit are marked
  `experimental-dsa-cp-over-8`.

## Appendix C: Measurement scope

Input generation, KV quantization, and sparse-index construction are outside
the timed interval. Each iteration flushes the requested L2-sized buffer before
the start event, and the flush itself is excluded from elapsed time.

In physical jobs, iterations are start-aligned across ranks. Kernel latency is
reported as the median of the iteration-wise maximum rank latency. Process
group setup, synchronization, error coordination, and result reduction remain
outside the kernel interval.

The distributed communication columns are scoped probes, not full-layer or
SGLang end-to-end latency:

| Metric | Timed boundary |
| --- | --- |
| `kernel_ms` | Rank-local attention kernel, reduced as synchronous rank-max p50. |
| `cp_latent_allgather_probe_ms` | NCCL all-gather of this layer's new/padded BF16 latent KV plus rank-major to token-major rearrangement. |
| `cp_allgather_probe_plus_kernel_ms` | The CP probe followed by the selected kernel on one CUDA stream. |
| `tp_o_proj_allreduce_probe_ms` | NCCL all-reduce of a correctly shaped post-`o_proj` BF16 payload. |

The benchmark does not all-reduce raw head-sharded attention output and does
not reconstruct distributed numerical output after trimming dummy token/head
padding. Projection GEMMs, indexer/top-k work, MLP communication, cached-prefix
gather, and model-final output gather are not included.

Runtime failures are coordinated through a Gloo control group. NCCL groups use
a 120-second timeout, and the automatic launcher uses a two-hour whole-job
watchdog. Set `MLA_PREFILL_BENCH_WATCHDOG_SECONDS` to override the latter.

## Appendix D: Advanced validation

Audit the H64/H128 launch topology with Torch Profiler/CUPTI:

```bash
PYTHONPATH="$PWD" python -B \
  tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads 128 --case edge --launch-only --audit-launches \
  --trace-json /tmp/v8_h128_launches.json
```

The audit expects one native-kernel activity for H64 and two for H128. It also
rejects CUDA combine/copy/memset activity or `aten::cat` inside the API interval.

Run a minimal Compute Sanitizer check:

```bash
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="$PWD" \
compute-sanitizer --tool memcheck --error-exitcode=1 \
  python -B tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads 128 --case edge --launch-only
```

## Appendix E: Advanced benchmark commands

On a one-GPU host, the following commands simulate TP/CP shapes:

```bash
python -B benchmark/v8/prefill_benchmark.py --tp
python -B benchmark/v8/prefill_benchmark.py --cp
python -B benchmark/v8/prefill_benchmark.py --tp --cp
```

Run an explicit matrix:

```bash
python -B benchmark/v8/prefill_benchmark.py \
  --tp 8 --cp 1 2 4 8 \
  --batch 1 8 \
  --new 10000 --cached 90000 \
  --topk 512 1024 2048
```

Inspect every simulated CP rank:

```bash
python -B benchmark/v8/prefill_benchmark.py \
  --tp 8 --cp 8 --all-ranks --batch 1 --new 875
```

In a physical job, CP rank is derived from the process topology, so
`--cp-rank` and `--all-ranks` are rejected.
