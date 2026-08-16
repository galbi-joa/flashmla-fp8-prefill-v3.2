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

Select the scenario that matches the hardware and topology you want to
measure. If `--gpus` is omitted, the runner uses every visible GPU; the
examples below select the device count explicitly so that the execution mode
is unambiguous.

**Quick smoke test — check that the kernel loads and produces timings on one
GPU.**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
python -B benchmark/v8/prefill_benchmark.py \
  --gpus 1 --quick --batch 1 --csv ""
```

**One GPU — benchmark both H64 and H128 over the canonical batch sweep.** The
one-GPU runner executes the H64 rank shape first and the H128 rank shape
second; TP/CP are shape metadata in this mode, not a distributed-performance
claim.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
python -B benchmark/v8/prefill_benchmark.py \
  --gpus 1 \
  --batch 1 2 4 8 16 32 \
  --new 875 --cached 69125 --topk 2048 \
  --warmup 10 --iters 50 --flush-mb 256 \
  --csv benchmark/v8/results_1gpu.csv
```

**Two GPUs — measure physical TP2/CP1 and TP2/CP2 with one NCCL worker per
GPU.** `CP=1` gives pure attention TP with H64 per rank. `CP=2` gives the DSA
CP endpoint with H128 and approximately half the query rows per rank.

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD" \
python -B benchmark/v8/prefill_benchmark.py \
  --gpus 2 --tp 2 --cp 1 2 \
  --batch 1 2 4 8 16 32 \
  --new 875 --cached 69125 --topk 2048 \
  --warmup 10 --iters 50 --flush-mb 256 \
  --csv benchmark/v8/results_2gpu_tp_cp.csv
```

**Preview a topology — inspect rank shapes and padding without allocating CUDA
tensors.**

```bash
python -B benchmark/v8/prefill_benchmark.py \
  --dry-run --gpus 2 --tp 2 --cp 1 2 \
  --batch 1 2 4 8 16 32 --csv ""
```

**Compare optional backends — add BF16 prefill, FP8-cache dequantization plus
BF16 prefill, TRTLLM-GEN, and the chunked decode floor.**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
python -B benchmark/v8/prefill_benchmark.py \
  --gpus 1 --batch 1 8 \
  --bf16 --fp8-dequant --trtllm --decode-chunked
```

Use `python -B benchmark/v8/prefill_benchmark.py --help` to list every option.
The equivalent short module command for the default scenario is:

```bash
python -m benchmark.v8
```

See the appendices below for tensor contracts, H128 launch details, distributed
topology, timing boundaries, and advanced benchmark options.

## Benchmark Result

All values below are kernel latency in milliseconds. The common workload is
DeepSeek V3.2 FP8 KV with 875 new tokens per request, 69,125 cached tokens,
KV length 70,000, and top-k 2,048. Each timed iteration uses a 256MiB L2 flush.
Input generation, KV quantization, sparse-index construction, and full-model
work are outside the timed interval.

### One GPU: H64 and H128 across batch sizes

H64 was measured on an NVIDIA B200 (50 warmups, 300 timed iterations), while
H128 was measured on an NVIDIA B300 (10 warmups, 50 timed iterations). H128 is
implemented as two H64 launches into disjoint output slices. Because the two
head extents were measured on different GPU models, the columns are independent
results rather than a hardware-normalized H64-versus-H128 comparison.

Speedup is `(direct FP8 decode - v8) / direct FP8 decode` and is calculated
from the unrounded measurements.

| Batch | H64 v8 (B200) | H64 direct FP8 decode | H64 v8 vs decode | H128 v8 (B300) | H128 direct FP8 decode | H128 v8 vs decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **0.3727** | 0.3828 | **2.63% faster** | **0.7076** | 0.7211 | **1.87% faster** |
| 2 | **0.7301** | 0.7741 | **5.69% faster** | **1.3998** | 1.4940 | **6.30% faster** |
| 4 | **1.4469** | 1.5556 | **6.99% faster** | **2.7842** | 3.0064 | **7.39% faster** |
| 8 | **2.8972** | 3.0823 | **6.00% faster** | **5.5491** | 5.9730 | **7.10% faster** |
| 16 | **5.7469** | N/A | N/A | **11.1309** | N/A | N/A |
| 32 | **11.5210** | N/A | N/A | **22.3345** | N/A | N/A |

Direct FP8 decode is unavailable at batch 16 and 32 because its folded
metadata requires 273KiB and 546KiB of shared memory, respectively, exceeding
the 227KiB device limit. The v8 prefill kernel completes both cases.

### Two GPUs: physical TP/CP

The physical run used two NV18-connected NVIDIA B300 SXM6 GPUs with one NCCL
worker per GPU. Global H128 was mapped to the two measured topologies as
follows:

- `TP2/CP1`: pure attention TP, H64 per GPU, all query rows per rank.
- `TP2/CP2`: CP endpoint, H128 per GPU, approximately half the query rows per
  rank.

Results are the average of two independent physical runs. Distributed latency
is the median of the iteration-wise maximum rank latency. Speedup is calculated
from the unrounded measurements.

**TP2/CP1 — H64 per GPU**

| Batch | Direct FP8 decode | v8 prefill | v8 vs decode |
| ---: | ---: | ---: | ---: |
| 1 | 0.3812 | **0.3692** | **3.15% faster** |
| 2 | 0.7676 | **0.7152** | **6.84% faster** |
| 4 | 1.5256 | **1.4067** | **7.80% faster** |
| 8 | 3.0099 | **2.7912** | **7.27% faster** |
| 16 | N/A | **5.5018** | N/A |
| 32 | N/A | **10.9833** | N/A |

**TP2/CP2 — H128 per GPU**

| Batch | Direct FP8 decode | v8 prefill | v8 vs decode |
| ---: | ---: | ---: | ---: |
| 1 | 0.3895 | **0.3743** | **3.91% faster** |
| 2 | 0.7406 | **0.7213** | **2.61% faster** |
| 4 | 1.5129 | **1.4129** | **6.61% faster** |
| 8 | 3.0266 | **2.7969** | **7.59% faster** |
| 16 | 5.9956 | **5.5600** | **7.27% faster** |
| 32 | N/A | **10.9983** | N/A |

Direct FP8 decode is N/A at TP2/CP1 batch 16/32 and TP2/CP2 batch 32 because
its metadata exceeds the 227KiB shared-memory limit. The v8 kernel completes
all cases. The maximum v8 difference between the two repeated runs is 1.30%.
Both GPUs independently passed 11 H64 and 11 H128 correctness cases (44/44
total), with no CUDA or NCCL errors.

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
