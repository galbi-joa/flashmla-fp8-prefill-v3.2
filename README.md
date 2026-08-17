# Native FP8 Sparse Prefill for DeepSeek V3.2

This FlashMLA extension provides a native FP8 sparse-prefill kernel for the
DeepSeek V3.2 KV-cache format, optimized for NVIDIA B200 and B300 GPUs.

## Highlights

- Natively supports DeepSeek V3.2 FP8 sparse prefill (H64 and H128) on NVIDIA
  B200 and B300 GPUs.
- Delivers up to 7.8% lower kernel latency than FlashMLA's direct FP8
  decode-as-prefill path in measured workloads.
- Includes numerical correctness, edge-case, launch-audit, and sanitizer tests.
- Includes standalone single-GPU and physical multi-GPU TP/CP benchmarks.
- **For a detailed explanation of the kernel design and execution pipeline, see
  the [technical report](docs/3.2_sm100_native_prefill_explain.md).** 

## Requirements

- NVIDIA B200 or B300 (`SM100f`)
- CUDA 12.9 or newer
- CUDA-enabled PyTorch built for the installed CUDA toolkit
- Python 3.10 or newer
- `setuptools`, `wheel`, and `ninja`
- `kernelkit` when running the numerical correctness tests

## Files Changed

| Path | Purpose |
| --- | --- |
| **`csrc/sm100/prefill/sparse/fwd_v32_kernelv8.cu`** | **Standalone Native V3.2 FP8 sparse-prefill kernel.** |
| `csrc/api/v32_kernelv8_fp8_prefill_fwd_bridge.h` | PyTorch bridge and H64/H128 dispatch. |
| `tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py` | Consolidated correctness and validation suite. |
| `benchmark/v8/prefill_benchmark.py` | Single- and multi-GPU benchmark runner. |
| `benchmark/v8/test_prefill_benchmark.py` | CPU-only benchmark-planner tests. |

## Build

On a minimal Debian or Ubuntu installation, install the Python venv and C++
build packages first:

```bash
apt-get update
apt-get install -y python3-venv python3-dev build-essential git
```

If `python3 -m venv` still reports that `ensurepip` is unavailable, install the
version-matched package, such as `python3.12-venv` for Python 3.12.

```bash
git clone https://github.com/galbi-joa/flashmla-fp8-prefill-v3.2.git
cd flashmla-fp8-prefill-v3.2
git submodule update --init --recursive

python3 -m venv --clear .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel ninja
python3 -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu130

python3 - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU"
assert torch.version.cuda and torch.version.cuda.startswith("13.0")
PY

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_MLA_DISABLE_SM90=1
export NVCC_THREADS=8
export MAX_JOBS=8

test -x "$CUDA_HOME/bin/nvcc"
python3 setup.py build_ext --inplace -v
```

Verify that Python loads the extension from this checkout:

```bash
PYTHONPATH="$PWD" python3 - <<'PY'
import flash_mla.cuda as cuda

print(cuda.__file__)
assert hasattr(cuda, "v32_kernelv8_fp8_prefill_fwd")
PY
```

To install the package instead, keep the same build environment and run:

```bash
python3 -m pip install -v --no-build-isolation .
```

## Validate

**Numerical suite — H=64/128 · cases=all**

```bash
PYTHONPATH="$PWD" python3 -B \
  tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads all --case all
```

**Planner tests — device=CPU · runner=unittest**

```bash
python3 -B -m unittest benchmark.v8.test_prefill_benchmark
```

**More:** [launch audit + Compute Sanitizer](#appendix-d-advanced-validation)

## Benchmark

**GPU selection:** `--gpus 1` = local 1-GPU mode · `--gpus N` (`N >= 2`) =
physical N-GPU mode · omitted = all visible GPUs

**Smoke — GPU=1 · B=1 · warmup=2 · iters=5 · CSV=off**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
python3 -B benchmark/v8/prefill_benchmark.py \
  --gpus 1 --quick --batch 1 --csv ""
```

**1 GPU — H64=(TPshape=2, CPshape=1) · H128=(TPshape=1, CPshape=1) ·
B=1/2/4/8/16/32 · NCCL workers=0**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
python3 -B benchmark/v8/prefill_benchmark.py \
  --gpus 1 \
  --batch 1 2 4 8 16 32 \
  --new 875 --cached 69125 --topk 2048 \
  --warmup 10 --iters 50 --flush-mb 256 \
  --csv benchmark/v8/results_1gpu.csv
```

**2 GPUs — TP=2 · CP=1/2 · B=1/2/4/8/16/32 · NCCL workers=2**

- `CP=1`: attention TP=2 · H/GPU=64 · B1 rows/rank=876
- `CP=2`: attention TP=1 · H/GPU=128 · B1 rows/rank=438

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD" \
python3 -B benchmark/v8/prefill_benchmark.py \
  --gpus 2 --tp 2 --cp 1 2 \
  --batch 1 2 4 8 16 32 \
  --new 875 --cached 69125 --topk 2048 \
  --warmup 10 --iters 50 --flush-mb 256 \
  --csv benchmark/v8/results_2gpu_tp_cp.csv
```

**Dry run — GPU=2 · TP=2 · CP=1/2 · CUDA allocations=0**

```bash
python3 -B benchmark/v8/prefill_benchmark.py \
  --dry-run --gpus 2 --tp 2 --cp 1 2 \
  --batch 1 2 4 8 16 32 --csv ""
```

**Optional backends — BF16 · FP8-dequant+BF16 · TRTLLM-GEN · decode-chunked**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
python3 -B benchmark/v8/prefill_benchmark.py \
  --gpus 1 --batch 1 8 \
  --bf16 --fp8-dequant --trtllm --decode-chunked
```

**Help:** `python3 -B benchmark/v8/prefill_benchmark.py --help`

**Default short form:**

```bash
python3 -m benchmark.v8
```

**Details:** [tensor contract](#appendix-a-kernel-and-tensor-contract) ·
[TP/CP topology](#appendix-b-benchmark-topology) ·
[timing scope](#appendix-c-measurement-scope) ·
[advanced commands](#appendix-e-advanced-benchmark-commands)

## Benchmark Result

| Setting | Value |
| --- | --- |
| KV layout | DeepSeek V3.2 FP8, 656B/token |
| New tokens | 875/request |
| Cached tokens | 69,125 |
| KV length | 70,000 |
| Top-k | 2,048 |
| Reported latency | Kernel, milliseconds |
| L2 flush | 256MiB/iteration |
| Timed | Kernel only |
| Excluded | Input generation, KV quantization, sparse-index construction, full model |

### 1 GPU: H64 and H128

**Speedup:** `(direct FP8 decode - v8) / direct FP8 decode` · source=unrounded
measurements

**H64 — NVIDIA B200 · warmup=50 · iters=300**

| Batch | Direct FP8 decode | v8 prefill | v8 vs decode |
| ---: | ---: | ---: | ---: |
| 1 | 0.3828 | **0.3727** | **2.63% faster** |
| 2 | 0.7741 | **0.7301** | **5.69% faster** |
| 4 | 1.5556 | **1.4469** | **6.99% faster** |
| 8 | 3.0823 | **2.8972** | **6.00% faster** |
| 16 | N/A | **5.7469** | N/A |
| 32 | N/A | **11.5210** | N/A |

**H128 — NVIDIA B300 · warmup=10 · iters=50**

| Batch | Direct FP8 decode | v8 prefill | v8 vs decode |
| ---: | ---: | ---: | ---: |
| 1 | 0.7211 | **0.7076** | **1.87% faster** |
| 2 | 1.4940 | **1.3998** | **6.30% faster** |
| 4 | 3.0064 | **2.7842** | **7.39% faster** |
| 8 | 5.9730 | **5.5491** | **7.10% faster** |
| 16 | N/A | **11.1309** | N/A |
| 32 | N/A | **22.3345** | N/A |

**Decode N/A:** B16=273KiB · B32=546KiB · SMEM limit=227KiB · v8=PASS

### 2 GPUs: physical TP/CP

**Hardware:** GPU=2× NVIDIA B300 SXM6 · interconnect=NV18 · NCCL workers=2 ·
global H=128

- `TP2/CP1`: attention TP=2 · H/GPU=64 · B1 rows/rank=876
- `TP2/CP2`: attention TP=1 · H/GPU=128 · B1 rows/rank=438

**Aggregation:** runs=2 (mean) · latency=p50(iteration-wise rank max) ·
speedup source=unrounded measurements

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

**Decode N/A:** TP2/CP1 B16/B32 · TP2/CP2 B32 · SMEM limit=227KiB · v8=PASS

**Repeatability:** max v8 run-to-run delta=1.30%

**Correctness:** H64=22/22 · H128=22/22 · total=44/44 · CUDA/NCCL errors=0

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

**Fixed execution:**

- query/key dimension: `d_qk = 576`
- value/output dimension: `d_v = 512`
- KV-cache page size: 64 tokens
- KV-cache storage: 656 bytes per token
- supported execution head extents: H64 and H128

| Hexec | Native launches | Output head slices |
| ---: | ---: | --- |
| 64 | 1 | `[0:64]` |
| 128 | 2 | `[0:64]`, `[64:128]` |

**H128 extras:** concat=0 · combine kernels=0 · D2D copies=0 · cross-head
reductions=0

**Batch/row mapping:** rows=`batch * new` · KV pools/process=1 · KV pool
length=`cached + new` · physical CP=SGLang padding + rank partition

## Appendix B: Benchmark topology

### Automatic execution mode

| Visible GPUs | Processes | NCCL workers | TP | Default CP | Mode |
| ---: | ---: | ---: | --- | --- | --- |
| 1 | 1 | 0 | 2 (H64) / 1 (H128), shape-only | 1 | Local H64/H128 |
| N (`N >= 2`) | N | N | N | 1 or N | Physical TP + CP endpoint |

**Selection:** `--gpus N` = first N visible devices · `--gpus 1` = force local
mode

**Physical constraints:** nodes=1 · `WORLD_SIZE=TP=N` · GPU=SM100 B200/B300

### Rank layout

**Rank mapping:**

```text
attention_tp      = TP / CP
Hlocal            = global_heads / attention_tp
attention_tp_rank = global_rank % attention_tp
cp_rank           = global_rank // attention_tp

Hexec = 64   when 1 <= Hlocal <= 64
        128  when 65 <= Hlocal <= 128
```

**CP padding/partition:**

```text
Tlogical  = batch * new
Tpadded   = ceil_align(ceil_align(Tlogical, attention_tp), 2 * CP)
rows      = Tpadded / CP
logical_r = Tlogical // CP + int(r < Tlogical % CP)
padding_r = rows - logical_r

kv_len         = cached + new
topk_effective = floor(min(topk, kv_len) / 64) * 64
```

**Padding example:** B=1 · new=875 · TP=8 · CP=8 · Tlogical=875 · Tpadded=880 ·
rows/rank=110 · rank 0-2=`logical 110 + pad 0` · rank 3-7=`logical 109 + pad 1`

**Head padding:** Hlocal=[1,64] -> Hexec=64 · Hlocal=[65,128] -> Hexec=128 ·
CSV fields=`local_heads, execution_heads`

### Topology status

- `CP=1` -> `production-pure-tp`
- `DP=1, CP=TP` -> `sglang-dsa-cp-endpoint`
- `1 < CP < TP` (example: `TP=8, CP=2/4`) -> `experimental-tp-cp`
- `CP > 8` -> `experimental-dsa-cp-over-8`

## Appendix C: Measurement scope

**Timed contribution:** input generation=0 · KV quantization=0 · sparse-index
construction=0 · process-group setup=0 · synchronization=0 · error
coordination=0 · result reduction=0

**L2 flush:** before start event · elapsed-time contribution=0

**Physical aggregation:** start-aligned ranks ·
latency=p50(iteration-wise rank max)

**Communication metric scope:** probe=1 · full layer=0 · SGLang end-to-end=0

| Metric | Timed boundary |
| --- | --- |
| `kernel_ms` | Rank-local attention kernel, reduced as synchronous rank-max p50. |
| `cp_latent_allgather_probe_ms` | NCCL all-gather of this layer's new/padded BF16 latent KV plus rank-major to token-major rearrangement. |
| `cp_allgather_probe_plus_kernel_ms` | The CP probe followed by the selected kernel on one CUDA stream. |
| `tp_o_proj_allreduce_probe_ms` | NCCL all-reduce of a correctly shaped post-`o_proj` BF16 payload. |

**Not included:** raw attention-output all-reduce · distributed output
reconstruction · projection GEMMs · indexer/top-k · MLP communication ·
cached-prefix gather · model-final output gather

**Control/timeouts:** error group=Gloo · NCCL timeout=120s · launcher
watchdog=7,200s · override=`MLA_PREFILL_BENCH_WATCHDOG_SECONDS`

## Appendix D: Advanced validation

**CUPTI audit — H=128 · case=edge · launch-only=1**

```bash
PYTHONPATH="$PWD" python3 -B \
  tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads 128 --case edge --launch-only --audit-launches \
  --trace-json /tmp/v8_h128_launches.json
```

**Expected audit:** H64 native launches=1 · H128 native launches=2 · CUDA
combine/copy/memset=0 · `aten::cat`=0

**Compute Sanitizer — tool=memcheck · H=128 · case=edge · launch-only=1**

```bash
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="$PWD" \
compute-sanitizer --tool memcheck --error-exitcode=1 \
  python3 -B tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads 128 --case edge --launch-only
```

## Appendix E: Advanced benchmark commands

**1 GPU — TP/CP shape simulation:**

```bash
python3 -B benchmark/v8/prefill_benchmark.py --tp
python3 -B benchmark/v8/prefill_benchmark.py --cp
python3 -B benchmark/v8/prefill_benchmark.py --tp --cp
```

**Matrix — TP=8 · CP=1/2/4/8 · B=1/8 · new=10,000 · cached=90,000 ·
top-k=512/1,024/2,048:**

```bash
python3 -B benchmark/v8/prefill_benchmark.py \
  --tp 8 --cp 1 2 4 8 \
  --batch 1 8 \
  --new 10000 --cached 90000 \
  --topk 512 1024 2048
```

**All simulated ranks — TP=8 · CP=8 · B=1 · new=875:**

```bash
python3 -B benchmark/v8/prefill_benchmark.py \
  --tp 8 --cp 8 --all-ranks --batch 1 --new 875
```

**Physical mode:** CP rank=process topology · `--cp-rank`=rejected ·
`--all-ranks`=rejected
