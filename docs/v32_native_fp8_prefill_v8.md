# Native V3.2 FP8 Sparse Prefill v8

This repository adds an SM100 native-FP8 sparse-prefill path for the DeepSeek
V3.2 cache layout (`d_qk=576`, `d_v=512`, 656 bytes per KV token).

The CUDA kernel computes 64 query heads per launch. The public v8 bridge
accepts either H64 or H128:

- H64: one v8 CUDA-kernel launch.
- H128: one H128 `out`/`max_logits`/`lse` allocation, followed by two H64
  launches writing directly into disjoint head slices.

There is no `torch.cat`, output-combine kernel, D2D copy, or numerical
reduction between the two head groups. Heads are independent in attention, so
splitting the head dimension this way is mathematically exact. The consolidated
test checks the H128 results against two explicit H64 calls bit-for-bit.

## Requirements

- NVIDIA SM100 GPU (B200 or B300).
- CUDA 12.9 or newer. CUDA 13.0 was used for the B300 validation.
- A CUDA-enabled PyTorch build compatible with the installed toolkit.
- Python build packages such as `setuptools`, `wheel`, and `ninja`.
- `kernelkit` for the upstream numerical test helpers.

## Clone and compile

```bash
git clone https://github.com/galbi-joa/FlashMLA_v3.2_fp8_prefill.git
cd FlashMLA_v3.2_fp8_prefill
git submodule update --init --recursive
```

Create or activate an environment that already contains a CUDA-enabled
PyTorch build. A minimal preparation step is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja
# Install a CUDA-enabled PyTorch build matching this host, then install
# kernelkit if you intend to run the numerical tests.
```

Build only the SM100 targets and leave the extension in this checkout:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_MLA_DISABLE_SM90=1
export NVCC_THREADS=8
export MAX_JOBS=8

python setup.py build_ext --inplace -v
```

Verify that Python resolves the newly built in-tree extension, not an older
site-packages copy:

```bash
PYTHONPATH="$PWD" python - <<'PY'
import flash_mla.cuda as cuda
print(cuda.__file__)
assert hasattr(cuda, "v32_kernelv8_fp8_prefill_fwd")
PY
```

Alternatively, install the package after setting the same build variables:

```bash
python -m pip install -v --no-build-isolation .
```

## Run the consolidated test

H64 and H128 correctness, edge cases, and the H64 trace entry point:

```bash
PYTHONPATH="$PWD" python \
  tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads all --case all
```

Audit the launch topology with Torch Profiler/CUPTI. Tensor creation and the
explicit H64×2 correctness oracle are outside the profiled interval:

```bash
mkdir -p benchmark/v8/traces
PYTHONPATH="$PWD" python \
  tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads 128 --case edge --launch-only --audit-launches \
  --trace-json benchmark/v8/traces/h128.json
```

The audit requires exactly one `v8_kernel` activity for the H64 control and
exactly two for H128. Any CUDA combine/copy/memset activity or `aten::cat`
inside the API interval fails the test.

A minimal Compute Sanitizer launch check is:

```bash
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="$PWD" \
compute-sanitizer --tool memcheck --error-exitcode=1 \
  python tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads 128 --case edge --launch-only
```

The instrumented trace API intentionally supports H64 only. H128 trace calls
are rejected explicitly because one shared timestamp buffer cannot represent
two head groups without changing the trace schema.

## Run the v8 benchmark

The benchmark is standalone: it does not import SGLang or install dependencies.
It measures only the native v8 path. Run it from the repository root with the
in-tree extension first on `PYTHONPATH`.

### TP=1, CP=1, H128, multiple batch sizes

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
  --csv "benchmark/v8/tp1_h128_bs_sweep_${RUN_TAG}.csv"
```

Here `attention_tp=TP/CP=1`, so every rank owns all 128 query heads. Every
measured v8 API call therefore launches the H64 CUDA kernel twice.

### Fixed total TP with a CP sweep

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
  --csv "benchmark/v8/tp8_cp_sweep_${RUN_TAG}.csv"
```

For `(TP, CP) = (8, 1), (8, 2), (8, 4), (8, 8)`, the benchmark reports
`Hlocal = 16, 32, 64, 128` and `Hexec = 64, 64, 64, 128`. Query tokens are
distributed with quotient/remainder CP semantics; no remainder is silently
dropped. Use `--dry-run` to inspect the complete plan without allocating CUDA
tensors.

The microbenchmark models a DP=1 attention group. In the common SGLang DSA
configuration where `CP=TP`, `attention_tp=1`, each rank owns H128 and uses the
two-launch H128 bridge.
