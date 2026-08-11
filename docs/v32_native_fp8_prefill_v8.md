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
git clone https://github.com/galbi-joa/flashmla-fp8-prefill-v3.2.git
cd flashmla-fp8-prefill-v3.2
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

## Run the prefill benchmark

The benchmark automatically selects a truthful execution mode from the number
of visible CUDA devices:

```bash
PYTHONPATH="$PWD" python -B benchmark/v8/prefill_benchmark.py
```

- One visible GPU keeps the rank-local H64/H128 shape microbenchmark for BS
  `1, 2, 4, 8, 16, 32`. Its TP/CP fields are shape metadata, not distributed
  measurements.
- `N >= 2` visible GPUs automatically launch one NCCL worker per GPU with
  `WORLD_SIZE=TP=N`. The default measures `CP=1` (pure attention TP) and the
  `CP=N` topology forced by current DP=1 DSA prefill CP.
- Use `--gpus N` to select `N` visible devices, or `--gpus 1` to force the
  local path.

NCCL uses the host's actual interconnect (for example NVLink or PCIe); no
communication is simulated in physical mode.

In physical mode, rank coordinates are:

```text
attention_tp      = TP / CP
attention_tp_rank = global_rank % attention_tp
cp_rank           = global_rank // attention_tp
Hlocal            = 128 / attention_tp
```

SGLang pads the flattened query-token stream before round-robin CP splitting.
The benchmark reproduces that physical workload:

```text
Tlogical = batch * new
Tpadded  = ceil_align(ceil_align(Tlogical, attention_tp), 2 * CP)
Tlocal   = Tpadded / CP
```

Thus `batch=1,new=875,TP=CP=8` executes 110 rows on every rank after padding
to 880. Ranks 0-2 own 110 logical rows; ranks 3-7 own 109 plus one dummy row.

Physical latency uses `median(iteration-wise max-rank CUDA latency)`. Alignment
barriers, L2 flushes, input/buffer setup, and latency reduction stay outside the
event interval. The CP probe deliberately includes the materializing
`transpose`/`reshape` rerange inside its interval. The communication columns are
deliberately scoped:

- `kernel_ms`: kernel-only critical path.
- `cp_latent_allgather_probe_ms`: real NCCL all-gather of only the current
  layer's new/padded BF16 latent KV `[Tlocal, 576]`, plus SGLang's
  `view`/`transpose`/`reshape` token-order rerange. It does not gather the
  cached prefix.
- `cp_allgather_probe_plus_kernel_ms`: that probe followed by the kernel. The
  gathered probe output is not packed into the kernel's prebuilt FP8 cache. It
  also excludes indexer/top-k, projections, MLP collectives, and final
  hidden-state gather, and is not production end-to-end latency.
- `tp_o_proj_allreduce_probe_ms`: real NCCL all-reduce on the correct
  post-`o_proj` BF16 shape `[Tlocal, 7168]`. The projection GEMMs are not in this
  standalone benchmark, so this is a communication probe only.

Raw head-sharded attention output is never all-reduced; different TP ranks own
different heads, so such a reduction would be mathematically invalid.

This runner measures topology and latency rather than distributed numerical
reconstruction: it does not trim dummy token/head padding and gather the logical
attention output into a global reference tensor. Payload-byte CSV fields are
local input sizes, not wire-byte estimates. The post-`o_proj` production
collective may be all-reduce or reduce-scatter depending on runtime settings;
this benchmark deliberately reports an all-reduce probe only.

Timed failures are synchronized through a separate Gloo control group. NCCL
groups use a 120-second failure timeout, and the automatic launcher terminates
the complete worker process group after a two-hour whole-job watchdog. Override
the latter with `MLA_PREFILL_BENCH_WATCHDOG_SECONDS` for an intentionally longer
run.

Add other comparison kernels individually:

```bash
python -B benchmark/v8/prefill_benchmark.py \
  --bf16 --fp8-dequant --trtllm --decode-chunked
```

On one GPU, value-free TP/CP flags retain the original shape sweeps:

```bash
python -B benchmark/v8/prefill_benchmark.py --tp       # TP 1,2,4,8 at CP=1
python -B benchmark/v8/prefill_benchmark.py --cp       # CP 1,2,4,8 at TP=8
python -B benchmark/v8/prefill_benchmark.py --tp --cp  # valid TP x CP matrix
```

Explicit TP, CP, shape, and rank values remain available. Use `--dry-run` to
inspect the complete plan without allocating CUDA tensors.

On a physical `N`-GPU host, a bare `--tp` means `TP=N,CP=1`; a bare `--cp`
measures every divisor of `N` with real NCCL groups. Current DP=1 DSA CP forces
`CP=TP`; `CP=1` is the pure-TP path. Intermediate divisors are marked
`experimental-tp-cp` rather than presented as current DSA configurations:

```bash
# Physical TP8 with CP=1,2,4,8. CP2/CP4 are topology what-if cases.
python -B benchmark/v8/prefill_benchmark.py --gpus 8 --cp

# Preview the same plan without needing eight local GPUs.
python -B benchmark/v8/prefill_benchmark.py --dry-run --gpus 8 --cp --csv ""
```

`--cp-rank` and `--all-ranks` are one-GPU simulation controls. Physical jobs
derive CP rank from `RANK` and reject those flags. The automatic launcher is
single-node and requires every participating GPU to be SM100. Current SGLang
single-node DSA CP is limited to TP/CP at most 8; larger endpoint plans are
marked `experimental-dsa-cp-over-8`. SGLang currently warns that DSA CP has
only been verified on Hopper, so SM100 results here are an experimental
kernel/communication study rather than a production-validation claim.
