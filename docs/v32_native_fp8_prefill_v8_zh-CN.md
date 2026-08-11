# 原生 V3.2 FP8 稀疏 Prefill v8

该仓库为 DeepSeek V3.2 KV Cache 布局加入了 SM100 原生 FP8 稀疏
prefill 路径：`d_qk=576`、`d_v=512`，每个 KV token 占 656 字节。

CUDA kernel 每次处理 64 个 query heads。公开的 v8 bridge 支持 H64 和
H128：

- H64：启动一次 v8 CUDA kernel。
- H128：只分配一次 H128 的 `out`、`max_logits` 和 `lse`，然后启动两次
  H64 kernel，分别直接写入互不重叠的 head 区间。

两个 head group 之间没有 `torch.cat`、合并 kernel、D2D copy 或数值归约。
Attention 的不同 heads 在数学上彼此独立，因此按 head 维度拆分是严格正确的。
统一测试会把 H128 的三个结果与两次显式 H64 调用逐 bit 比较。

## 环境要求

- NVIDIA SM100 GPU（B200 或 B300）。
- CUDA 12.9 或更高版本；B300 验证使用 CUDA 13.0。
- 与本机 CUDA 工具链匹配的 CUDA 版 PyTorch。
- Python 构建依赖，例如 `setuptools`、`wheel`、`ninja`。
- 运行上游数值测试时需要 `kernelkit`。

## 克隆与编译

```bash
git clone https://github.com/galbi-joa/flashmla-fp8-prefill-v3.2.git
cd flashmla-fp8-prefill-v3.2
git submodule update --init --recursive
```

创建或激活已经安装 CUDA 版 PyTorch 的环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja
# 然后安装与本机匹配的 CUDA 版 PyTorch；若要运行数值测试，再安装
# kernelkit。
```

只编译 SM100，并把扩展保留在当前源码目录中：

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_MLA_DISABLE_SM90=1
export NVCC_THREADS=8
export MAX_JOBS=8

python setup.py build_ext --inplace -v
```

确认 Python 加载的是刚刚在源码目录中编译的扩展，而不是
site-packages 中的旧版本：

```bash
PYTHONPATH="$PWD" python - <<'PY'
import flash_mla.cuda as cuda
print(cuda.__file__)
assert hasattr(cuda, "v32_kernelv8_fp8_prefill_fwd")
PY
```

也可以在设置相同编译环境变量后安装：

```bash
python -m pip install -v --no-build-isolation .
```

## 运行统一测试

运行 H64/H128 正确性、边界条件以及 H64 trace 接口测试：

```bash
PYTHONPATH="$PWD" python \
  tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads all --case all
```

使用 Torch Profiler/CUPTI 检查 kernel 启动拓扑。Tensor 构造和显式
H64×2 的正确性 oracle 都不在 profiler 区间内：

```bash
mkdir -p benchmark/v8/traces
PYTHONPATH="$PWD" python \
  tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads 128 --case edge --launch-only --audit-launches \
  --trace-json benchmark/v8/traces/h128.json
```

该 audit 要求 H64 control 恰好出现一个 `v8_kernel` activity，H128
恰好出现两个。API 区间内只要出现 CUDA 合并/copy/memset activity 或
`aten::cat`，测试就会失败。

最小的 Compute Sanitizer 检查命令：

```bash
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="$PWD" \
compute-sanitizer --tool memcheck --error-exitcode=1 \
  python tests/test_flash_mla_sparse_prefill_fp8_v32_v8.py \
  --heads 128 --case edge --launch-only
```

带时间戳的 trace API 目前有意只支持 H64。H128 trace 会被明确拒绝，
因为一个共享 timestamp buffer 无法在不修改 trace schema 的情况下表示两个
head group。

## 运行 prefill benchmark

默认命令会对 BS `1、2、4、8、16、32` 同时测量 H64 和 H128，并比较
FlashMLA FP8 sparse decode 与原生 prefill 路径：

```bash
PYTHONPATH="$PWD" python -B benchmark/v8/prefill_benchmark.py
```

可以逐项增加其他比较 kernel：

```bash
python -B benchmark/v8/prefill_benchmark.py \
  --bf16 --fp8-dequant --trtllm --decode-chunked
```

不带数值的 TP/CP 参数会启用内置扫描：

```bash
python -B benchmark/v8/prefill_benchmark.py --tp       # CP=1，TP 1、2、4、8
python -B benchmark/v8/prefill_benchmark.py --cp       # TP=8，CP 1、2、4、8
python -B benchmark/v8/prefill_benchmark.py --tp --cp  # 有效 TP x CP 矩阵
```

仍然可以显式指定 TP、CP、shape 和 rank。使用 `--dry-run` 可以在不分配
CUDA tensor 的情况下查看完整计划。

该 microbenchmark 模拟 DP=1 的 attention group。在常见的 SGLang DSA
配置中 `CP=TP`，因此 `attention_tp=1`，每个 rank 都拥有 H128，并走两次
H64 launch 的 H128 bridge。
