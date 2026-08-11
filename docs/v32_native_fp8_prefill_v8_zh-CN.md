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

benchmark 会根据可见 CUDA 设备数量自动选择真实、明确的执行模式：

```bash
PYTHONPATH="$PWD" python -B benchmark/v8/prefill_benchmark.py
```

- 只有一张可见 GPU 时，保留 BS `1、2、4、8、16、32` 的 rank-local
  H64/H128 shape microbenchmark。此时 TP/CP 只是 shape 元数据，不代表分布式
  测量。
- 有 `N >= 2` 张可见 GPU 时，自动为每张 GPU 启动一个 NCCL worker，并令
  `WORLD_SIZE=TP=N`。默认测量 `CP=1`（pure attention TP），以及当前 DP=1
  DSA prefill CP 强制采用的 `CP=N` topology。
- `--gpus N` 选择前 `N` 张可见 GPU；`--gpus 1` 强制使用本地单卡路径。

物理多卡模式由 NCCL 使用主机上的真实互联（例如 NVLink 或 PCIe），不会模拟
通信。

物理多卡模式中的 rank 坐标为：

```text
attention_tp      = TP / CP
attention_tp_rank = global_rank % attention_tp
cp_rank           = global_rank // attention_tp
Hlocal            = 128 / attention_tp
```

SGLang 会先 padding 展平后的 query-token 流，再进行 round-robin CP 切分。
benchmark 复现相同的物理工作量：

```text
Tlogical = batch * new
Tpadded  = ceil_align(ceil_align(Tlogical, attention_tp), 2 * CP)
Tlocal   = Tpadded / CP
```

例如 `batch=1,new=875,TP=CP=8` 会先 padding 到 880，因此每个 rank 都执行
110 rows。rank 0-2 有 110 个逻辑 rows；rank 3-7 有 109 个逻辑 rows 加一个
dummy row。

物理多卡延迟采用 `每次迭代先取最慢 rank，再取中位数`。对齐 barrier、L2
flush、输入/buffer setup 和结果归约均位于 CUDA event 区间之外。CP probe 的
materializing `transpose`/`reshape` rerange 则有意包含在 event 区间内。通信列的
边界有意保持清晰：

- `kernel_ms`：仅 kernel 的同步 critical path。
- `cp_latent_allgather_probe_ms`：真实 NCCL all-gather，只传输当前 layer
  新增/补齐的 BF16 latent KV `[Tlocal, 576]`，随后使用与 SGLang 相同的
  `view`/`transpose`/`reshape` 做 token-order rerange；不会传输已有 cached
  prefix。
- `cp_allgather_probe_plus_kernel_ms`：上述 probe 后顺序执行 kernel。gather 的
  probe 输出不会被 pack 到 kernel 使用的预构建 FP8 cache 中；它也不包含
  indexer/top-k、projection、MLP collective 或最终 hidden-state gather，因此
  不是 production end-to-end 延迟。
- `tp_o_proj_allreduce_probe_ms`：在数学上正确的 post-`o_proj` BF16 shape
  `[Tlocal, 7168]` 上执行真实 NCCL all-reduce。该 standalone benchmark 不包含
  projection GEMM，所以这只是通信 probe。

benchmark 绝不会 all-reduce raw head-sharded attention output；不同 TP rank
拥有不同 heads，这样归约在数学上是错误的。

该程序测量 topology 和 latency，不是分布式数值重建 oracle：它不会裁掉 dummy
token/head padding 后再 gather 全局逻辑 attention output。CSV 中的 payload bytes
是每个 rank 的本地输入大小，并非 NCCL wire bytes 估算。根据 runtime 设置，
post-`o_proj` production collective 可能是 all-reduce 或 reduce-scatter；这里仅
明确测量 all-reduce probe。

计时阶段的错误通过独立 Gloo control group 在 ranks 间同步。NCCL group 使用
120 秒 failure timeout；自动 launcher 另有两小时的全任务 watchdog，超时后会
终止所有 worker。只有确实需要更长运行时，才设置
`MLA_PREFILL_BENCH_WATCHDOG_SECONDS` 覆盖该值。

可以逐项增加其他比较 kernel：

```bash
python -B benchmark/v8/prefill_benchmark.py \
  --bf16 --fp8-dequant --trtllm --decode-chunked
```

单卡上，不带数值的 TP/CP 参数仍启用原来的 shape 扫描：

```bash
python -B benchmark/v8/prefill_benchmark.py --tp       # CP=1，TP 1、2、4、8
python -B benchmark/v8/prefill_benchmark.py --cp       # TP=8，CP 1、2、4、8
python -B benchmark/v8/prefill_benchmark.py --tp --cp  # 有效 TP x CP 矩阵
```

仍然可以显式指定 TP、CP、shape 和 rank。使用 `--dry-run` 可以在不分配
CUDA tensor 的情况下查看完整计划。

在物理 `N` 卡主机上，单独使用 `--tp` 表示 `TP=N,CP=1`；单独使用 `--cp`
会用真实 NCCL group 测量 `N` 的所有因数。当前 DP=1 DSA CP 强制 `CP=TP`，
而 `CP=1` 是 pure-TP 路径；中间因数会标记为 `experimental-tp-cp`，不会
伪装成当前 DSA 配置：

```bash
# 物理 TP8，测量 CP=1、2、4、8；CP2/CP4 是 topology what-if。
python -B benchmark/v8/prefill_benchmark.py --gpus 8 --cp

# 不需要八张本地 GPU，也可以只预览同一计划。
python -B benchmark/v8/prefill_benchmark.py --dry-run --gpus 8 --cp --csv ""
```

`--cp-rank` 与 `--all-ranks` 只用于单卡模拟。物理任务从 `RANK` 推导 CP rank，
并拒绝这两个参数。自动 launcher 仅支持单机，且所有参与 GPU 都必须是 SM100。
当前 SGLang 单机 DSA CP 仅支持 TP/CP 不超过 8；更大的 endpoint plan 会标记为
`experimental-dsa-cp-over-8`。SGLang 目前还会提示 DSA CP 只在 Hopper 上完成
验证，因此这里的 SM100 结果属于实验性的 kernel/communication 研究，而不是
production validation 声明。
