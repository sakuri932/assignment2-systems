# CS336 Assignment 2 (systems): 系统与并行化

**版本 26.1.1 | CS336 课程团队 | Spring 2026**

---

## 1 作业概述

本作业将带你深入体验提升单 GPU 训练速度以及将训练扩展到多 GPU 的方法。

**你将实现以下内容：**

1. 基准测试与性能分析框架
2. 激活检查点（Activation Checkpointing）
3. FlashAttention-2 Triton 核函数
4. 分布式数据并行训练（DDP）
5. 优化器状态分片（ZeRO Stage 1）
6. 全分片数据并行训练（FSDP）

**代码结构：**

- `cs336-basics/`：包含 Assignment 1 的参考实现，含 `cs336_basics` 包
- `/`：cs336-systems 根目录，含空的 `cs336_systems` 模块
- `tests/*.py`：所有必须通过的测试，通过 `tests/adapters.py` 中的适配器调用你的实现
- `README.md`：目录结构说明与环境配置指南

**提交内容：**

- `writeup.pdf`：所有书面问题的答案（需排版）
- `code.zip`：所有你编写的代码（运行 `test_and_make_submission.sh` 生成）

---

## 2 性能分析与基准测试

在第一部分，我们将探索如何优化 Transformer 模型，最大化利用 GPU。先进行性能剖析（profiling）以了解前向和反向传播的时间与内存消耗，再用自定义 GPU 核函数优化自注意力操作。

### 2.1 性能剖析

在实施任何优化之前，应先对程序进行剖析，了解资源（时间、内存）的消耗分布。否则可能优化了影响不大的部分，导致端到端指标没有明显改善。

我们将实现三条性能评估路径：

1. 用 Python 标准库进行简单的端到端基准测试
2. 用 NVIDIA Nsight Systems 工具进行计算剖析
3. 内存剖析

#### 2.1.1 配置：导入 Transformer 模型

先确认能从 Assignment 1 中导入模型。根目录的 `pyproject.toml` 已经指向 `./cs336-basics` 包，运行 `uv run python` 即可使用。如需使用自己的实现，修改 `pyproject.toml` 中的路径即可。

#### 2.1.2 模型规格

本作业统一使用词表大小 10,000、batch size 4，上下文长度依配置而定。除排行榜外，使用以下模型配置（均基于 GPT-2 超参）：

| 尺寸   | d_model | d_ff  | num_layers | num_heads |
|--------|---------|-------|------------|-----------|
| small  | 768     | 3072  | 12         | 12        |
| medium | 1024    | 4096  | 24         | 16        |
| large  | 1280    | 5120  | 36         | 20        |
| xl     | 2560    | 10240 | 32         | 32        |
| 10B    | 4608    | 12288 | 50         | 36        |

**默认上下文长度为 512，除非另有说明。**

#### 2.1.3 端到端基准测试

> **问题（benchmarking_script）：基准测试脚本（4 分）**
>
> (a) 编写脚本，对模型的前向传播、反向传播及优化器步骤进行基本的端到端基准测试：
>
> - 根据超参初始化模型
> - 生成随机 batch 数据
> - 执行 $w$ 次热身步骤后，计时 $n$ 次执行（支持仅前向、前向+反向、完整训练步三种模式）
> - 使用 `timeit.default_timer()` 计时
> - 每步后调用 `torch.cuda.synchronize()`
>
> **交付物：** 一个支持以上功能的基准测试脚本。
>
> (b) 对 2.1.2 中各模型尺寸计时，使用 5 次热身 + 10 次测量，报告均值和标准差。
>
> **交付物：** 1-2 句时间结果说明。
>
> (c) 对比不进行热身的情况，解释差异原因。
>
> **交付物：** 2-3 句回答。

**重要提示：CUDA 调用是异步的。** `torch.matmul` 调用返回时，GPU 上的矩阵乘法可能还未完成。必须调用 `torch.cuda.synchronize()` 等待所有 CUDA 核函数完成，才能获得准确的计时。

#### 2.1.4 Nsight Systems 性能剖析器

端到端基准测试无法告诉我们每个组件消耗了多少时间。Nsight Systems（`nsys`）可以分析 CUDA 核函数的执行情况。

基本用法：

```bash
uv run nsys profile -- python benchmark.py
```

完整用法：

```bash
uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=functions-trace,autograd-shapes-nvtx \
  --cudabacktrace=all --python-backtrace=cuda \
  --gpu-metrics-devices=0 -- python benchmark.py
```

也可以用 NVTX 标注代码：

```python
import torch.cuda.nvtx as nvtx

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(...):
    with nvtx.range("computing attention scores"):
        ...
    with nvtx.range("computing softmax"):
        ...
    with nvtx.range("final matmul"):
        ...
```

> **问题（nsys_profile）：Nsight Systems 性能剖析（5 分）**
>
> 选择 2 种模型尺寸、3 种 2 的幂次方上下文长度（大于 128），对前向+反向+优化器步进行剖析，回答：
>
> (a) 前向传播总耗时是否与 Python 计时一致？
> (b) 前向传播中累计 GPU 时间最多的 CUDA 核函数是哪个？该核函数被调用了多少次？
> (c) 除矩阵乘法外，还有哪些核函数占用了可观的运行时？
> (d) 对比推理（仅前向）和完整训练步（前向+反向+优化器），矩阵乘法时间占比如何变化？
> (e) 在自注意力层内，softmax 与矩阵乘法的运行时之比，与其 FLOPs 之比相比如何？

#### 2.1.5 混合精度

现代 NVIDIA GPU 含有 Tensor Core，可加速低精度矩阵乘法。B200 的 FP32 峰值为 80 TFLOPS，而 FP16/BF16 峰值高达 2500 TFLOPS。

PyTorch 的 `torch.autocast` 上下文管理器可实现混合精度：

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    y = model(x)
```

> **问题（mixed_precision_accumulation）：混合精度累加（1 分）**
>
> 运行以下代码并评论精度：
>
> ```python
> s = torch.tensor(0, dtype=torch.float32)
> for i in range(1000):
>     s += torch.tensor(0.01, dtype=torch.float32)
> print(s)
>
> s = torch.tensor(0, dtype=torch.float16)
> for i in range(1000):
>     s += torch.tensor(0.01, dtype=torch.float16)
> print(s)
>
> s = torch.tensor(0, dtype=torch.float32)
> for i in range(1000):
>     s += torch.tensor(0.01, dtype=torch.float16)
> print(s)
>
> s = torch.tensor(0, dtype=torch.float32)
> for i in range(1000):
>     x = torch.tensor(0.01, dtype=torch.float16)
>     s += x.type(torch.float32)
> print(s)
> ```
>
> **交付物：** 2-3 句注释。

> **问题（benchmarking_mixed_precision）：混合精度基准测试（2 分）**
>
> (a) 对以下 ToyModel 使用 FP16 自动混合精度，各组件的数据类型是什么？
>   - 自动混合精度上下文中的模型参数
>   - fc1 输出
>   - LayerNorm 输出
>   - 模型 logits
>   - 损失
>   - 梯度
>
> (b) 如果用 BF16 代替 FP16，LayerNorm 是否仍需特殊处理？
>
> (c) 修改基准测试脚本，支持 BF16 混合精度，对比各模型尺寸的前向/反向耗时。

#### 2.1.6 内存剖析

用 PyTorch 内存剖析器分析内存分配：

```python
torch.cuda.memory._record_memory_history(max_entries=1000000)
# ... 你想剖析的代码 ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

生成的 pickle 文件可上传至 [pytorch.org/memory_viz](https://pytorch.org/memory_viz) 查看。

> **问题（memory_profiling）：内存剖析（4 分）**
>
> 对 xl 模型在上下文长度 128 和 2048 下进行完整训练步的内存剖析，回答：
>
> (a) 内存时间线图（推理 vs 训练步）
> (b) 各上下文长度的峰值内存
> (c) 使用混合精度对内存的影响
> (d) 残差流激活张量的大小（单精度，MiB）
> (e) 最大分配来自哪里？
> (f) 单个 TransformerBlock 为反向传播保存了多少内存？

---

## 3 单 GPU 内存优化

### 3.1 自动求导中间张量（Autograd Residuals）

为了执行反向传播，需要保存前向传播中产生的激活值（称为 "residuals" 或 "saved tensors"）。可以用 `torch.autograd.graph.saved_tensors_hooks` 观察这些保存行为：

```python
def pack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    print(f"Loading residual: {shape=}")
    return t

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()
```

#### 3.1.1 算子融合

使用 `torch.compile` 可以自动融合算子，显著减少保存张量的数量：

```python
ln = torch.compile(RMSNorm(x.shape[-1]))
```

融合后，RMSNorm 只需保存 1 个全尺寸激活（输入 x），而非融合前的 5 个。

### 3.2 激活检查点（Activation Checkpointing）

对于 xl 模型（batch=4, seq=2048），单个 TransformerBlock 就需要保存约 3.6 GiB 的中间张量用于反向传播；32 层合计约 114 GiB，远超 GPU 内存。

**解决方案：梯度检查点（gradient checkpointing）**

`torch.utils.checkpoint.checkpoint` 修改函数行为：

- **前向传播时**：只保存函数的输入，抑制中间张量的保存
- **反向传播时**：在需要时重新执行前向传播（recomputation），然后正常完成反向传播

```python
from torch.utils.checkpoint import checkpoint

def two_blocks(x):
    x = block(x)
    x = block(x)
    return x

def four_blocks_checkpoint(x):
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x
```

使用检查点后，4 个 Block 只需保存约 160 MiB（相比未使用时的 14605 MiB）。

**权衡：** 检查点越小（粒度越细），内存越省，但重计算开销越大。可以**递归嵌套** `checkpoint` 以进一步减少峰值内存。

> **问题（gradient_checkpointing）：内存最优梯度检查点（4 分）**
>
> 对于 N 个 TransformerBlock 顺序堆叠的 Transformer：
>
> (a) 在不考虑计算开销的情况下，什么检查点策略能最小化峰值激活内存？给出代码草图、渐近峰值内存和计算复杂度（关于 N）。
>
> (b) 对 xl 模型（batch=4, seq=2048），在只允许一次重计算（不嵌套）的情况下，最佳检查点策略是什么？通过剖析验证你的预测，并对比相邻大小的检查点块。

---

## 4 GPU 核函数

### 4.1 用 FlashAttention-2 优化注意力

#### 4.1.1 PyTorch 注意力基准测试

标准注意力实现需要保存形状为 `seq_len × seq_len` 的注意力分数矩阵，内存消耗随序列长度平方增长。FlashAttention-2 通过分块计算避免显式存储该矩阵。

> **问题（pytorch_attention）：PyTorch 注意力基准测试（2 分）**
>
> 编写脚本：固定 batch size=8，不使用多头（去掉 head 维度），遍历以下笛卡尔积：
> - head embedding 维度 d_model：[16, 32, 64, 128]
> - 序列长度：[256, 1024, 4096, 8192, 16384]
>
> 对每种配置：计时 100 次前向传播、测量反向传播前的内存使用量、计时 100 次反向传播。
>
> **交付物：** 时间表格、OOM 分析及 1-2 段说明。

### 4.2 JIT 编译注意力基准测试

PyTorch 2.0+ 的 `torch.compile` 可自动生成融合的 Triton 核函数：

```python
compiled_layer = torch.compile(layer)
```

> **问题（torch_compile）：Torch Compile（2 分）**
>
> (a) 在注意力基准测试中加入编译版本，与未编译版本对比。
>
> (b) 对整个 Transformer 模型使用 `torch.compile`，对比前向/反向+优化器步的性能变化。

#### 4.2.1 示例：加权求和（Weighted Sum）

在实现 FlashAttention Triton 核函数之前，先通过加权求和示例了解 Triton 与 PyTorch 的互操作。

给定输入矩阵 $X$ 和列权重向量 $w$，计算 $y_i = \sum_j w_j \cdot X_{ij}$。

**Triton 核函数关键概念：**

- `tl.program_id(0)`：获取当前程序实例的索引（类似 CUDA 中的 blockIdx）
- `tl.make_block_ptr`：创建块指针，管理对内存区域的访问
- `tl.load` / `tl.store`：从全局内存加载/存储数据
- `tl.constexpr`：编译时常量，用于块大小等

**Triton 加权求和前向核函数：**

```python
@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr, output_ptr,
    x_stride_row, x_stride_dim,
    weight_stride_dim, output_stride_row,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    # ... 使用块指针加载数据、计算、存储结果 ...
```

**PyTorch autograd.Function 包装：**

```python
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # 启动 Triton 核函数
        weighted_sum_fwd[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](...)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        # 调用反向核函数
        ...
```

#### 4.2.2 FlashAttention-2 前向传播

**标准注意力的低效性：**

标准注意力的前向传播：
$$S = QK^\top / \sqrt{d}, \quad P_{ij} = \text{softmax}_j(S)_{ij}, \quad O = PV$$

反向传播需要 $P$（形状 `batch×seq×seq`），内存随序列长度平方增长。

**FlashAttention 的三大技术：**

1. **分块（Tiling）**：不需要完整行就能增量计算 softmax（在线 softmax 技术）
2. **重计算（Recomputation）**：前向传播只保存 $L$（logsumexp）和 $O$，反向传播时重计算 $P$
3. **算子融合（Operator Fusion）**：单个 Triton 核函数完成所有操作，减少 HBM↔SRAM 数据传输

**关键公式：保存的 logsumexp**

$$L_i = \log\left(\sum_j \exp(S_{ij})\right)$$

**反向传播利用 $L$ 简化计算：**

$$P_{ij} = \exp(S_{ij} - L_i), \quad D = \text{rowsum}(dO \circ O)$$

$$dS_{ij} = P_{ij} \circ (dP_{ij} - D_i), \quad dQ = dSK/\sqrt{d}, \quad dK = dS^\top Q/\sqrt{d}$$

**Algorithm 1：FlashAttention-2 前向传播**

1. 将 $Q$ 分成 $T_q$ 个大小为 $B_q \times d$ 的块 $Q_1, \ldots, Q_{T_q}$
2. 将 $K, V$ 分成 $T_k$ 个大小为 $B_k \times d$ 的块
3. 对每个查询块 $i$：
   - 初始化 $O_i^{(0)} = 0$，$l_i^{(0)} = 0$，$m_i^{(0)} = -\infty$
   - 对每个键值块 $j$：
     - 计算 $S_i^{(j)} = Q_i (K^{(j)})^\top / \sqrt{d}$
     - 更新运行最大值 $m_i^{(j)} = \max(m_i^{(j-1)}, \text{rowmax}(S_i^{(j)}))$
     - 计算 $\tilde{P}_i^{(j)} = \exp(S_i^{(j)} - m_i^{(j)})$
     - 更新 $l_i^{(j)} = \exp(m_i^{(j-1)} - m_i^{(j)}) l_i^{(j-1)} + \text{rowsum}(\tilde{P}_i^{(j)})$
     - 更新 $O_i^{(j)} = \text{diag}(\exp(m_i^{(j-1)} - m_i^{(j)})) O_i^{(j-1)} + \tilde{P}_i^{(j)} V^{(j)}$
   - 归一化：$O_i = \text{diag}(l_i^{(T_k)})^{-1} O_i^{(T_k)}$
   - 计算：$L_i = m_i^{(T_k)} + \log(l_i^{(T_k)})$
   - 将 $O_i$、$L_i$ 写入全局内存

> **问题（flash_forward）：FlashAttention-2 前向传播（15 分）**
>
> (a) 用纯 PyTorch（无 Triton）实现 `autograd.Function`，实现 FlashAttention-2 前向传播。
>   - 接受 $Q, K, V$ 及 `is_causal` 标志，返回输出 $O$ 和 logsumexp $L$
>   - 保存 $L, Q, K, V, O$ 供反向传播使用
>   - 瓦片大小至少为 16×16
>
>   运行测试：`uv run pytest -k test_flash_forward_pass_pytorch`
>
> (b) 编写 Triton 核函数，实现 FlashAttention-2 前向传播。
>   - 启动网格：`(T_q, batch_size)`，每个程序实例处理一个查询块
>   - 核函数只有一个循环（遍历键值块）
>   - 使用给定的函数签名（含 `D: tl.constexpr`、`IS_CAUSAL: tl.constexpr`）
>   - 片上缓冲区（$O_i, l, m$）使用 `tl.float32`
>
>   运行测试：`uv run pytest -k test_flash_forward_pass_triton`
>
> (c) 添加因果掩码支持（`is_causal: tl.constexpr`），对超出因果范围的位置的注意力分数加 -1e6。

**Triton 编程技巧：**

- `tl.device_print` 用于调试
- 矩阵乘法用 `tl.dot`
- 推进块指针用 `block_ptr = block_ptr.advance(...)`
- 上板缓冲区精度保持 `tl.float32`
- 在乘以 $V$ 之前将 $\tilde{P}$ 转换为 $V$ 的数据类型

> **问题（flash_backward）：FlashAttention-2 反向传播（5 分）**
>
> 用 PyTorch（不用 Triton）和 `torch.compile` 实现反向传播。接受 $Q, K, V, O, dO, L$，返回 $dQ, dK, dV$。记得计算 $D$ 向量。
>
> 运行测试：`uv run pytest -k test_flash_backward`

> **问题（flash_benchmarking）：FlashAttention-2 基准测试（5 分）**
>
> (a) 用 `triton.testing.do_bench` 对比 FlashAttention-2 与标准 PyTorch 注意力的性能。
>   - 固定 batch size=1，启用因果掩码
>   - 遍历序列长度（$2^7$ 到 $2^{16}$）、embedding 维度（$2^4$ 到 $2^7$）、精度（bfloat16 和 float32）
>   - 报告前向、反向及端到端延迟对比表

#### 4.2.3 可选：Triton 反向传播

**Algorithm 2：分块 FlashAttention-2 反向传播**

1. 预计算 $D = \text{rowsum}(dO \circ O) \in \mathbb{R}^{N_q}$
2. 对每个键值块 $j$：
   - 初始化 $dK^{(j)} = dV^{(j)} = 0$
   - 对每个查询块 $i$：
     - 计算 $P_i^{(j)} = \exp(S_i^{(j)} - L_i)$
     - $dV^{(j)} += (P_i^{(j)})^\top dO_i$
     - $dP_i^{(j)} = dO_i (V^{(j)})^\top$
     - $dS_i^{(j)} = P_i^{(j)} \circ (dP_i^{(j)} - D_i) / \sqrt{d}$
     - 原子更新 $dQ_i += dS_i^{(j)} K^{(j)}$
     - $dK^{(j)} += (dS_i^{(j)})^\top Q_i$
   - 将 $dK^{(j)}, dV^{(j)}$ 写入全局内存

---

## 5 分布式数据并行训练（DDP）

### 5.1 单节点分布式通信

PyTorch 提供 `torch.distributed`（简称 `dist`）包用于分布式通信：

```python
import os
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} result: {data}")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size,), nprocs=world_size, join=True)
```

**关键术语：**

| 术语 | 含义 |
|------|------|
| node | 网络中的一台机器 |
| worker | 参与训练的进程 |
| world size | 进程组中的进程总数 |
| global rank | worker 在进程组中的唯一 ID（0 到 world_size-1） |
| local world size | 单节点上的 worker 数量 |
| local rank | 节点内 worker 的唯一 ID |

**通信后端：** GPU 训练使用 NCCL，CPU 调试使用 Gloo。

#### 5.1.1 分布式应用基准测试最佳实践

- 同一台机器上进行基准测试以便对照
- 计时前至少进行 5 次热身
- 使用 `torch.cuda.synchronize()` 等待 CUDA 操作完成
- 用 `dist.all_gather_object` 从所有 rank 收集结果
- 本地用 Gloo 调试，GPU 上用 NCCL 基准测试

> **问题（distributed_communication_single_node）：分布式通信（5 分）**
>
> 编写脚本，基准测试单节点多进程的 all-reduce 运行时间。测试以下设置：
> - 数据大小：1MB、10MB、100MB、1GB
> - GPU/进程数：2、4、6
>
> **交付物：** 对比图表及 2-3 句说明。

### 5.2 朴素 DDP 实现

DDP（Distributed Data Parallel）将 batch 分片到多个 GPU：

1. 从 rank 0 广播模型参数到所有 rank（确保初始参数相同）
2. 各 rank 用本地参数计算各自数据分片的前向和反向传播
3. **All-reduce** 梯度，使每个 rank 持有所有样本的平均梯度
4. 各 rank 用相同的平均梯度进行优化器步，参数保持同步

> **问题（naive_ddp）：朴素 DDP（5 分）**
>
> 编写脚本实现朴素 DDP：反向传播后 all-reduce 各参数梯度。用小模型在随机数据上训练，验证权重与单进程训练结果匹配。

> **问题（naive_ddp_benchmarking）：朴素 DDP 基准测试（3 分）**
>
> 对 xl 模型（1 节点 × 2 GPU）基准测试每个训练步的总时间及通信梯度所占时间比例。

### 5.3 改进朴素 DDP

朴素 DDP 的两个主要限制：
1. 每个参数单独 all-reduce，通信开销大
2. 等待整个反向传播完成后才通信，无法与计算重叠

#### 5.3.1 减少通信调用次数

将所有梯度拼接成一个大张量后进行单次 all-reduce，可减少通信调用开销。

> **问题（minimal_ddp_flat_benchmarking）：扁平化梯度 DDP 基准测试（2 分）**
>
> 修改朴素 DDP 实现，使用单次 all-reduce 传输拼接的梯度张量，对比两种实现的性能。

#### 5.3.2 将反向传播计算与梯度通信重叠

反向传播是增量计算的（从输出层向输入层），某个参数的梯度一旦就绪就可以立即通信，而不必等待所有参数的梯度都计算完毕。

**关键接口：**

- `register_post_accumulate_grad_hook`：在参数梯度积累完成后自动调用函数
- 异步通信：`dist.all_reduce(tensor, async_op=True)` 返回 handle，稍后调用 `handle.wait()` 等待完成

```python
# 异步通信示例
handles = []
for tensor in tensors:
    handle = dist.all_reduce(tensor, async_op=True)
    handles.append(handle)
# ... 执行不依赖 all_reduce 结果的其他操作 ...
for handle in handles:
    handle.wait()
```

> **问题（ddp_overlap_individual_parameters）：DDP 重叠通信（5 分）**
>
> 实现 DDP 容器类，接口如下：
>
> - `__init__(self, module: nn.Module)`：广播 rank 0 的参数，注册梯度 hook
> - `forward(self, *inputs, **kwargs)`：调用被包装模块的前向传播
> - `finish_gradient_synchronization(self)`：等待所有异步通信完成
>
> 使用方式：
>
> ```python
> model = ToyModel().to(device)
> ddp_model = DDP(model)
> for _ in range(train_steps):
>     logits = ddp_model(x)
>     loss = loss_fn(logits, y)
>     loss.backward()
>     ddp_model.finish_gradient_synchronization()
>     optimizer.step()
> ```
>
> 实现适配器 `adapters.get_ddp`，运行测试：`uv run pytest tests/test_ddp_individual_parameters.py`

> **问题（ddp_overlap_individual_parameters_benchmarking）：重叠 DDP 基准测试（1 分）**
>
> (a) 基准测试重叠实现，与之前的 DDP 变体对比（1 节点，2 GPU，xl 模型）。
>
> (b) 用 Nsight profiler 对比初始 DDP 与重叠 DDP 的 trace，截图展示计算与通信的重叠情况。

---

## 6 优化器状态分片（ZeRO Stage 1）

DDP 需要每个 rank 保存完整的模型参数和优化器状态（如 AdamW 的 m、v 各一份，为参数量的 2 倍）。

优化器状态分片方案：
- 每个 rank 只维护约 `1/world_size` 的参数的优化器状态
- 优化器步骤只更新本 rank 负责的那部分参数
- 更新后，每个 rank 将自己更新的参数广播给其他 rank

> **问题（optimizer_state_sharding）：优化器状态分片（15 分）**
>
> 实现分片优化器类，接口如下：
>
> - `__init__(self, params, optimizer_cls, **kwargs)`：将参数分片到各 rank，每个 rank 只创建自己分片的本地优化器
> - `step(self, closure, **kwargs)`：本地优化器步骤后，广播各 rank 更新的参数
> - `add_param_group(self, param_group)`：将参数组添加到分片优化器
>
> 实现适配器 `adapters.get_sharded_optimizer`，运行测试：`uv run pytest tests/test_sharded_optimizer.py`

> **问题（optimizer_state_sharding_accounting）：优化器状态分片分析（5 分）**
>
> (a) 测量有无优化器状态分片时的峰值内存（模型初始化后、优化器步骤前后）。
>
> (b) 分片对训练速度的影响（每步时间对比）。
>
> (c) 与 ZeRO 论文中的 ZeRO Stage 1（$P_{os}$）相比，有何不同？

---

## 7 全分片数据并行训练（FSDP）

DDP 和优化器状态分片后，模型权重仍在每个 GPU 上都有完整副本。FSDP 进一步对权重本身进行分片：

- 每个 GPU 只存储每个权重张量的一个分片
- 在前向/反向传播前，通过 **all-gather** 从其他 GPU 收集分片，重建完整权重
- 梯度计算完成后，通过 **reduce-scatter** 只保留本 rank 对应分片的梯度

**分片策略：** 只对 Linear 和 Embedding 层分片；RMSNorm 等小层不分片（称为"replicated"参数）。

**混合精度：** 主权重（master weights）保存为 FP32，但可以转换为低精度后再通信和计算，节省带宽。

> **问题（fsdp）：全分片数据并行（15 分）**
>
> 实现 FSDP 容器类，接口如下：
>
> - `__init__(self, module: nn.Module, compute_dtype=None)`：对模型中所有 Linear/Embedding 层的权重进行 1D 扁平化分片
> - `forward(self, *inputs, **kwargs)`：前向传播前 all-gather 完整权重（根据 compute_dtype 转换），执行前向传播
> - `finish_gradient_synchronization(self)`：
>   - 分片参数：all-reduce 完整梯度 + 除以 world_size，保留本 rank 分片的梯度
>   - 复制参数：all-reduce 梯度 + 除以 world_size
>
> **分片细节：** 将参数展平为 1D，每个 rank 拥有 `ceil(total/world_size)` 个元素；最后一个 rank 可能有 padding。
>
> 实现适配器 `adapters.get_fsdp`，运行测试：`uv run pytest tests/test_fsdp.py`

> **问题（fsdp_accounting）：FSDP 内存分析（5 分）**
>
> (a) 相比 DDP + 优化器状态分片，FSDP 预计节省多少峰值内存？
>
> (b) 对 xl 模型（2 GPU）进行 profiling，all-gather 是否在前向传播开始前完成？

---

## 8 并行策略分析

常见并行策略：

| 策略 | 说明 |
|------|------|
| **数据并行（DP）** | 将 batch 分片到多设备，梯度取平均 |
| **全分片数据并行（FSDP）** | 在 DP 基础上，将优化器状态、梯度和权重也分片 |
| **张量并行（TP）** | 按输入或输出维度对权重矩阵分片 |
| **流水线并行（PP）** | 将模型按层分割到不同设备 |
| **专家并行（EP）** | 将 MoE 中的专家分配到不同设备 |

本节分析单个 FFN 层的并行化，FFN 前向传播为：

$$x_1 = xW_1, \quad x_2 = xW_2, \quad z = f(x_1) * x_2, \quad y = zW_3$$

其中 $x \in \mathbb{R}^{B \times D}$，$W_1, W_2 \in \mathbb{R}^{D \times D_{FF}}$，$W_3 \in \mathbb{R}^{D_{FF} \times D}$。

### 8.1 通信原语

**Ring All-gather：** 每个设备初始有大小 $S/N$ 的分块，最终获得完整张量（大小 $S$）。时间：$(N-1)/N \cdot S/W$ 秒。

**Ring Reduce-scatter：** 每个设备初始有完整张量 $x^{(i)}$，最终每个设备获得对应分块的归约结果。时间：$(N-1)/N \cdot S/W$ 秒。

**Ring All-reduce = Reduce-scatter + All-gather**：时间：$2(N-1)/N \cdot S/W$ 秒。

> **问题（alternate_ring_all_reduce）：替代环形 all-reduce（1 分）**
>
> 以下算法（直接在环上传递并累加，不拆分为 scatter+gather）需要多长时间？给出关于 $S, N, W$ 的答案及一句理由。

### 8.2 数据并行分析

DP 中有 $N_{DP}$ 个设备，输入 $x$ 按 batch 分片为 $x^{(i)}$（大小 $B/N_{DP} \times D$）。

前向传播无需 collective 操作；反向传播后需要 all-reduce $dW_1, dW_2, dW_3$。

> **问题（data_parallel_calcs）：数据并行计算（3 分）**
>
> 设备速度 $C$（FLOP/s），egress 带宽 $W$（字节/秒），权重和激活均为 FP16（2 字节）：
>
> (a) DP 反向传播需要多少 FLOPs？（以 $B, D, D_{FF}, N_{DP}$ 表示）
>
> (b) DP 反向传播需要多少通信时间？（以 $D, D_{FF}, N_{DP}, W$ 的子集表示）
>
> (c) $N_{DP}$ 增大到多少会开始受通信瓶颈限制？（给出关于 $B, D, D_{FF}, C, W$ 的不等式）

### 8.3 FSDP 分析

FSDP 中权重分片大小为 $D \cdot D_{FF} / N_{FSDP}$。前向传播需要 all-gather 三个权重矩阵，反向传播后需要 reduce-scatter。

> **问题（fsdp_calcs）：FSDP 计算（3 分）**
>
> (a) FSDP 前向/反向传播各需多少 FLOPs？
>
> (b) FSDP 前向/反向传播各需多少通信时间？
>
> (c) $N_{FSDP}$ 增大到多少会开始受通信瓶颈限制？

### 8.4 张量并行分析

TP 中，$W_1, W_2$ 按输出维度分片（column parallel），$W_3$ 按输入维度分片（row parallel）。

TP 前向传播：

$$x_1^{(i)} = xW_1^{(i)}, \quad x_2^{(i)} = xW_2^{(i)}, \quad z^{(i)} = f(x_1^{(i)}) * x_2^{(i)}, \quad y^{(i)} = z^{(i)}W_3^{(i)}$$
$$y = \text{all-reduce}(\{y^{(i)}\})$$

其中 $W_1^{(i)}, W_2^{(i)} \in \mathbb{R}^{D \times D_{FF}/N_{TP}}$，$W_3^{(i)} \in \mathbb{R}^{D_{FF}/N_{TP} \times D}$。

> **问题（tp_calcs）：张量并行计算（4 分）**
>
> (a) 写出上述 TP 策略的反向传播方程。
>
> (b) TP 前向/反向传播各需多少 FLOPs？
>
> (c) TP 前向/反向传播各需多少通信时间？
>
> (d) $N_{TP}$ 增大到多少会开始受通信瓶颈限制？

### 8.5 二维并行（FSDP + TP）

结合 FSDP 和 TP：共 $N = N_{TP} \times N_{FSDP}$ 个设备，形成 2D 网格。

每个设备 $(i, j)$ 持有：$W_1^{(i,j)}, W_2^{(i,j)} \in \mathbb{R}^{D/N_{FSDP} \times D_{FF}/N_{TP}}$，$W_3^{(i,j)} \in \mathbb{R}^{D_{FF}/N_{TP} \times D/N_{FSDP}}$。

> **问题（fsdp_tp_calcs）：2D 并行计算（6 分）**
>
> (a) FSDP+TP 前向传播需要多少 FLOPs？
>
> (b) 假设 FSDP 轴和 TP 轴的通信可以重叠，前向传播需要多少通信时间？
>
> (c) 最优 $N_{TP}$ 和 $N_{FSDP}$ 设置下，$N$ 增大到多少会开始受通信瓶颈限制？
>
> (d) 如果 FSDP 和 TP 轴共享网络资源（不能重叠），最优设置下 $N$ 的上限是多少？

---

## 9 排行榜

排行榜测试 8B 模型完整训练步（前向 + 损失 + 反向 + AdamW）的速度，测试配置：

```python
class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2
```

在 2 张 B200 GPU 上测试，期望超过朴素基线 10 秒。

**优化思路：**

- 调整 Triton 核函数的瓦片大小（用 `triton.autotune` 自动调优）
- 实现融合 AdamW
- 融合 LM head 和 cross-entropy 损失
- FlashAttention 改进：
  - Triton 反向传播
  - 因果掩码时提前终止全零块的处理
  - 在 Hopper+ 架构上使用 TMA
- 用激活检查点换内存

> **问题（leaderboard）：排行榜（10 分）**
>
> **交付物：** 完整前向+反向训练步的最佳 wall-clock 时间。提交至 [github.com/stanford-cs336/assignment2-systems-leaderboard](https://github.com/stanford-cs336/assignment2-systems-leaderboard)

---

## 参考文献

1. T. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," arXiv:2307.08691
2. T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022
3. M. Milakov and N. Gimelshein, "Online normalizer calculation for softmax," arXiv:1805.02867
4. H. He, "Making Deep Learning Go Brrrr From First Principles," 2022
5. S. Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," 2020
6. J. Austin et al., "How to Scale Your Model," 2025
7. Nouamane Tazi et al., "The Ultra-Scale Playbook: Training LLMs on GPU Clusters," 2025
